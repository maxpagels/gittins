# Spec: the public API

Design doc reference: D2 (hashed encoding as the single declaration), D5
(decision records), section 9 (bindings). Implemented in
`src/gittins_reference/api.py` (PR 19); mirrored by the Rust core's
`api.rs`; pinned by `spec/golden.json`, section `api`.

## What it is

The dict-shaped facade every binding exposes — specified once, here in the
reference, mirrored exactly by each binding, and gated in CI by replaying
the golden `api` section through the binding's public surface alone. It is
deliberately a facade: every function is the layered API unchanged, with
the hashed encoding folded inside `decide` so sparse pairs never cross the
public boundary.

The surface is nine names:

| function | returns | is |
|---|---|---|
| `create(bits, horizon, default_reward=0.0, epsilon=0.05, forgetfulness=0.999)` | a state handle | `new_bandit` with `bits` (the one encoding declaration) in place of a raw dimension: the model spans the 2^bits hashed space |
| `decide(state, context, candidates, t, salt, score=None, explore=None)` | the decision record, carrying its inputs (`bits`, `context`, `candidates` — the caller's own objects) | encode each `(arm_id, action dict)` candidate against the context dict, in candidate order, then `decide.py`'s decide; `score`/`explore` are the BYO callbacks (byo.md) |
| `learn(state, decision_id, reward, train=None)` | the resolution, or None/null if it was a no-op | ledger.py, unchanged; with `train`, the BYO training tap replaces the built-in update (byo.md) |
| `censor(state, decision_id)` | the resolution, or None/null if it was a no-op | ledger.py, unchanged |
| `expire(state, t, train=None)` | the resolutions, in ledger order | ledger.py, unchanged; `train` as on `learn`, fired per due record (byo.md) |
| `serialize(state)` | the state as one hex string | the canonical byte layout (serialization.md), hex-encoded |
| `deserialize(data)` | a state handle | the inverse; rejects anything malformed |
| `model_bits(state)` | the `bits` declaration | recovered from the model dimension |
| `log_line(record \| resolution)` | one experience-log line (ope.md) | canonical compact JSON of exactly the event `verify`/`evaluate`/`replay` consume; logging is appending what each call returns |

## The record carries its inputs; the ledger does not

The record `decide` returns carries `bits` and the very `context` and
`candidates` passed in (no copies) — attached at the only moment they
exist, so a record logs as-is via `log_line`. The ledger keeps only the
compact record: fixed memory, unchanged serialization. A record that
crossed the state boundary — the one a `train` callback receives —
therefore carries None/null in those three fields, and `log_line`
refuses it (its inputs are already in the log, written at decide time).
`candidate_hash` is the commitment binding a record to its logged
inputs; `verify` (ope.md) enforces it.

`log_line` is implemented at each app-facing surface (the reference,
the wheel, the browser module) because it serializes the caller's own
objects; the core's `api` module carries the other eight names. Its
output is compact single-line JSON in canonical field order, feature
dicts kept in their own insertion order (the order they encoded with);
number rendering is each host's shortest round-trip, so lines from
different hosts are parse-equivalent, byte-equal in the common case.

## State handling: one convention everywhere

The state is an **opaque handle, updated in place**: `create` and
`deserialize` return one, `decide`/`learn`/`censor`/`expire` mutate it and
return only their results. Every implementation — the reference, the
Python wheel, the browser module — uses exactly this shape, so code and
docs are interchangeable across them. (The convention is set by the least
common denominator: a JS binding cannot idiomatically return
`(result, state)` tuples, so nothing does.)

This is purely a facade choice. The reference's internal layers
(`decide.py`, `ledger.py`, `state.py`) remain pure functions over
immutable values; `api.py`'s handle is a one-field cell that swaps which
immutable value it holds. Snapshotting, rollback, and replay are one
`serialize` away.

## Serialization is a plain string

`serialize` returns the canonical byte layout (serialization.md) as a
lowercase hex string, and `deserialize` takes one back (either case
accepted; the canonical form is lowercase). A string goes anywhere text
goes — a file, a database column, localStorage, version control — so no
caller, in any language, ever handles raw bytes, base64, or `Uint8Array`
conversions. The bytes remain the specified format; hex is its one public
text form, identical across implementations (the golden `api` section's
`state_hex` is exactly what `serialize` returns).

## Inputs

- `context` is one feature dict; each candidate is an `(arm_id, action
  feature dict)` pair. Feature values follow encoding.py's contract:
  strings are categorical tokens, numbers (bools included) are numeric
  contributions, `None` means absent.
- Duplicate candidates are allowed and score identically, exactly as
  duplicate encodings do in the layered API.
- `bits` must be in [1, 24] (`create`). `decide` recovers it from the
  model dimension: states built by `create` always satisfy dim == 2^bits;
  a state whose dimension is not a power of two in that range was built
  against the layered API and is rejected with ValueError (the core's
  `Error`).

## Bring your own model / exploration

The three optional callback parameters — `score` and `explore` on
`decide`, `train` on `learn`/`expire` — are the whole BYO surface (R7),
specified in `byo.md` and pinned by the golden `byo` section. They are
per-call values, never stored; passed as `None`/omitted, every function
is the pre-existing path bit for bit.

## No new state format, no new semantics

The handle wraps a plain `BanditState`; the serialization format (PR 18)
is unchanged, and `bits` is recoverable rather than stored. The facade
adds no randomness, no reordering, and no arithmetic: `decide` through the
facade is bit-identical to `encode` + `decide` by hand, which the test
suites pin on both sides.

## The golden `api` section

One compact scenario driven purely through the public surface: `create`
at bits 4, four decides over one three-arm catalog (action features
included — `price` numeric, `trial` boolean — which the episode section
never exercises), out-of-order rewards, a censor, an exact-horizon expiry,
and the final state's canonical bytes as hex. A binding that reproduces
this section — final hex included — through its own public API alone is
accepted.
