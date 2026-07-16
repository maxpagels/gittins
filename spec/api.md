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

The surface is eight names:

| function | is |
|---|---|
| `create(bits, horizon, default_reward=0.0, epsilon=0.05, forgetting=0.999)` | `new_bandit` with `bits` (the one encoding declaration) in place of a raw dimension: the model spans the 2^bits hashed space |
| `decide(state, context, candidates, t, salt)` | encode each `(arm_id, action dict)` candidate against the context dict, in candidate order, then `decide.py`'s decide |
| `learn(state, decision_id, reward)` | ledger.py, unchanged |
| `censor(state, decision_id)` | ledger.py, unchanged |
| `expire(state, t)` | ledger.py, unchanged |
| `serialize(state)` | state.py, unchanged |
| `deserialize(data)` | state.py, unchanged |
| `model_bits(state)` | the `bits` declaration recovered from the model dimension |

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

## No new state, no new semantics

`create` returns a plain `BanditState`; the serialization format (PR 18)
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
