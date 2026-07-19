# Spec: the experience log and offline policy evaluation

Design doc reference: R3 (offline policy evaluation), R8d (fleet pooling:
logs merge, models rebuild offline). Implemented in
`src/gittins_reference/ope.py`; mirrored by the CLI binding
(`bindings/cli`, the `gittins` binary); pinned by `spec/golden.json`,
section `ope`.

## The experience log

The append-only stream of what the engine already returns — decision
records and resolutions — plus the dict-shaped inputs each decision was
made over. One JSON object per line (JSONL), in **arrival order**; order
is load-bearing, because training is order-dependent (ledger.md). Files
may be gzip-compressed; readers detect the gzip magic bytes
(`1f 8b`), not the file name. Empty lines are ignored; unknown object
fields are ignored (forward compatibility).

Reading is **incremental by contract**: one event per line means a log
is processed line by line — plain files read as they stream, gzipped
files fed through a streaming decompressor — and `verify`, `evaluate`,
and `replay` are single-pass walks holding only their own bookkeeping
(seen ids, open decisions), so a log's size is bounded by disk, never
by memory. Events are compact, one JSON object per line; readers never
need the whole file at once. The reference exposes this as generators
(`read_log`/`parse_log`); the CLI as a line iterator.

Two event kinds:

```jsonl
{"event": "decision", "bits": 8, "context": {...}, "candidates": [["arm", {...}], ...], "record": {...}}
{"event": "resolution", "decision_id": "...", "kind": "rewarded", "reward": 1.0}
```

- `bits` is the logging agent's encoding declaration (`model_bits`),
  required so the hashed encoding — and with it `candidate_hash` and the
  record's `features` — can be reproduced exactly. Per event, so merged
  logs may mix agents.
- `context` and `candidates` are the exact dict-shaped inputs passed to
  `decide` (api.md's shapes). Logging the full candidate set is what
  makes counterfactual evaluation possible at all: the record alone
  carries only the chosen candidate.
- `record` is the decision record verbatim, with `features` as
  `[[index, value], ...]`.
- `kind` is `rewarded` | `expired` | `censored`; `reward` is the value
  the resolution reported — `null` for censored, the logging agent's
  `default_reward` for expired.

The app writes a decision event after each `decide` and a resolution
event after each non-null `learn`/`censor` result and each `expire`
resolution, in the order the engine returned them. Merging a fleet's
logs is file concatenation or timestamp interleaving; the semantics
below never assume a single agent.

Parsing is strict: a line that is not valid JSON, or an object that is
not one of the two shapes above, is a hard error naming the line
(`line N: not valid JSON`, `line N: each event must be a JSON object
with an 'event' kind`, `line N: unknown event kind`,
`line N: malformed decision event`, `line N: malformed resolution
event`). Semantic problems are `verify`'s job, below.

## `verify`

The R4-flavored guard: recompute what the log claims and collect every
violation (never stopping at the first) as `line N: ...` strings, in
event order. An empty result means the log is trustworthy input for
`evaluate` and `replay`. The checks, and their exact messages after the
`line N: ` prefix:

- `duplicate decision id '{id}'`
- `decision {id}: bits must be between 1 and 24`
- `decision {id}: chosen index out of range`
- `decision {id}: propensity must be in (0, 1]`
- `decision {id}: candidates failed to encode` — the dict-shaped inputs
  violate encoding.md's value contract
- `decision {id}: candidates do not match the logged candidate_hash` —
  the candidate set re-encoded at `bits` no longer hashes to
  `record.candidate_hash`; the inputs were tampered with or do not
  belong to this record
- `decision {id}: chosen features do not match the logged candidates` —
  the re-encoded chosen candidate differs (bit-exactly) from
  `record.features`
- `decision {id}: t decreases within salt '{salt}'` — decision times
  must be nondecreasing per salt
- `resolution for unknown decision '{id}'`
- `second resolution for decision '{id}'` — resolutions are
  exactly-once, in the log as in the ledger
- `resolution for {id}: unknown kind '{kind}'`
- `resolution for {id}: censored resolutions carry no reward`
- `resolution for {id}: reward must be a finite number`

Decisions with no resolution by end of log are normal (open at capture
time), not a violation.

## `evaluate` — progressive IPS/SNIPS

Estimates how a **candidate configuration** of the built-in engine
(`bits`, `epsilon`, `forgetfulness`; ridge fixed as ever) would have
performed on the logged traffic. The walk is progressive: the candidate
policy is always evaluated with the model it would actually have had at
that moment, then trained at the resolution's logged position — the
right default under non-stationarity.

In event order:

- **decision**: re-encode the logged candidates at the candidate
  `bits`; score them with the candidate model's current state; build
  the epsilon-greedy distribution at the candidate `epsilon`; take
  `q = p[chosen]` and the importance weight `w = q / record.propensity`.
  The pair (chosen candidate's encoding, `w`) waits for the resolution.
  (Evaluate is strict where verify is lenient: empty candidates, a
  chosen index out of range, or a propensity outside (0, 1] raise.)
- **resolution**: `rewarded` and `expired` resolutions contribute their
  logged reward `r` — in this order of accumulation:
  `sum_r += r`, `sum_w += w`, `sum_wr += w * r`, `sum_w2 += w * w` —
  and then train the candidate model on (encoding, `r`). `censored`
  resolutions are excluded from both the estimate and training, but
  counted. Resolutions for unknown ids are skipped.

The report, over the `resolved` (non-censored) resolutions:

- `logged_mean = sum_r / resolved` — the logging policy's realized mean
  reward, the baseline
- `ips = sum_wr / resolved`
- `snips = sum_wr / sum_w` (null if `sum_w` is 0)
- `ess = (sum_w * sum_w) / sum_w2` — effective sample size (null if
  `sum_w2` is 0)
- `max_weight` — the largest single importance weight
- counts: `decisions`, `resolved`, `censored`, `unresolved`

All null when `resolved` is 0. No clipping and no clipping knob:
epsilon-greedy logging already bounds every weight structurally at
`k / epsilon`. The diagnostics (`ess`, `max_weight`) are printed next to
every estimate because an IPS number without them is a trap; the CLI
never shows one without the other.

## `replay`

The fleet-pooling rebuild (R8d): the same walk, no estimator — train a
fresh model on every rewarded/expired resolution's (re-encoded chosen
candidate, logged reward), in log order, and emit a complete deployable
state: the trained model, `model_version` = observations absorbed,
`next_seq` 0, empty ledger, and the caller's declared `horizon`,
`default_reward`, `epsilon`, `forgetfulness` for future behavior.
Expired resolutions train with their *logged* reward (the logging
agent's `default_reward` at the time); the config's `default_reward`
only governs the rebuilt state's future expiries. The output is the
canonical hex string (serialization.md) — ready to check in and deploy.

A replay at the logging agent's own configuration reproduces that
agent's model **bit for bit** (same update sequence, same order); the
reference test suite pins this, and the corollary — evaluating the
logging configuration yields `w = 1` everywhere, so
`ips = snips = logged_mean` and `ess = resolved` exactly — is pinned in
the golden section as a built-in sanity check every implementation must
reproduce.

## Determinism and the golden `ope` section

Same log + same configuration ⇒ bit-identical reports and bit-identical
replay output, on every platform: every accumulation runs in event
order, every division in the order written above, all in IEEE-754
doubles.

Two traps every implementation must respect, both found the hard way:
JSON *number parsing* must be correctly rounded (serde_json needs its
`float_roundtrip` feature; its default parse can be one ulp off, which
is one ulp more than a bit-identical engine tolerates), and JSON
*object key order* must be preserved end to end (serde_json's
`preserve_order`), because feature iteration order is semantic under
hashed encoding — per-slot accumulation under collisions is
order-sensitive at the last bit. The golden log's feature dicts are
deliberately written in canonical (sorted) key order so the corpus's
sorted rendering cannot reorder them. The golden section carries one small api-driven log (two
candidate sets, out-of-order rewards, a censor, an exact-horizon
expiry), its clean `verify`, evaluation reports for the logging
configuration and two others, the replay hex, and one tampered copy of
the log with its expected `verify` findings.

## The CLI binding

`bindings/cli` ships the `gittins` binary over these semantics:

```
gittins verify --log FILE[.gz]
gittins eval   --log FILE --bits N [--epsilon X] [--forgetfulness X]
gittins sweep  --log FILE --bits N[,N...] [--epsilon X[,X...]] [--forgetfulness X[,X...]]
gittins replay --log FILE --bits N --horizon S [--default-reward X] [--epsilon X] [--forgetfulness X] [> new.state]
```

`verify` exits nonzero when it finds problems; `eval` prints the report
(estimates with diagnostics, never one without the other); `sweep` is
`eval` over the config grid as one markdown table; `replay` writes the
state hex to stdout. The CLI evaluates built-in configurations — BYO
policies (byo.md) are evaluated through the bindings, where callbacks
exist. Presentation (flags, tables, exit codes) is the CLI's own; the
semantics are this spec's, gated by the golden section like every other
implementation surface.
