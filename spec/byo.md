# Spec: bring your own model and exploration

Design doc reference: R7 (simple algorithms, bring-your-own models).
Implemented in `src/gittins_reference/decide.py` + `api.py`; mirrored by
the Rust core's `decide.rs` + `api.rs`; pinned by `spec/golden.json`,
section `byo`.

## What it is

Three optional callbacks on the public API, each replacing exactly one
built-in component while everything else — hashed encoding, the counter
RNG draw, the decision record, propensity logging, the ledger's
exactly-once join, offline evaluation — is inherited unchanged:

| callback | on | replaces | signature |
|---|---|---|---|
| `score` | `decide` | the built-in model's reward estimates | `score(context, candidates) -> k numbers` |
| `explore` | `decide` | epsilon-greedy | `explore(estimates, epsilon) -> k probabilities` |
| `train` | `learn`, `expire` | the built-in model's update | `train(record, reward)` |

A **BYO model** is `score` + `train`; a **BYO exploration rule** is
`explore`. Any subset may be used, per call.

Callbacks are per-call values, never stored: the state stays plain data,
the serialization format is untouched, and a state driven by callbacks
round-trips exactly like any other. When no callback is passed, every
code path is the pre-existing one, bit for bit.

## `score`

Called exactly once per decision, after candidate encoding, with **the
very objects the caller passed** to `decide` (the context dict and the
`(arm_id, action)` candidate list — not the encoded pairs). Must return
one reward estimate per candidate, in candidate order.

Validation, in every implementation, message-for-message:

- not a sequence of numbers, wrong length, or any non-finite entry →
  `score must return one finite estimate per candidate` (ValueError /
  the core's `Error` / a thrown JS error).

The estimates feed whatever exploration rule is in force (built-in
epsilon-greedy, or `explore`). The built-in model is not consulted.

*(At the reference's layered level, `decide.py`'s `score` receives the
encoded candidates instead — the only inputs that layer has. The public
dict-shaped signature above is the specified, binding-visible one.)*

## `explore`

Called exactly once per decision with the estimates list (the built-in
model's, or `score`'s output coerced to floats) and the state's declared
`epsilon`. Must return one probability per candidate, in candidate
order. The engine — not the callback — then draws from the returned
distribution by inverse CDF in index order over counter 0 of the
decision's RNG stream, exactly as it draws from the built-in
distribution. So a BYO decision is exactly as deterministic and
replayable, and the logged `propensity` is the returned distribution's
value at the chosen index — genuine, bounded-by-construction OPE data.

Validation, in order:

- wrong length or not a sequence →
  `explore must return one probability per candidate`
- any entry non-finite or negative, or the index-order sum farther than
  `PROBABILITY_TOLERANCE = 1e-9` from 1.0 →
  `explore probabilities must be finite, nonnegative, and sum to 1`

The tolerance is generous against benign rounding (the built-in rule
lands within a few ulps of 1) and unforgiving of real errors — a wrong
sum poisons every logged propensity. The distribution is used exactly as
returned, never renormalized.

A callback (`score` or `explore`) that raises propagates to the caller
before any state change: the decision counter has not advanced and
nothing joined the ledger.

## `train`

The BYO model's training tap: the engine owns the join between decisions
and rewards (ledger.md), and `train` is how a user model receives it.
When `learn` (or `expire`, for each due record) resolves a decision and
`train` was passed:

- the built-in model is **not** updated;
- `model_version` still advances — it counts observations absorbed by
  whichever model is in force, so records keep pinning model staleness;
- the resolution commits **first** (the record leaves the ledger), then
  `train(record, reward)` fires — the full `DecisionRecord` (so the user
  model can key on `decision_id` or train on the encoded `features`) and
  the reward (`learn`'s reward, or `default_reward` on expiry).

Commit-then-fire is the safety rule: `train` fires at most once per
decision, ever. A raising callback loses that one training example
loudly (the exception propagates; the resolution stands), but can never
cause a double-train — the silent-corruption direction is
unrepresentable, in the same spirit as the ledger itself. Within one
`expire` sweep, records commit and fire one at a time in ledger order,
so an exception leaves earlier records resolved and later ones still
open for the next sweep.

`censor` never trains and takes no callback. Mixing per deployment is
legal but on the caller: a `learn` without `train` still updates the
built-in model (the golden `byo` scenario deliberately pins one such
mixed call).

## The golden `byo` section

One compact scenario through the public surface with the two
spec-defined callbacks, replayed by every implementation with the
callbacks written natively in its own language:

- `score(context, candidates)`:
  `base = 1.0 if context["seg"] == "a" else 0.0`, then per candidate `i`
  the estimate is `(base + (0.5 * price)) - i` in exactly that
  association order, where `price` is the candidate's `price` action
  feature, `0.0` if absent, and `i` is the candidate index as a double.
- `explore(estimates, epsilon)`: ignores both inputs;
  `p[i] = (i + 1) / (k * (k + 1) / 2)` where the denominator
  `k * (k + 1) / 2` is an exact integer and the division is one double
  division of `(i + 1)` by it.
- `train(record, reward)`: appends
  `(decision_id, features, reward)` to the `trained` list the section
  pins — the exact observation sequence, in resolution order.

Five decides over one catalog — score only, explore only, both, neither,
score again — then out-of-order rewards through `train`, one plain
`learn` (built-in update, deliberately mixed), a censor, and an
exact-horizon expiry with `train`. The section pins every record
(choices and propensities under the BYO distribution), every resolution,
the `trained` sequence, and the final state hex — which proves `train`
replaced the built-in update everywhere except the one plain `learn`.
