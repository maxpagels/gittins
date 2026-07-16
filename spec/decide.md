# Spec: decision records and the decide layer

Design doc reference: D1 (the core is a plain function with no hidden
state), D5 (decisions are first-class records; propensities are always
logged), section 5 (the layer stack). Implemented in
`src/gittins_reference/decide.py` (PR 6), on top of the model
(`spec/model.md`), the exploration rule (`spec/exploration.md`), and the
counter RNG (`spec/rng.md`).

## State

A `BanditState` is `(model, next_seq, model_version, horizon,
default_reward, epsilon, ledger)`:

- `model` — the LinearModel of `spec/model.md`.
- `next_seq` — the sequence number of the next decision.
- `model_version` — how many observations the model has absorbed. Bumped
  by the ledger's trained resolutions (`spec/ledger.md`); logged in every
  record so offline evaluation can tell which policy made each decision.
- `horizon`, `default_reward` — the application's reward-handling
  declaration, made once at creation (`spec/ledger.md`).
- `epsilon` — the uniform exploration mass every decision spends
  (`spec/exploration.md`), declared once at creation; defaults to
  `DEFAULT_EPSILON = 0.05`, an expert override rather than a routine knob.
- `ledger` — the open decision records, in decision order
  (`spec/ledger.md`).

## The decision record

`decide` never returns a bare action. Its result is a `DecisionRecord`:

| field | meaning |
|---|---|
| `decision_id` | `"{salt}:{seq}"` — see uniqueness below |
| `t` | the caller-supplied decision time |
| `candidate_hash` | 64-bit hash of the entire candidate set |
| `chosen` | index into the candidate list |
| `features` | the chosen candidate's feature vector |
| `propensity` | the probability the choice was made with |
| `model_version` | the state's `model_version` at decision time |
| `salt` | the RNG salt, making the draw exactly replayable |

The record is self-contained: `learn` (PR 7) needs nothing but a record
and an outcome, which is what makes hand-assembled training data
unrepresentable (R4), and the append-only record log doubles as the OPE
dataset with no extra instrumentation (R3).

**ID uniqueness is structural.** Within one agent the sequence counter
never repeats; across a fleet, each agent must be given its own salt — the
same rule that already keeps their random streams distinct. No hashing, no
collision probability to reason about.

**Candidate-set hash.** FNV-1a (64-bit, `spec/rng.md`) over a canonical
*sparse* encoding: the candidate count as 8 little-endian bytes, then each
vector as its dimension (8 LE bytes), its nonzero-entry count (8 LE bytes),
and its nonzero entries in increasing index order, each as the index (8 LE
bytes) followed by the value (little-endian IEEE-754 double). Entries equal
to zero — either sign — are absent by definition, so the hash costs
O(nonzeros), not O(dimension): hash-encoded candidates are nearly all
zeros, and an implementation that keeps candidates as sparse (index, value)
pairs never materializes the dense vector to hash it. Order- and
value-sensitive; the dimension and count prefixes keep differently-shaped
sets with equal flattenings distinct.

## The decide pipeline

`decide(state, candidates, t, salt)`, where each candidate is a feature
vector in the model's space (folding context + action description into
that vector is the feature encoder's job, PR 8):

1. `estimate = estimate_factored(f, x)` for every candidate, in list order
   (`spec/model.md`); the weights are solved once per decision and shared.
   No uncertainties are computed anywhere on the decision path.
2. `p = epsilon_greedy_probabilities(estimates, state.epsilon)`
   (`spec/exploration.md`).
3. `key = derive_key(decision_id, salt)`; the choice is
   `sample_index(p, key, 0)`. **Counter 0 of a decision's RNG stream is
   reserved for the sampling draw**; later counters are reserved for future
   per-decision randomness.
4. The model is untouched; the new state is the old state with
   `next_seq + 1` and the record appended to the ledger.

Steps 1–3 are fixed-order IEEE-754 arithmetic over already-deterministic
layers, so the whole record is bit-identical across platforms.

## Where epsilon comes from

`epsilon` is state, not schedule: declared once at `new_bandit`, spent
identically on every decision. A fresh model estimates every candidate at
0.0 and the exact-tie split makes the first distributions uniform; the
epsilon mass guarantees exploration — and re-exploration after a world
shift, alongside the model's forgetting — ever after (R2). There is no
uncertainty aggregate, no gamma schedule, and no swept scale constant;
the epsilon default itself is settled by the battery sweep
(`sim/sweep.py`), like every engine constant.

## Golden vectors

Model dim 2, default forgetting (0.999); updates `([1,0], 1.0)` then
`([0,1], -0.5)`. State at `next_seq` 7, `model_version` 2. Candidates
`[[1,0], [0,1], [1,1]]`, decided at T0+3600 = 1752003600.0 with salt
`"pepper"`:

    decision_id    = "pepper:7"
    candidate_hash = 8340395383735871362
    chosen         = 0
    features       = (1.0, 0.0)
    propensity     = 0.9666666666666667
