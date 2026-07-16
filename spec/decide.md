# Spec: decision records and the decide layer

Design doc reference: D1 (the core is a plain function with no hidden
state), D5 (decisions are first-class records; propensities are always
logged), section 5 (the layer stack). Implemented in
`src/gittins_reference/decide.py` (PR 6), on top of the model
(`spec/model.md`), the exploration rule (`spec/exploration.md`), and the
counter RNG (`spec/rng.md`).

## State

A `BanditState` is `(model, next_seq, model_version, horizon,
default_reward, ledger)`:

- `model` — the LinearModel of `spec/model.md`.
- `next_seq` — the sequence number of the next decision.
- `model_version` — how many observations the model has absorbed. Bumped
  by the ledger's trained resolutions (`spec/ledger.md`); logged in every
  record so offline evaluation can tell which policy made each decision.
- `horizon`, `default_reward` — the application's reward-handling
  declaration, made once at creation (`spec/ledger.md`).
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
| `propensity` | the floored probability the choice was made with |
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

1. `(estimate, uncertainty) = predict(model, x, t)` for every candidate,
   in list order.
2. `gamma = choose_gamma(uncertainties)` — see below.
3. `p = apply_floor(inverse_gap_probabilities(estimates, gamma))`.
4. `key = derive_key(decision_id, salt)`; the choice is
   `sample_index(p, key, 0)`. **Counter 0 of a decision's RNG stream is
   reserved for the sampling draw**; later counters are reserved for future
   per-decision randomness.
5. The model is untouched; the new state is the old state with
   `next_seq + 1` and the record appended to the ledger.

Steps 1–4 are fixed-order IEEE-754 arithmetic over already-deterministic
layers, so the whole record is bit-identical across platforms.

## The gamma schedule

    gamma = GAMMA_SCALE / mean(uncertainty over the candidates)

with `GAMMA_SCALE = 300.0` a fixed engine constant. If the mean is 0
(every candidate a zero vector), gamma is 0 (uniform).

Why this is the right shape: with n effective observations uncertainty
shrinks like 1/sqrt(n), so gamma grows like sqrt(n), which is the schedule
SquareCB's regret guarantee wants — obtained from the model's own
posterior instead of a clock or a knob. A fresh model estimates every
candidate at 0, so the first distributions are uniform whatever gamma is,
and the probability floor guarantees exploration ever after. Because
decaying sums bound the effective sample size, gamma is bounded (never
fully greedy, R2), and after a world shift uncertainty regrows and gamma
falls back automatically. The constant's value was settled by the battery
sweep (`sim/sweep.py`): 300 was within 1.1x of the best swept value on 9
of 12 environments, never worse than 1.34x, and ahead of epsilon-greedy
overall; the original provisional 1.0 kept gamma so low the engine spent
over half its traffic on non-best arms indefinitely. The choice of mean as
the uncertainty aggregate remains provisional; the interface — gamma is
computed inside the engine, never asked of the user — is settled.

## Golden vectors

Model dim 2, half-life 3600 s, created at T0 = 1752000000.0; updates
`([1,0], 1.0, T0)` and `([0,1], -0.5, T0+1800)`. State at `next_seq` 7,
`model_version` 2. Candidates `[[1,0], [0,1], [1,1]]`, decided at T0+3600
with salt `"pepper"`:

    decision_id    = "pepper:7"
    candidate_hash = 8340395383735871362
    chosen         = 0
    features       = (1.0, 0.0)
    propensity     = 0.9482851123609886
