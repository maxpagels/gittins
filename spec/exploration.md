# Spec: the exploration rule

Design doc reference: section 5 (inverse-gap weighting with a probability
floor), R2 (never fully converges), R3 (bounded importance weights), R7
(needs only reward estimates). Implemented in
`src/gittins_reference/exploration.py` (PR 5), consuming the counter-based
RNG of `spec/rng.md`.

## Inverse-gap weighting (SquareCB)

Input: one reward estimate per candidate (any floats), and a greediness
`gamma >= 0`. With `k` candidates and `best` the index of the *first*
maximum estimate (ties broken by index order, deterministically):

    p[i]    = 1 / (k + gamma * (estimate[best] - estimate[i]))   for i != best
    p[best] = 1 - sum of the others

Properties, all structural:

- Every gap is >= 0, so each non-best probability is <= 1/k, so the best
  candidate's remainder is >= 1/k > 0: the result is always a valid
  distribution, and it sums to 1.0 *exactly* in float arithmetic (the best
  entry is defined as the complement).
- `gamma = 0` gives the uniform distribution; `gamma -> infinity` gives
  argmax. A candidate tied with the best gets exactly `1/k`.
- Nothing is required from the model layer except the estimates (R7).

`gamma` is a parameter of this layer, not a user knob: the decide layer
(PR 6) supplies it, and scheduling/self-tuning it (design doc D4) is that
layer's concern. This keeps the rule a pure, replayable function.

## The probability floor

    p'[i] = FLOOR_MASS / k  +  (1 - FLOOR_MASS) * p[i]

with `FLOOR_MASS = 0.05`, a fixed engine constant (5% of the probability
mass is always spent uniformly). Guarantees, for any input distribution:

- `p'[i] >= FLOOR_MASS / k` for every candidate, so offline-evaluation
  importance weights are bounded by `k / FLOOR_MASS` (R3), and every arm
  keeps accumulating evidence forever (R2).
- The map is affine with positive slope `1 - FLOOR_MASS`: it preserves the
  candidates' probability order (the greedy choice is unchanged) and total
  mass.

## Sampling

One uniform draw `u = random_unit(key, counter)` (see `spec/rng.md`) walks
the cumulative sum of the floored distribution in index order; the first
index whose cumulative sum exceeds `u` is chosen (inverse CDF). If rounding
leaves the final cumulative sum below `u`, the last index is chosen. Every
operation in this module is IEEE-754 add/multiply/divide in a fixed order,
so both the distribution and the choice are bit-identical across platforms.

`choose(estimates, gamma, key, counter)` composes the three steps and
returns `(chosen index, its floored probability)` — the propensity that the
decision record must log (D5).

## Golden vectors

Estimates `[0.5, -0.25, 0.0, 0.5]`, gamma 10, `FLOOR_MASS` 0.05, key
derived from `("decision-0001", "pepper")`. Floored distribution:

    [0.5368357487922705, 0.0951086956521739, 0.11805555555555554, 0.25]

Choices at counters 0, 1, 2:

    (0, 0.5368357487922705), (1, 0.0951086956521739), (0, 0.5368357487922705)
