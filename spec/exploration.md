# Spec: the exploration rule

Design doc reference: section 5 (epsilon-greedy, revised from inverse-gap
weighting), R2 (never fully converges), R3 (bounded importance weights),
R7 (needs only reward estimates). Implemented in
`src/gittins_reference/exploration.py`, consuming the counter-based RNG of
`spec/rng.md`.

## Epsilon-greedy

Input: one reward estimate per candidate (any floats), and a uniform
exploration mass `epsilon` in `[0, 1]`. With `k` candidates and `m` the
number of candidates whose estimate equals the maximum exactly (IEEE-754
equality):

    p[i] = epsilon / k  +  (1 - epsilon) / m    if estimate[i] is maximal
    p[i] = epsilon / k                          otherwise

Properties, all structural:

- Every candidate keeps at least `epsilon / k`, so offline-evaluation
  importance weights are bounded by `k / epsilon` (R3) and every arm keeps
  accumulating evidence forever (R2).
- **Ties split the greedy mass** rather than falling to the first index.
  This is the cold-start rule: a fresh model estimates every candidate at
  exactly 0.0, so the first distributions are uniform instead of a
  `1 - epsilon` lock on index 0. The split is deterministic (exact float
  equality, index order), no tie-break draw.
- `epsilon = 1` gives the uniform distribution; `epsilon = 0` gives pure
  (tie-split) argmax.
- Nothing is required from the model layer except the estimates (R7): no
  uncertainties, no schedule, no per-candidate division.

`epsilon` is a parameter of this layer, not a scheduling concern: the
decide layer supplies it from the state, where it was declared once at
construction (default `DEFAULT_EPSILON = 0.05`, an expert override rather
than a routine knob). This keeps the rule a pure, replayable function.

**Why not SquareCB.** The previous rule was inverse-gap weighting with a
5% uniform floor. The floor R2 demands already forfeits SquareCB's regret
guarantee, so the rules differ only in how transient exploration is spent
— and at very different prices: SquareCB needs per-candidate uncertainties
(double the prediction arithmetic plus a sqrt each), a swept gamma-schedule
constant, and a full distribution build. The battery (sim/) prices the
regret difference; the sweep driver (`sim/sweep.py`) settles the epsilon
default by evidence, as it settled the old gamma scale.

## Sampling

One uniform draw `u = random_unit(key, counter)` (see `spec/rng.md`) walks
the cumulative sum of the distribution in index order; the first index
whose cumulative sum exceeds `u` is chosen (inverse CDF). If rounding
leaves the final cumulative sum below `u`, the last index is chosen. Every
operation in this module is IEEE-754 add/multiply/divide in a fixed order,
so both the distribution and the choice are bit-identical across platforms.

`choose(estimates, epsilon, key, counter)` composes the two steps and
returns `(chosen index, its probability)` — the propensity that the
decision record must log (D5).

## Golden vectors

Estimates `[0.5, -0.25, 0.0, 0.5]` (a two-way tie for the maximum,
deliberately), epsilon 0.05, key derived from `("decision-0001",
"pepper")`. Distribution:

    [0.4875, 0.0125, 0.0125, 0.4875]

Choices at counters 0, 1, 2:

    (0, 0.4875), (3, 0.4875), (0, 0.4875)
