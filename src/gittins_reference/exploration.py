"""The exploration rule: epsilon-greedy (design doc section 5, revised).

Given one reward estimate per candidate, epsilon-greedy turns the estimates
into a probability distribution: spend `epsilon` of the probability mass
uniformly over all k candidates, and the remaining `1 - epsilon` on the
greedy choice,

    p[i] = epsilon / k  +  (1 - epsilon) / m    if estimate[i] is maximal
    p[i] = epsilon / k                          otherwise

where m is the number of candidates tied (exactly, in float equality) for
the maximal estimate. Splitting the greedy mass across ties instead of
picking the first maximum matters at cold start: a fresh model estimates
every candidate at 0.0, so the first distributions are uniform rather than
hammering index 0. This rule needs nothing from the model except reward
estimates — no uncertainties, no schedule, no per-candidate divisions.

One constant does three jobs: the system never fully converges,
importance weights in offline evaluation stay bounded (by k / epsilon), and
every arm keeps accumulating fresh evidence, so a fallen-then-recovered arm
is always rediscovered. `DEFAULT_EPSILON` is the engine default; overriding
it is an expert move (decide.py), not routine tuning.

Epsilon-greedy replaced inverse-gap weighting (SquareCB) here. SquareCB's
regret guarantee is forfeited the moment a permanent exploration floor is
mixed in — and this engine demands that floor — so the two rules differ only in how
they spend the transient exploration, at very different prices: SquareCB
needs per-candidate uncertainties (twice the prediction arithmetic plus a
sqrt), a gamma schedule with a swept constant, and a full distribution
build; epsilon-greedy needs one max-scan. The battery (sim/) prices the
regret difference; the sweep driver (sim/sweep.py) settles the epsilon
default by evidence, exactly as it settled the old gamma scale.

Sampling is inverse-CDF: one uniform draw from the counter-based RNG walks
the cumulative sum in index order. Every step everywhere in this module is
IEEE-754 add/multiply/divide in a fixed order, so the distribution and the
choice are bit-identical across platforms.
"""

from gittins_reference.rng import random_unit

# Probability mass spent uniformly over the candidates; every candidate is
# guaranteed at least DEFAULT_EPSILON / k. The engine default, settled by
# evidence (sim/sweep.py); an expert override at construction, not a knob
# to routinely tune.
DEFAULT_EPSILON = 0.05


def epsilon_greedy_probabilities(estimates: list[float], epsilon: float) -> list[float]:
    """The epsilon-greedy distribution over candidates: `epsilon` uniform,
    `1 - epsilon` split across the maximal estimates (exact-tie split)."""
    k = len(estimates)
    if k < 1:
        raise ValueError("need at least one candidate")
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon must be in [0, 1]")
    best = estimates[0]
    for i in range(1, k):
        if estimates[i] > best:
            best = estimates[i]
    m = 0
    for i in range(k):
        if estimates[i] == best:
            m += 1
    uniform = epsilon / k
    share = (1.0 - epsilon) / m
    return [uniform + share if estimates[i] == best else uniform for i in range(k)]


def sample_index(p: list[float], key: int, counter: int) -> int:
    """Draw one index from distribution `p` using the uniform value at
    position `counter` of RNG stream `key` (inverse CDF, fixed index order)."""
    u = random_unit(key, counter)
    cumulative = 0.0
    for i in range(len(p)):
        cumulative += p[i]
        if u < cumulative:
            return i
    return len(p) - 1  # u landed in the rounding gap above the final sum


def choose(estimates: list[float], epsilon: float, key: int, counter: int) -> tuple[int, float]:
    """The full exploration rule: (chosen index, its propensity).

    The propensity is the probability the index was chosen with — exactly
    what the decision record must log (D5).
    """
    p = epsilon_greedy_probabilities(estimates, epsilon)
    i = sample_index(p, key, counter)
    return i, p[i]
