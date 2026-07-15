"""The exploration rule: inverse-gap weighting with a probability floor (design
doc section 5).

Given one reward estimate per candidate, inverse-gap weighting (the SquareCB
algorithm) turns the estimates into a probability distribution: the best
candidate gets most of the probability, and every other candidate gets
probability inversely proportional to how far its estimate falls below the
best,

    p[i]    = 1 / (k + gamma * (best_estimate - estimate[i]))    for i != best
    p[best] = 1 - sum of the others

where k is the number of candidates and `gamma` is the greediness: gamma = 0
is a uniform distribution, gamma -> infinity is pure argmax. Each non-best
probability is at most 1/k, so the best candidate always keeps at least 1/k.
This rule needs nothing from the model except reward estimates (R7), and it
comes with the SquareCB regret guarantee.

The floor then mixes in a fixed sliver of the uniform distribution,

    p[i] = FLOOR_MASS / k  +  (1 - FLOOR_MASS) * p[i]

so no candidate's probability is ever below FLOOR_MASS / k. One constant does
three jobs (R2, R3): the system never fully converges, importance weights in
offline evaluation stay bounded (by k / FLOOR_MASS), and every arm keeps
accumulating fresh evidence, so a fallen-then-recovered arm is always
rediscovered. FLOOR_MASS is a fixed engine constant, not a knob.

Sampling is inverse-CDF: one uniform draw from the counter-based RNG walks
the cumulative sum in index order. Every step everywhere in this module is
IEEE-754 add/multiply/divide in a fixed order, so the distribution and the
choice are bit-identical across platforms.

`gamma` is supplied by the layer above (the decide layer, PR 6): scheduling
or self-tuning it is that layer's concern, keeping this rule a pure function.
"""

from gittins_reference.rng import random_unit

# Total probability mass reserved for uniform exploration; each of k
# candidates is guaranteed at least FLOOR_MASS / k.
FLOOR_MASS = 0.05


def inverse_gap_probabilities(estimates: list[float], gamma: float) -> list[float]:
    """The SquareCB distribution over candidates, before the floor."""
    k = len(estimates)
    if k < 1:
        raise ValueError("need at least one candidate")
    if not (gamma >= 0.0):
        raise ValueError("gamma must be non-negative")
    best = 0
    for i in range(1, k):
        if estimates[i] > estimates[best]:
            best = i
    p = [0.0] * k
    total = 0.0
    for i in range(k):
        if i != best:
            p[i] = 1.0 / (k + gamma * (estimates[best] - estimates[i]))
            total += p[i]
    p[best] = 1.0 - total
    return p


def apply_floor(p: list[float], floor_mass: float = FLOOR_MASS) -> list[float]:
    """Mix in `floor_mass` worth of the uniform distribution, guaranteeing
    every candidate probability >= floor_mass / len(p)."""
    if not (0.0 <= floor_mass <= 1.0):
        raise ValueError("floor_mass must be in [0, 1]")
    k = len(p)
    if k < 1:
        raise ValueError("need at least one candidate")
    uniform = floor_mass / k
    return [uniform + (1.0 - floor_mass) * p[i] for i in range(k)]


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


def choose(estimates: list[float], gamma: float, key: int, counter: int) -> tuple[int, float]:
    """The full exploration rule: (chosen index, its propensity).

    The propensity is the floored probability the index was chosen with —
    exactly what the decision record must log (D5).
    """
    p = apply_floor(inverse_gap_probabilities(estimates, gamma))
    i = sample_index(p, key, counter)
    return i, p[i]
