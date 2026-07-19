"""The built-in reward model: per-coordinate ridge regression with
per-update exponential forgetting (design doc D3, revised), stored as
pre-scaled sums so one update is O(nonzeros), not O(dim).

The model's entire memory is two blocks of recency-weighted sums over the
observed (features, reward) pairs, plus one scalar:

    scale —  the movable origin: the true sums are scale * xx_j, scale * xy_j
    xx    —  dim entries: pre-scaled sums of  x_j^2      (per-feature evidence)
    xy    —  dim entries: pre-scaled sums of  reward * x_j

Conceptually, updating on one observation multiplies every true sum by the
forgetting factor and adds the observation's contribution:

    true_xx_j  <-  forgetting * true_xx_j  +  x_j^2

Multiplying every entry per update would make training O(dim). Instead the
forgetting multiply is folded into the scalar and the contribution is
pre-divided out of it (the decay-on-read formulation the design already
chose for the earlier wall-clock decay):

    scale  <-  forgetting * scale
    xx_j   <-  xx_j  +  x_j^2      * (1 / scale)     for the nonzero x_j only
    xy_j   <-  xy_j  +  reward*x_j * (1 / scale)

so an update is one multiply, one divide, and O(nonzeros) multiply-adds;
untouched coordinates are untouched. `1 / scale` grows as evidence
accumulates, so when `scale` falls to RENORM_THRESHOLD (2^-512, chosen so
pre-scaled entries stay far from the 2^1024 overflow ceiling) every entry
is multiplied by `scale` once and the scalar resets to 1.0 — O(dim), but
only once every ~355k updates at the default forgetting; amortized O(1).
Features are sparse (index, value) pairs everywhere in this module — the
encoder's output format (encoding.py).

The system is diagonal, so prediction needs no matrix solve anywhere:

    a_j         = scale * xx_j + ridge
    theta_j     = scale * xy_j / a_j
    estimate    = theta . x
    uncertainty = sqrt( sum_j  x_j^2 / a_j )

Each weight is that feature's own recency-weighted, shrunk running
average, fully independent of every other feature — so `factorize` solves
lazily: a coordinate's (1/a_j, theta_j) is computed the first time a
candidate touches it and memoized for the rest of the decision, making a
decision's solve cost O(coordinates touched), never O(dim). Because the
sums forget, the effective sample size is bounded at ~1/(1 - forgetting),
so uncertainty has a floor and the model can never become absolutely
certain; evidence on features that stop appearing fades out of the
true sums with every subsequent update, recycling their dimensions under
feature churn. This is the classic tracking estimator (recursive
least squares with a forgetting factor): after the world changes, old
evidence is outweighed within ~one effective window regardless of how much
history preceded it.

Forgetting is clocked by updates, not by wall-clock time — a deliberate
simplification. The model has no notion of timestamps: training applies
observations in arrival order with arrival weight, so a late reward counts
slightly differently than an on-time one, and replaying a model requires
the ordered update sequence (which the decision log provides).

The state is deliberately the outer-product matrix's *diagonal*, not the
full matrix: memory is O(dim), scoring k candidates is O(k * nonzeros)
arithmetic with no factorization step, and the same code is practical from
a two-arm toy to hashed spaces of millions of dimensions and thousands of
candidates — the operating point proven at scale by hashed linear
learners like Vowpal Wabbit. The price is credit assignment: features that
always fire together each take full credit (co-occurring evidence is
double-counted, never split), so redundant feature descriptions inflate
estimates. Disentangling combinations of features is the encoder's job —
interaction dimensions, design doc D2 — the same division of labor that
lets a linear model personalize at all.

Every operation is IEEE-754 add/multiply/divide/sqrt in a fixed order
(sqrt is correctly rounded and therefore bit-identical across platforms),
so given the same update sequence the model is bit-identical everywhere.
The pre-scaled bookkeeping rounds differently than the former
multiply-every-entry bookkeeping — the same weights up to rounding, not
the same bits — a deliberate semantics change, pinned by the golden
corpus. Every variance term is nonnegative (x_j^2 >= 0, scale > 0,
denominators >= ridge > 0), so no rounding clamp is needed anywhere.

The two fixed constants, neither an exposed tuning knob in the default
API: `ridge` (1.0) is the prior precision — the fresh model's belief that
weights are zero; features are expected to be roughly unit-scale.
`forgetting` (0.999, an effective window of ~1000 observations) is the
tracking rate; `forgetting = 1.0` means never forget (the scale stays at
exactly 1.0 and the sums are plain sums). An expert override exists at
construction for both, and the right forgetting rate for a deployment is
selected offline from the decision log, not tuned online.
"""

import math
from dataclasses import dataclass, field, replace

DEFAULT_FORGETTING = 0.999

# Renormalize the pre-scaled sums when the scale falls this low: 1/scale
# stays <= 2^512, keeping every pre-scaled entry far from double overflow
# (2^1024) for roughly-unit-scale features. A power of two, but the renorm
# multiply still rounds — determinism comes from the fixed schedule, not
# from exactness.
RENORM_THRESHOLD = 2.0**-512

# Features everywhere in this module are sparse (index, value) pairs in
# strictly increasing index order with nonzero values — encoding.py's
# output format.
Features = tuple[tuple[int, float], ...] | list[tuple[int, float]]


@dataclass(frozen=True)
class LinearModel:
    dim: int
    ridge: float
    forgetting: float  # per-update geometric discount on the true sums
    scale: float  # movable origin: true sums are scale * xx, scale * xy
    xx: tuple[float, ...]  # dim pre-scaled sums of x_j^2 (the diagonal)
    xy: tuple[float, ...]  # dim pre-scaled sums of reward * x_j


def new_model(
    dim: int, forgetting: float = DEFAULT_FORGETTING, ridge: float = 1.0
) -> LinearModel:
    if dim < 1:
        raise ValueError("dim must be at least 1")
    if not (0.0 < forgetting <= 1.0):
        raise ValueError("forgetting must be in (0, 1] (1.0 = never forget)")
    if not (ridge > 0.0):
        raise ValueError("ridge must be positive")
    return LinearModel(
        dim=dim,
        ridge=ridge,
        forgetting=forgetting,
        scale=1.0,
        xx=(0.0,) * dim,
        xy=(0.0,) * dim,
    )


def update(model: LinearModel, x: Features, reward: float) -> LinearModel:
    """Absorb one observation: fold the forgetting discount into the scale,
    add the observation's pre-scaled contribution to its touched
    coordinates only. O(nonzeros) arithmetic. (The reference's immutable
    tuples still copy O(dim) pointers per update — memcpy-speed, no
    arithmetic; a compiled core mutates in place.)"""
    scale = model.forgetting * model.scale
    inv = 1.0 / scale
    xx = list(model.xx)
    xy = list(model.xy)
    for j, v in x:
        xx[j] += (v * v) * inv
        xy[j] += (reward * v) * inv
    if scale <= RENORM_THRESHOLD:
        xx = [scale * u for u in xx]
        xy = [scale * u for u in xy]
        scale = 1.0
    return replace(model, scale=scale, xx=tuple(xx), xy=tuple(xy))


@dataclass
class Factorization:
    """The candidate-independent part of prediction, solved lazily: a
    coordinate's precision and weight are computed the first time any
    candidate touches it and memoized for the rest of the decision
    (decide.py scores every candidate against one factorization). A
    diagonal system needs one reciprocal per *touched* coordinate — never
    O(dim) — the name is kept for the once-per-decision shape it gives the
    layer above. Valid until the next update."""

    scale: float
    ridge: float
    xx: tuple[float, ...]
    xy: tuple[float, ...]
    cache: "dict[int, tuple[float, float]]" = field(default_factory=dict)

    def coordinate(self, j: int) -> "tuple[float, float]":
        """(1 / a_j, theta_j) for one coordinate, memoized."""
        got = self.cache.get(j)
        if got is None:
            inv_a = 1.0 / (self.scale * self.xx[j] + self.ridge)
            got = (inv_a, (self.scale * self.xy[j]) * inv_a)
            self.cache[j] = got
        return got


def factorize(model: LinearModel) -> Factorization:
    """The lazy solve: O(1) now, one reciprocal per touched coordinate later."""
    return Factorization(scale=model.scale, ridge=model.ridge, xx=model.xx, xy=model.xy)


def estimate_factored(f: Factorization, x: Features) -> float:
    """Estimated reward only, for callers that need no uncertainty (the
    decide layer): one multiply-add per nonzero and no sqrt."""
    estimate = 0.0
    for j, v in x:
        estimate += v * f.coordinate(j)[1]
    return estimate


def predict_factored(f: Factorization, x: Features) -> tuple[float, float]:
    """(estimated reward, uncertainty) for features x, given a factorization
    built from the same model state. O(nonzeros)."""
    estimate = 0.0
    variance = 0.0
    for j, v in x:
        inv_a, theta = f.coordinate(j)
        estimate += v * theta
        variance += (v * v) * inv_a
    return estimate, math.sqrt(variance)


def predict(model: LinearModel, x: Features) -> tuple[float, float]:
    """(estimated reward, uncertainty) for features x."""
    return predict_factored(factorize(model), x)
