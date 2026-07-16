"""The built-in reward model: per-coordinate ridge regression with
per-update exponential forgetting (design doc D3, revised).

The model's entire memory is two blocks of recency-weighted sums over the
observed (features, reward) pairs:

    xx  —  dim vector:  sum of  x_j^2        (per-feature evidence)
    xy  —  dim vector:  sum of  reward * x_j

Updating on one observation first multiplies every entry by the forgetting
factor, then adds the observation's contribution:

    xx_j  <-  forgetting * xx_j  +  x_j^2
    xy_j  <-  forgetting * xy_j  +  reward * x_j

so each observation's influence fades geometrically with the number of
*updates* since it arrived — constant time, no refit, no clock. The system
is diagonal, so prediction needs no matrix solve anywhere:

    theta_j     = xy_j / (xx_j + ridge)
    estimate    = theta . x
    uncertainty = sqrt( sum_j  x_j^2 / (xx_j + ridge) )

Each weight is that feature's own recency-weighted, shrunk running
average, fully independent of every other feature. Because the sums
forget, the effective sample size is bounded at ~1/(1 - forgetting), so
uncertainty has a floor and the model can never become absolutely certain
(R2); evidence on features that stop appearing fades out of the sums with
every subsequent update, recycling their dimensions under feature churn
(R1). This is the classic tracking estimator (recursive least squares with
a forgetting factor), which is what makes the model non-stationary-capable:
after the world changes, old evidence is outweighed within ~one effective
window regardless of how much history preceded it.

Forgetting is clocked by updates, not by wall-clock time — a deliberate
simplification. The model has no notion of timestamps: training applies
observations in arrival order with arrival weight, so a late reward counts
slightly differently than an on-time one, and replaying a model requires
the ordered update sequence (which the decision log provides). The former
design (timestamp-weighted decay) bought order-independence and exact
cross-agent state merging at the price of wall-clock machinery everywhere;
this design trades those properties away for a smaller engine, and fleet
aggregation is done offline from concatenated decision logs instead.

The state is deliberately the outer-product matrix's *diagonal*, not the
full matrix: memory is O(dim), scoring k candidates is O(k * dim)
arithmetic with no factorization step, and the same code is practical from
a two-arm toy to hashed spaces of millions of dimensions and thousands of
candidates (R5, R6) — the operating point proven at scale by hashed linear
learners like Vowpal Wabbit. The price is credit assignment: features that
always fire together each take full credit (co-occurring evidence is
double-counted, never split), so redundant feature descriptions inflate
estimates. Disentangling combinations of features is the encoder's job —
interaction dimensions, design doc D2 — the same division of labor that
lets a linear model personalize at all.

Every operation is IEEE-754 add/multiply/divide/sqrt in a fixed order
(sqrt is correctly rounded and therefore bit-identical across platforms),
so given the same update sequence the model is bit-identical everywhere.
Every variance term is nonnegative (x_j^2 >= 0, denominators >= ridge > 0),
so no rounding clamp is needed anywhere.

The two fixed constants, neither an exposed tuning knob in the default
API: `ridge` (1.0) is the prior precision — the fresh model's belief that
weights are zero; features are expected to be roughly unit-scale.
`forgetting` (0.999, an effective window of ~1000 observations) is the
tracking rate; `forgetting = 1.0` means never forget. An expert override
exists at construction for both, and the right forgetting rate for a
deployment is selected offline from the decision log, not tuned online.
"""

import math
from dataclasses import dataclass, replace

DEFAULT_FORGETTING = 0.999


@dataclass(frozen=True)
class LinearModel:
    dim: int
    ridge: float
    forgetting: float  # per-update geometric discount on both sums
    xx: tuple[float, ...]  # dim entries: sums of x_j^2 (the outer product's diagonal)
    xy: tuple[float, ...]  # dim entries: sums of reward * x_j


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
        xx=(0.0,) * dim,
        xy=(0.0,) * dim,
    )


def update(model: LinearModel, x: list[float], reward: float) -> LinearModel:
    """Absorb one observation: discount everything remembered, add this."""
    f = model.forgetting
    return replace(
        model,
        xx=tuple([f * v + xi * xi for v, xi in zip(model.xx, x, strict=True)]),
        xy=tuple([f * v + reward * xi for v, xi in zip(model.xy, x, strict=True)]),
    )


@dataclass(frozen=True)
class Factorization:
    """The candidate-independent part of prediction: the solved weights and
    per-coordinate precisions, functions of the model only. Scoring one
    decision's k candidates computes this once and reuses it (decide.py).
    A diagonal system is already solved — "factorizing" is dim reciprocals —
    the name is kept for the once-per-decision shape it gives the layer
    above. Valid until the next update."""

    inv_a: list[float]  # 1 / (xx_j + ridge)
    theta: list[float]


def factorize(model: LinearModel) -> Factorization:
    """Solve for the weights."""
    inv_a = [1.0 / (v + model.ridge) for v in model.xx]
    theta = [model.xy[j] * inv_a[j] for j in range(model.dim)]
    return Factorization(inv_a=inv_a, theta=theta)


def predict_factored(f: Factorization, x: list[float]) -> tuple[float, float]:
    """(estimated reward, uncertainty) for features x, given a factorization
    built from the same model state.

    Zero entries are skipped: hash-encoded candidates are nearly all zeros,
    and the ±0.0 terms they contribute never change a finite sum, so the
    result is bit-identical to the dense loops."""
    theta = f.theta
    inv_a = f.inv_a
    estimate = 0.0
    variance = 0.0
    for j in range(len(x)):
        v = x[j]
        if v == 0.0:
            continue
        estimate += v * theta[j]
        variance += v * v * inv_a[j]
    return estimate, math.sqrt(variance)


def predict(model: LinearModel, x: list[float]) -> tuple[float, float]:
    """(estimated reward, uncertainty) for features x."""
    return predict_factored(factorize(model), x)


def dot(a: "list[float] | tuple[float, ...]", b: list[float]) -> float:
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total
