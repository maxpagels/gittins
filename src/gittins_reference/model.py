"""The built-in reward model: per-coordinate ridge regression on decaying sums
(design doc D3).

The model's entire memory is two blocks of decaying sums over the observed
(features, reward) pairs, where each observation's contribution fades with age:

    xx  —  dim vector:  sum of  x_j^2        (per-feature evidence)
    xy  —  dim vector:  sum of  reward * x_j

Updating on one observation adds its squared and reward-weighted features to
the sums: constant time, no refit. The system is diagonal, so prediction
needs no matrix solve anywhere:

    theta_j     = xy_j / (xx_j + ridge)
    estimate    = theta . x
    uncertainty = sqrt( sum_j  x_j^2 / (xx_j + ridge) )

Each weight is that feature's own decayed, shrunk running average, fully
independent of every other feature. Uncertainty is large in feature
directions with little surviving (undecayed) evidence. Because the sums
decay, the effective sample size is bounded, so uncertainty has a floor and
grows back during data gaps: the model can never become absolutely certain
(R2).

The state is deliberately the outer-product matrix's *diagonal*, not the
full matrix: memory is O(dim), scoring k candidates is O(k * dim) arithmetic
with no factorization step, and the same code is practical from a two-arm
toy to hashed spaces of millions of dimensions and thousands of candidates
(R5, R6) — the operating point proven at scale by hashed linear learners
like Vowpal Wabbit. The price is credit assignment: features that always
fire together each take full credit (co-occurring evidence is double-counted,
never split), so redundant feature descriptions inflate estimates.
Disentangling combinations of features is the encoder's job — interaction
dimensions, design doc D2 — the same division of labor that lets a linear
model personalize at all.

Every operation is IEEE-754 add/multiply/divide/sqrt in a fixed order (sqrt
is correctly rounded and therefore bit-identical across platforms), so
prediction is as deterministic as the sums themselves. Every variance term
is nonnegative (x_j^2 >= 0, denominators >= ridge > 0), so no rounding
clamp is needed anywhere.

`ridge` is the prior precision: the strength of the fresh model's belief that
weights are zero. It is a fixed default (1.0) in the engine, not an exposed
tuning knob; features are expected to be roughly unit-scale.
"""

import math
from dataclasses import dataclass, replace

from gittins_reference.decay import (
    DecayedVector,
    new_vector,
    vector_add,
    vector_merge,
    vector_read,
)


@dataclass(frozen=True)
class LinearModel:
    dim: int
    ridge: float
    xx: DecayedVector  # dim entries: sums of x_j^2 (the outer product's diagonal)
    xy: DecayedVector  # dim entries: sums of reward * x_j


def new_model(dim: int, half_life: float, t: float, ridge: float = 1.0) -> LinearModel:
    if dim < 1:
        raise ValueError("dim must be at least 1")
    if not (ridge > 0.0):
        raise ValueError("ridge must be positive")
    return LinearModel(
        dim=dim,
        ridge=ridge,
        xx=new_vector(half_life, t, dim),
        xy=new_vector(half_life, t, dim),
    )


def update(model: LinearModel, x: list[float], reward: float, t: float) -> LinearModel:
    """Absorb one observation. `t` is the decision's timestamp, so a late
    reward counts exactly as if it had arrived on time (see decay.py)."""
    return replace(
        model,
        xx=vector_add(model.xx, [xi * xi for xi in x], t),
        xy=vector_add(model.xy, [reward * xi for xi in x], t),
    )


@dataclass(frozen=True)
class Factorization:
    """The candidate-independent part of prediction: the solved weights and
    per-coordinate precisions, functions of (model, t) only. Scoring one
    decision's k candidates computes this once and reuses it (decide.py).
    A diagonal system is already solved — "factorizing" is dim reciprocals —
    the name is kept for the once-per-decision shape it gives the layer
    above. Ephemeral: decay moves the evidence under the fixed ridge between
    timestamps, so a factorization is only valid at the `t` it was built for."""

    inv_a: list[float]  # 1 / (xx_j + ridge)
    theta: list[float]


def factorize(model: LinearModel, t: float) -> Factorization:
    """Read the decayed sums as of time t and solve for the weights."""
    xx = vector_read(model.xx, t)
    xy = vector_read(model.xy, t)
    inv_a = [1.0 / (v + model.ridge) for v in xx]
    theta = [xy[j] * inv_a[j] for j in range(model.dim)]
    return Factorization(inv_a=inv_a, theta=theta)


def predict_factored(f: Factorization, x: list[float]) -> tuple[float, float]:
    """(estimated reward, uncertainty) for features x, given a factorization
    built from the same model at the same t."""
    estimate = dot(x, f.theta)
    variance = 0.0
    for j in range(len(x)):
        variance += x[j] * x[j] * f.inv_a[j]
    return estimate, math.sqrt(variance)


def predict(model: LinearModel, x: list[float], t: float) -> tuple[float, float]:
    """(estimated reward, uncertainty) for features x as of time t."""
    return predict_factored(factorize(model, t), x)


def merge_models(a: LinearModel, b: LinearModel) -> LinearModel:
    """Combine two models as if one had seen both observation streams."""
    if a.dim != b.dim:
        raise ValueError("cannot merge models with different dimensions")
    if a.ridge != b.ridge:
        raise ValueError("cannot merge models with different ridge priors")
    return replace(a, xx=vector_merge(a.xx, b.xx), xy=vector_merge(a.xy, b.xy))


def dot(a: "list[float] | tuple[float, ...]", b: list[float]) -> float:
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total
