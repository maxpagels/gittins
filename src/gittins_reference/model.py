"""The built-in reward model: ridge regression on decaying sums (design doc D3).

The model's entire memory is two blocks of decaying sums over the observed
(features, reward) pairs, where each observation's contribution fades with age:

    xx  —  dim x dim matrix (stored row-major):  sum of  x xT
    xy  —  dim vector:                           sum of  reward * x

Updating on one observation adds its outer product and reward-weighted
features to the sums: constant time, no refit. Predicting solves the ridge
system (xx + ridge * I) theta = xy for the weights, then returns

    estimate    = theta . x
    uncertainty = sqrt( x . (xx + ridge * I)^-1 x )

— the standard Bayesian linear-regression posterior shape: uncertainty is
large in feature directions with little surviving (undecayed) evidence.
Because the sums decay, the effective sample size is bounded, so uncertainty
has a floor and grows back during data gaps: the model can never become
absolutely certain (R2).

The solve uses a Cholesky factorization written as plain loops in a fixed
order. Every operation is IEEE-754 add/multiply/divide/sqrt (sqrt is
correctly rounded and therefore bit-identical across platforms), so
prediction is as deterministic as the sums themselves.

`ridge` is the prior precision: the strength of the fresh model's belief that
weights are zero. It is a fixed default (1.0) in the engine, not an exposed
tuning knob; features are expected to be roughly unit-scale.
"""

import math
from dataclasses import dataclass

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
    xx: DecayedVector  # dim*dim entries, row-major
    xy: DecayedVector  # dim entries


def new_model(dim: int, half_life: float, t: float, ridge: float = 1.0) -> LinearModel:
    if dim < 1:
        raise ValueError("dim must be at least 1")
    if not (ridge > 0.0):
        raise ValueError("ridge must be positive")
    return LinearModel(
        dim=dim,
        ridge=ridge,
        xx=new_vector(half_life, t, dim * dim),
        xy=new_vector(half_life, t, dim),
    )


def update(model: LinearModel, x: list[float], reward: float, t: float) -> LinearModel:
    """Absorb one observation. `t` is the decision's timestamp, so a late
    reward counts exactly as if it had arrived on time (see decay.py)."""
    outer = [xi * xj for xi in x for xj in x]
    return LinearModel(
        dim=model.dim,
        ridge=model.ridge,
        xx=vector_add(model.xx, outer, t),
        xy=vector_add(model.xy, [reward * xi for xi in x], t),
    )


def predict(model: LinearModel, x: list[float], t: float) -> tuple[float, float]:
    """(estimated reward, uncertainty) for features x as of time t."""
    d = model.dim
    xx = vector_read(model.xx, t)
    xy = vector_read(model.xy, t)
    a = [
        [xx[i * d + j] + (model.ridge if i == j else 0.0) for j in range(d)]
        for i in range(d)
    ]
    lower = cholesky(a)
    theta = solve_cholesky(lower, xy)
    z = solve_cholesky(lower, list(x))
    estimate = dot(x, theta)
    variance = dot(x, z)
    if variance < 0.0:  # only reachable through rounding; never meaningfully
        variance = 0.0
    return estimate, math.sqrt(variance)


def merge_models(a: LinearModel, b: LinearModel) -> LinearModel:
    """Combine two models as if one had seen both observation streams."""
    if a.dim != b.dim:
        raise ValueError("cannot merge models with different dimensions")
    if a.ridge != b.ridge:
        raise ValueError("cannot merge models with different ridge priors")
    return LinearModel(
        dim=a.dim,
        ridge=a.ridge,
        xx=vector_merge(a.xx, b.xx),
        xy=vector_merge(a.xy, b.xy),
    )


def dot(a: "list[float] | tuple[float, ...]", b: list[float]) -> float:
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


def cholesky(a: list[list[float]]) -> list[list[float]]:
    """Lower-triangular L with L LT = a, for symmetric positive-definite a
    (guaranteed here by ridge > 0). Plain textbook loops, fixed order."""
    d = len(a)
    lower = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i + 1):
            s = a[i][j]
            for k in range(j):
                s -= lower[i][k] * lower[j][k]
            if i == j:
                lower[i][j] = math.sqrt(s)
            else:
                lower[i][j] = s / lower[j][j]
    return lower


def solve_cholesky(lower: list[list[float]], b: list[float]) -> list[float]:
    """Solve (L LT) x = b by forward then back substitution."""
    d = len(lower)
    y = [0.0] * d
    for i in range(d):
        s = b[i]
        for k in range(i):
            s -= lower[i][k] * y[k]
        y[i] = s / lower[i][i]
    x = [0.0] * d
    for i in range(d - 1, -1, -1):
        s = y[i]
        for k in range(i + 1, d):
            s -= lower[k][i] * x[k]
        x[i] = s / lower[i][i]
    return x
