# Spec: the built-in reward model

Design doc reference: D3 (model state is decaying running sums), section 5
(the reward prediction layer), R7 (the plug-in interface it implements).
Implemented in `src/gittins_reference/model.py` (PR 4), on top of the
decaying sums of `spec/decay.md`.

## State

A `LinearModel` is `(dim, ridge, xx, xy)`:

- `dim` — the feature-vector length, fixed at creation.
- `ridge` — the prior precision (the fresh model's belief that all weights
  are zero). Fixed default 1.0; not a tuning knob in the default API.
  Features are expected to be roughly unit-scale.
- `xx` — a DecayedVector of `dim * dim` entries, row-major: the decaying sum
  of outer products `x xT` of every observed feature vector.
- `xy` — a DecayedVector of `dim` entries: the decaying sum of
  `reward * x`.

Both vectors share the model's half-life; their origins advance in lockstep
because they receive the same timestamps.

## Operations

**update(model, x, reward, t)** — adds `x xT` into `xx` and `reward * x`
into `xy`, both timestamped `t` (the *decision's* time, so late rewards
count exactly as on-time ones — inherited from the decay layer). Constant
time, no refit. This is the R7 plug-in `update` shape.

**predict(model, x, t)** — with `XX = read(xx, t)` and `Xy = read(xy, t)`:

    A = XX + ridge * I
    theta = A^-1 Xy              (the ridge-regression weights)
    estimate    = x . theta
    uncertainty = sqrt( x . A^-1 x )

The solve is a Cholesky factorization of A written as plain nested loops in
a fixed order (see `cholesky` / `solve_cholesky`), followed by fixed-order
dot products. A is symmetric positive definite because `ridge > 0`. All
operations are IEEE-754 add/multiply/divide/sqrt; sqrt is correctly rounded
by the standard, so prediction is bit-identical across platforms — no
vendored polynomial needed beyond the decay layer's exp2. A negative
variance can arise only from rounding at magnitudes ~1e-16 and is clamped
to zero before the sqrt.

**merge_models(a, b)** — entry-by-entry DecayedVector merge of `xx` and
`xy`; requires equal `dim` and `ridge`. Inherits the decay layer's
exactness: bit-exact commutative, equal to a central model up to ~1e-12.

## Why uncertainty behaves correctly with no extra machinery

`x . A^-1 x` is large in feature directions with little surviving evidence.
Because the sums decay, "surviving" means *recent*: the effective sample
size is bounded at roughly one half-life of data, so uncertainty has a
floor, and during a data gap it climbs back toward the prior
(`sqrt(x . x / ridge)`). This is R2's "never fully converge" property
falling out of D3's state shape — there is no separate exploration-bonus
bookkeeping to maintain. Verified in `TestUncertainty`.

## Golden vectors

Model: dim 2, half-life 3600 s, ridge 1.0, created at T0 = 1752000000.0.
Updates: `([1,0], 1.0, T0)`, `([0,1], -0.5, T0+1800)`,
`([1,1], 0.25, T0+3600)`. Prediction for `x = [1,-1]` at T0+5400:

    estimate    = 0.4318473976784247
    uncertainty = 1.1847437129969538
