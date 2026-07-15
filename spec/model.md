# Spec: the built-in reward model

Design doc reference: D3 (model state is decaying running sums), section 5
(the reward prediction layer), R7 (the plug-in interface it implements).
Implemented in `src/gittins_reference/model.py` (PR 4; state reduced to the
diagonal in the pre-battery revision), on top of the decaying sums of
`spec/decay.md`.

## State

A `LinearModel` is `(dim, ridge, xx, xy)`:

- `dim` — the feature-vector length, fixed at creation.
- `ridge` — the prior precision (the fresh model's belief that all weights
  are zero). Fixed default 1.0; not a tuning knob in the default API.
  Features are expected to be roughly unit-scale.
- `xx` — a DecayedVector of `dim` entries: the decaying sum of `x_j^2` per
  feature (the outer-product matrix's diagonal; the full matrix is
  deliberately not kept — see "Why the diagonal" below).
- `xy` — a DecayedVector of `dim` entries: the decaying sum of
  `reward * x`.

Both vectors share the model's half-life; their origins advance in lockstep
because they receive the same timestamps.

## Operations

**update(model, x, reward, t)** — adds `x_j^2` into `xx` and `reward * x_j`
into `xy`, both timestamped `t` (the *decision's* time, so late rewards
count exactly as on-time ones — inherited from the decay layer). Constant
time, no refit. This is the R7 plug-in `update` shape.

**predict(model, x, t)** — with `XX = read(xx, t)` and `Xy = read(xy, t)`,
the ridge system is diagonal, so there is no matrix solve:

    theta_j     = Xy_j / (XX_j + ridge)
    estimate    = x . theta
    uncertainty = sqrt( sum_j  x_j^2 / (XX_j + ridge) )

Each weight is that feature's own decayed, shrunk running average, fully
independent of every other feature. All operations are IEEE-754
add/multiply/divide/sqrt in a fixed order; sqrt is correctly rounded by the
standard, so prediction is bit-identical across platforms — no vendored
polynomial needed beyond the decay layer's exp2. Every variance term is
nonnegative (`x_j^2 >= 0`, denominators `>= ridge > 0`), so no rounding
clamp is needed.

**factorize(model, t) / predict_factored(f, x)** — the candidate-independent
part of predict (`theta` and the reciprocals `1/(XX_j + ridge)`) is computed
once per decision and shared by every candidate (used by `decide`).
`predict` is exactly `predict_factored(factorize(model, t), x)`.

**merge_models(a, b)** — entry-by-entry DecayedVector merge of `xx` and
`xy`; requires equal `dim` and `ridge`. Inherits the decay layer's
exactness: bit-exact commutative, equal to a central model up to ~1e-12.

## Why the diagonal

Keeping only the diagonal makes memory O(dim) and scoring k candidates
O(k * dim) arithmetic with no factorization step, so the same code is
practical from a two-arm toy to hashed spaces of millions of dimensions and
thousands of candidates (R5, R6) — the operating point proven at scale by
hashed linear learners like Vowpal Wabbit. It is also what makes the whole
model readable by any competent programmer (R7): there is no linear-algebra
machinery anywhere.

The price is credit assignment: features that always fire together each
take full credit — co-occurring evidence is double-counted, never split.
Pinned in `TestPredict::test_cofiring_features_double_count`: two features
always seen together with reward 1.0 jointly predict ~2x the truth.
Disentangling *combinations* of features is the encoder's job (interaction
dimensions, `spec/encoding.md`) — the same division of labor that lets a
linear model personalize at all. Redundant feature descriptions inflate
estimates; measuring how much this costs against a credit-splitting model
is a Phase 0.5 battery question.

## Why uncertainty behaves correctly with no extra machinery

`sum x_j^2 / (XX_j + ridge)` is large in feature directions with little
surviving evidence. Because the sums decay, "surviving" means *recent*: the
effective sample size is bounded at roughly one half-life of data, so
uncertainty has a floor, and during a data gap it climbs back toward the
prior (`sqrt(x . x / ridge)`). This is R2's "never fully converge" property
falling out of D3's state shape — there is no separate exploration-bonus
bookkeeping to maintain. Verified in `TestUncertainty`.

## Golden vectors

Model: dim 2, half-life 3600 s, ridge 1.0, created at T0 = 1752000000.0.
Updates: `([1,0], 1.0, T0)`, `([0,1], -0.5, T0+1800)`,
`([1,1], 0.25, T0+3600)`. Prediction for `x = [1,-1]` at T0+5400:

    estimate    = 0.2905354624569479
    uncertainty = 0.9686914955549797

The full corpus lives in `spec/golden.json` (model section plus the
end-to-end episode).
