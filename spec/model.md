# Spec: the built-in reward model

Design doc reference: D3 (model state is recency-weighted running sums),
section 5 (the reward prediction layer), R7 (the plug-in interface it
implements). Implemented in `src/gittins_reference/model.py` (state reduced
to the diagonal in the pre-battery revision; wall-clock decay replaced by
per-update forgetting in the post-battery simplification).

## State

A `LinearModel` is `(dim, ridge, forgetting, scale, xx, xy)`:

- `dim` — the feature-space size, fixed at creation.
- `ridge` — the prior precision (the fresh model's belief that all weights
  are zero). Fixed default 1.0; not a tuning knob in the default API.
  Features are expected to be roughly unit-scale.
- `forgetting` — the per-update geometric discount, in (0, 1]; 1.0 means
  never forget. Fixed default 0.999 (an effective window of ~1000
  observations). An expert override at construction, not a default-API
  knob: the right rate for a deployment is selected offline from the
  decision log.
- `scale` — the movable origin, in (0, 1]; starts at 1.0. The *true*
  recency-weighted sums are `scale * xx_j` and `scale * xy_j`.
- `xx` — `dim` floats: the pre-scaled sum of `x_j^2` per feature (the
  outer-product matrix's diagonal; the full matrix is deliberately not
  kept — see "Why the diagonal" below).
- `xy` — `dim` floats: the pre-scaled sum of `reward * x_j`.

Features are sparse everywhere in this layer: (index, value) pairs in
strictly increasing index order, values nonzero — the encoder's output
format (`spec/encoding.md`).

## Operations

**update(model, x, reward)** — conceptually discounts every true sum by
`forgetting` and adds this observation:

    true_xx_j  <-  forgetting * true_xx_j  +  x_j^2
    true_xy_j  <-  forgetting * true_xy_j  +  reward * x_j

Executed without touching every coordinate by folding the discount into
the scale and pre-dividing the contribution out of it, exactly:

    scale  <-  forgetting * scale
    inv    =   1 / scale
    xx_j   <-  xx_j + (x_j * x_j) * inv        for the given pairs only
    xy_j   <-  xy_j + (reward * x_j) * inv

then, if `scale <= RENORM_THRESHOLD` (2^-512), every entry is multiplied
by `scale` once and `scale` resets to 1.0. The threshold keeps
`1/scale <= 2^512`, so pre-scaled entries stay far from double overflow
(2^1024) for roughly-unit-scale features; the sweep costs O(dim) but fires
only every ~355k updates at the default forgetting — amortized O(1), and
arithmetic per update is O(nonzeros). With `forgetting = 1.0` the scale
stays at exactly 1.0 and the sums are plain sums.

Observation i of n carries weight `forgetting^(n-1-i)`: influence fades
geometrically with the number of *updates* since it arrived. This is the
classic tracking estimator (recursive least squares with a forgetting
factor), and it is what makes the model non-stationary-capable: after the
world changes, old evidence is outweighed within ~one effective window
regardless of how much history preceded it. Training is order-dependent —
a late reward counts slightly less than an on-time one would have — so
replaying a model requires the ordered update sequence, which the decision
log provides. The pre-scaled bookkeeping rounds differently than a
multiply-every-entry formulation would (the same weights up to rounding,
not the same bits); the executed operation sequence above *is* the
contract, pinned by the golden corpus and
`TestPredict::test_prescaled_sums_match_explicit_bookkeeping_bit_for_bit`.

**predict(model, x)** — the ridge system is diagonal, so there is no
matrix solve:

    a_j         = scale * xx_j + ridge
    theta_j     = (scale * xy_j) / a_j
    estimate    = sum_j  x_j * theta_j          (index order)
    uncertainty = sqrt( sum_j  (x_j * x_j) / a_j )

Each weight is that feature's own recency-weighted, shrunk running
average, fully independent of every other feature. All operations are
IEEE-754 add/multiply/divide/sqrt in a fixed order; sqrt is correctly
rounded by the standard, so given the same update sequence the model is
bit-identical across platforms. Every variance term is nonnegative
(`x_j^2 >= 0`, `scale > 0`, denominators `>= ridge > 0`), so no rounding
clamp is needed. O(nonzeros): only the supplied pairs are visited.

**factorize(model) / predict_factored(f, x) / estimate_factored(f, x)** —
the candidate-independent part of predict is solved *lazily*: a
coordinate's `(1/a_j, theta_j)` is computed the first time any candidate
touches it and memoized for the rest of the decision, so one decision's
solve cost is O(coordinates touched), never O(dim). `predict` is exactly
`predict_factored(factorize(model), x)`; `estimate_factored` is the
estimate alone — one multiply-add per nonzero, no sqrt — and is what
`decide` scores candidates with (the epsilon-greedy rule consumes
estimates only). A factorization is valid until the next update.

## Why the diagonal

Keeping only the diagonal makes memory O(dim) and scoring k candidates
O(k * nonzeros) arithmetic with no factorization step, so the same code is
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
linear model personalize at all.

## Why uncertainty behaves correctly with no extra machinery

`sum x_j^2 / (xx_j + ridge)` is large in feature directions with little
surviving evidence. Because the sums forget, "surviving" means *recent*:
the effective sample size is bounded at ~1/(1 - forgetting), so `xx_j`
saturates and uncertainty has a floor — the model can never become
absolutely certain (R2). Evidence on features that stop appearing fades
with every subsequent update, so their uncertainty climbs back toward the
prior and their hashed dimensions are recycled under churn (R1). Pinned in
`TestUncertainty` and `TestForgetting`.

## Golden vectors

Model: dim 2, forgetting 0.9, ridge 1.0. Updates: `(((0,1.0),), 1.0)`,
`(((1,1.0),), -0.5)`, `(((0,1.0),(1,1.0)), 0.25)`, in that order.
Prediction for `x = ((0,1.0),(1,-1.0))`:

    estimate    = 0.4461897165296356
    uncertainty = 0.8370779368301933

The full corpus lives in `spec/golden.json` (model section — including the
state after every update — plus the end-to-end episode).
