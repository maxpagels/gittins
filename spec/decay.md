# Spec: exponentially decaying sums

Design doc reference: D3 (decaying running sums), §13 risk 3 (decay + late
rewards + merge). Implemented in `src/gittins_reference/decay.py` and
`src/gittins_reference/detmath.py` (PR 3).

## Why decay exists

Decay is one mechanism doing four jobs; every other part of the engine leans
on it.

1. **Adaptation (R2).** Undecayed sums only grow, so a model's effective
   sample size grows without bound and each new observation moves it less —
   it freezes. Decay caps the effective sample size at roughly one half-life
   of data, so estimates always track the present.
2. **Perpetual exploration (R2).** Exploration is driven by uncertainty.
   Undecayed evidence drives uncertainty monotonically to zero and
   exploration dies. Decay floors uncertainty: confidence is always
   confidence in *recent* evidence, so changed arms get rediscovered.
3. **Arm cleanup (R1).** Arm identity lives in hashed feature dimensions
   (D2). A vanished arm's dimensions stop receiving evidence and their sums
   fade to the prior — decay *is* the cleanup mechanism; no eviction code
   exists.
4. **Time-coherent merging (R8).** Fleet merges add sums together. Decayed
   sums make that a pool of recent knowledge; undecayed, the oldest agent
   dominates by sheer volume.

Exponential decay specifically — rather than windows, restarts, or change
detection — because it is the unique weighting where aging by Δ₁ then Δ₂
equals one multiplication by the weight for Δ₁+Δ₂. That property is what
makes constant-time updates, decay-on-read, and merge-by-addition possible,
and the alternatives all introduce knobs (window length, threshold), which
R2 forbids.

**Why not just a constant learning rate?** Online SGD with a fixed step
size already forgets exponentially — `w ← w + η(r − w)` is an exponential
moving average, so implicit forgetting is real. It is rejected as the
mechanism because: (a) η *is* the half-life knob in disguise, so it solves
nothing for R2; (b) SGD forgets per *update*, not per second — its memory
depends on traffic volume, and it forgets nothing during a traffic gap;
(c) SGD state is the endpoint of an ordered gradient path, so late rewards
can't be applied at decision-time weight (R4) and two agents' weights can't
be merged (R8) — sums are order-free, gradient paths are not; (d) SGD gives
a point estimate, while exploration needs the uncertainty that falls out of
decayed sufficient statistics for free. Decaying sums are the same
forgetting promoted from a side effect of step size to an explicit,
timestamped property of the data.

## Semantics

A `DecayedAccumulator` is `(half_life, origin, total)`, all f64. Timestamps
and half-lives are in seconds. A contribution of value `v` at time `t`,
read at time `T`, is worth

    v * 2^(-(T - t) / half_life)

`half_life = +inf` means "never forget" and requires no special casing
(all decay exponents become ±0, and 2^±0 = 1).

**Decay-on-read.** Contributions are stored pre-scaled to the origin:
`total = Σ v_i * 2^((t_i - origin) / half_life)`. So:

- `add(acc, v, t)`: `total += v * 2^((t - origin)/half_life)` — pure
  addition, no sweep over old data. For a reward, `t` is the *decision's*
  timestamp, which makes a late arrival worth exactly what an on-time
  arrival would have been.
- `read(acc, T)`: `total * 2^(-(T - origin)/half_life)`.
- `merge(a, b)`: rescale the older-origin total onto the newer origin, add
  the totals. Requires equal half-lives.

**Origin and renormalization.** The origin exists only to keep the stored
float in range; it starts as the accumulator's creation time. When an add
lands more than `RENORM_LIMIT = 128` half-lives past the origin, the total
is rescaled to a new origin at that add's timestamp:
`total *= 2^(-(t - origin)/half_life); origin = t`. The threshold keeps
stored weights at or below 2^128 (far from f64 overflow). Contributions
older than ~53 half-lives are below f64's representable ratio relative to
fresh ones (2^-53) — semantically negligible before they are numerically
lost, so renormalization discards nothing meaningful. Very old totals
rescale to exactly 0.0 (underflow), which is the correct "fully forgotten".

**Deterministic 2^x.** All decay weights come from the vendored
`detmath.exp2`, never the platform libm (§8): split `x` into integer `n`
and fraction `f ∈ [0,1)`; evaluate the pinned 16-coefficient Taylor
polynomial of 2^f (coefficient k is (ln 2)^k / k!) by Horner's rule from
the highest coefficient down; scale exactly by 2^n. Inputs ≥ 1024 give
+inf; ≤ −1075 give 0.0; NaN passes through. The pinned coefficients in
`detmath.py` are the definition — implementations reproduce the
computation, not "a correct exp2" (measured within ~1 ulp of correctly
rounded anyway).

## Exactness contract

Verified by tests; the Rust core must satisfy the same table.

| Property | Guarantee |
|---|---|
| Same inputs, same state | bit-exact (everything is deterministic) |
| `merge(a, b) == merge(b, a)` | bit-exact, including across renorm eras |
| Add order irrelevant (late = on-time) | bit-exact within one renorm era; ~1e-12 relative across eras |
| Renormalization preserves readings | ~1e-12 relative |
| Merge equals one central accumulator | ~1e-12 relative |

The "~1e-12 across eras" cases arise because rescaling rounds once; the
paths remain individually deterministic.

## Golden vectors

`exp2`: `exp2(0.5) = 1.414213562373095`, `exp2(-0.5) = 0.7071067811865475`,
`exp2(1.75) = 3.363585661014858`, `exp2(-100.25) = 6.633503073341491e-31`.

Accumulator (half-life 3600 s, created at T0 = 1752000000.0; add 1.0 at T0,
2.5 at T0+5400, −0.75 at T0+7200):

    origin = 1752000000.0
    total  = 5.0710678118654755
    read(T0 + 10800) = 0.6338834764831844
