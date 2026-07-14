"""Exponentially decaying sums — the engine's only memory (design doc D3).

A DecayedAccumulator holds a running sum in which a contribution of value v
made at time t counts, when read at a later time T, as

    v * 2^(-(T - t) / half_life)

i.e. every contribution loses half its weight per half-life. Timestamps and
half-lives are in seconds. half_life = inf means "never forget" and needs no
special casing anywhere.

Decay-on-read: contributions are stored pre-multiplied by 2^((t - origin) /
half_life), so `add` is a pure addition — no sweep over old data, no decay
step. The decay happens implicitly in `read`, which rescales the total once.
Because stored sums live on a shared absolute timescale, merging two
accumulators is (after aligning origins) plain addition, and a late reward
added with its decision-time timestamp is automatically weighted as if it had
arrived on time.

The origin exists only to keep the stored float in range: 2^(t/half_life) with
t an absolute timestamp would overflow. When an add's exponent would exceed
RENORM_LIMIT half-lives past the origin, the accumulator is renormalized:
the total is rescaled to a new origin at that timestamp. Contributions older
than ~53 half-lives are below double precision's representable ratio anyway
(2^-53 of the newest weight), so renormalization discards nothing meaningful.

Exact-versus-approximate contract (verified in tests):
- merge(a, b) == merge(b, a), bit for bit.
- Two adds commute bit for bit if neither triggers renormalization;
  across a renormalization they agree to ~1e-12 relative.
- All decay weights come from detmath.exp2, never the platform libm.
"""

from dataclasses import dataclass

from gittins_reference.detmath import exp2

# Renormalize when an add lands this many half-lives past the origin.
# Keeps stored weights at or below 2^128 — far from f64 overflow, and far
# beyond the ~53-half-life horizon where old contributions become invisible.
RENORM_LIMIT = 128.0


@dataclass(frozen=True)
class DecayedAccumulator:
    half_life: float  # seconds; math.inf = never forget
    origin: float  # timestamp the stored total is scaled relative to
    total: float  # sum of v_i * 2^((t_i - origin) / half_life)


def new_accumulator(half_life: float, t: float) -> DecayedAccumulator:
    """Create an empty accumulator. `t` (creation time) becomes the origin,
    so accumulators created at the same time stay bit-compatible regardless
    of the order their first contributions arrive in."""
    if not (half_life > 0.0):
        raise ValueError("half_life must be positive (math.inf is allowed)")
    return DecayedAccumulator(half_life=half_life, origin=t, total=0.0)


def add(acc: DecayedAccumulator, value: float, t: float) -> DecayedAccumulator:
    """Add `value` with timestamp `t` — for a reward, the *decision's* time,
    which is what makes late arrivals count neither more nor less than
    on-time ones."""
    origin = acc.origin
    total = acc.total
    age = (t - origin) / acc.half_life
    if age > RENORM_LIMIT:
        # Rescale everything to a new origin at t. exp2 underflows to 0.0
        # for very old totals, which is exactly "fully forgotten".
        total = total * exp2(-age)
        origin = t
        age = 0.0
    total = total + value * exp2(age)
    return DecayedAccumulator(acc.half_life, origin, total)


def read(acc: DecayedAccumulator, t: float) -> float:
    """The decayed sum as of time t (normally: now)."""
    return acc.total * exp2(-(t - acc.origin) / acc.half_life)


def merge(a: DecayedAccumulator, b: DecayedAccumulator) -> DecayedAccumulator:
    """Combine two accumulators as if every contribution had gone into one.

    The older origin's total is rescaled onto the newer origin, then the
    totals are added. Addition of two floats is commutative, so
    merge(a, b) == merge(b, a) exactly.
    """
    if a.half_life != b.half_life:
        raise ValueError("cannot merge accumulators with different half-lives")
    if a.origin < b.origin:
        a, b = b, a
    # a now has the newer (or equal) origin.
    shifted = b.total * exp2((b.origin - a.origin) / a.half_life)
    return DecayedAccumulator(a.half_life, a.origin, a.total + shifted)
