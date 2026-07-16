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


@dataclass(frozen=True)
class DecayedVector:
    """Many decaying sums sharing one clock: one half-life, one origin, one
    renormalization decision for every entry. Semantics per entry are exactly
    those of DecayedAccumulator; sharing the clock is what lets a model's
    whole state age coherently and merge entry-by-entry."""

    half_life: float
    origin: float
    values: tuple[float, ...]


def new_vector(half_life: float, t: float, size: int) -> DecayedVector:
    if not (half_life > 0.0):
        raise ValueError("half_life must be positive (math.inf is allowed)")
    return DecayedVector(half_life=half_life, origin=t, values=(0.0,) * size)


def vector_add(
    vec: DecayedVector, contributions: "tuple[float, ...] | list[float]", t: float
) -> DecayedVector:
    """Add one contribution per entry, all timestamped `t`."""
    origin = vec.origin
    values = vec.values
    age = (t - origin) / vec.half_life
    if age > RENORM_LIMIT:
        rescale = exp2(-age)
        values = tuple(v * rescale for v in values)
        origin = t
        age = 0.0
    weight = exp2(age)
    values = tuple([v + c * weight for v, c in zip(values, contributions, strict=True)])
    return DecayedVector(vec.half_life, origin, values)


def vector_read(vec: DecayedVector, t: float) -> tuple[float, ...]:
    factor = exp2(-(t - vec.origin) / vec.half_life)
    return tuple([v * factor for v in vec.values])


def vector_merge(a: DecayedVector, b: DecayedVector) -> DecayedVector:
    """Entry-by-entry merge; commutative bit for bit, like the scalar merge."""
    if a.half_life != b.half_life:
        raise ValueError("cannot merge vectors with different half-lives")
    if len(a.values) != len(b.values):
        raise ValueError("cannot merge vectors of different sizes")
    if a.origin < b.origin:
        a, b = b, a
    shift = exp2((b.origin - a.origin) / a.half_life)
    values = tuple(av + bv * shift for av, bv in zip(a.values, b.values, strict=True))
    return DecayedVector(a.half_life, a.origin, values)


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
