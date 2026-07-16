"""Seeded randomness for the harness.

Simulations are statistical, never bit-pinned — but every run must replay
exactly, so nothing here touches a stateful generator. Every draw is the
counter RNG from the reference (rng.py), on a stream keyed by
(name, seed, label): an environment's round `t` is a pure function of
(environment name, seed, t), identical no matter which policy is being
driven, so comparators on the same seed face the same world — paired by
construction.
"""

import math

from gittins_reference.rng import derive_key, random_unit


def stream(name: str, seed: int, label: str) -> int:
    """The 64-bit key of one named draw stream within one (name, seed) run."""
    return derive_key(f"{name}:{seed}:{label}", "sim")


def uniform(key: int, counter: int, lo: float = 0.0, hi: float = 1.0) -> float:
    """A uniform float in [lo, hi) at position `counter` of stream `key`."""
    return lo + (hi - lo) * random_unit(key, counter)


def randint(key: int, counter: int, n: int) -> int:
    """A uniform integer in [0, n) at position `counter` of stream `key`."""
    i = int(random_unit(key, counter) * n)
    return n - 1 if i >= n else i


def gaussian(key: int, counter: int) -> float:
    """A standard normal draw (Box-Muller); consumes stream positions
    2*counter and 2*counter + 1.

    The one piece of the harness that is not bit-portable across languages:
    libm log/cos carry no cross-platform rounding guarantee (everything else
    in a sim run is counter-RNG uniforms plus the engine's own bit-pinned
    arithmetic). Kept because sims are statistical; if the battery is ever
    promoted to a cross-language differential test against the compiled
    core, vendor deterministic log/cos here (pinned polynomial
    coefficients, fixed evaluation order — the treatment the reference once
    gave exp2)."""
    u1 = random_unit(key, 2 * counter)
    u2 = random_unit(key, 2 * counter + 1)
    if u1 <= 0.0:  # log(0) guard; random_unit can return exactly 0.0
        u1 = 2.0**-53
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
