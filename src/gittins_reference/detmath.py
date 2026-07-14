"""Deterministic transcendental math (design doc section 8).

Platform math libraries (libm) disagree in the last bits of functions like
exp2, which breaks cross-platform bit-identity. So the engine vendors its own:
a fixed polynomial evaluated in a fixed order is the *definition* of the
function, and every implementation (Python, Rust, WASM) reproduces it exactly.

Accuracy of this exp2 against a correctly rounded one is within ~1 ulp
(measured worst case 2.4e-16 relative over [0, 1)) — but the spec point is
reproducibility, not correct rounding.
"""

import math

# Taylor coefficients of 2^f = e^(f ln 2): coefficient k is (ln 2)^k / k!.
# These exact double values are the specification; they are never recomputed.
EXP2_COEFFS = (
    1.0,
    0.6931471805599453,
    0.2402265069591007,
    0.055504108664821576,
    0.009618129107628477,
    0.0013333558146428443,
    0.0001540353039338161,
    1.525273380405984e-05,
    1.3215486790144307e-06,
    1.0178086009239699e-07,
    7.054911620801122e-09,
    4.4455382718708106e-10,
    2.56784359934882e-11,
    1.3691488853904124e-12,
    6.778726354822543e-14,
    3.132436707088427e-15,
)


def exp2(x: float) -> float:
    """2 raised to the power x, deterministically.

    Split x into integer part n and fraction f in [0, 1); approximate 2^f with
    the fixed Taylor polynomial (Horner's rule, highest coefficient first);
    scale by 2^n exactly with ldexp. Out-of-range inputs clamp to 0 or
    infinity, which is the correct decayed-weight semantics for both extremes.
    """
    if x != x:  # NaN
        return x
    if x >= 1024.0:  # would overflow f64
        return math.inf
    if x <= -1075.0:  # would underflow f64
        return 0.0
    n = math.floor(x)
    f = x - n
    acc = EXP2_COEFFS[15]
    for k in range(14, -1, -1):
        acc = acc * f + EXP2_COEFFS[k]
    return math.ldexp(acc, n)
