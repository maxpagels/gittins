"""Counter-based deterministic random number generation (design doc section 8).

The engine never draws from a stateful random generator. Instead, every random
number is a pure function of (key, counter):

    key     = derive_key(decision_id, salt)   -- fixed per decision
    counter = 0, 1, 2, ...                    -- position within that decision

so any single historical decision can be reproduced in isolation: no need to
replay everything that came before it to recover the generator's state.

The construction is splitmix64: the value at position `counter` of the stream
seeded with `key` is `mix64(key + counter * GOLDEN)`. Because each position is
computed directly, it is counter-based by definition.

All arithmetic is unsigned 64-bit with wraparound (the `& MASK64` after every
addition and multiplication). The Rust core expresses the same operations as
`wrapping_add` / `wrapping_mul`.
"""

MASK64 = 0xFFFFFFFFFFFFFFFF

# splitmix64 constants (Steele, Lea & Flood 2014; public domain reference code).
GOLDEN = 0x9E3779B97F4A7C15
MIX_A = 0xBF58476D1CE4E5B9
MIX_B = 0x94D049BB133111EB

# FNV-1a 64-bit constants.
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def fnv1a_64(data: bytes) -> int:
    """Hash bytes to a 64-bit integer with FNV-1a."""
    h = FNV_OFFSET
    for byte in data:
        h = h ^ byte
        h = (h * FNV_PRIME) & MASK64
    return h


def mix64(x: int) -> int:
    """The splitmix64 finalizer: scrambles a 64-bit value so that similar
    inputs give unrelated outputs."""
    x = (x ^ (x >> 30)) * MIX_A & MASK64
    x = (x ^ (x >> 27)) * MIX_B & MASK64
    x = x ^ (x >> 31)
    return x


def derive_key(decision_id: str, salt: str) -> int:
    """Derive the 64-bit stream key for one decision.

    The decision ID is length-prefixed (8 bytes, little-endian) before the salt
    is appended, so ("ab", "c") and ("a", "bc") can never produce the same key.
    """
    id_bytes = decision_id.encode("utf-8")
    salt_bytes = salt.encode("utf-8")
    message = len(id_bytes).to_bytes(8, "little") + id_bytes + salt_bytes
    return fnv1a_64(message)


def random_u64(key: int, counter: int) -> int:
    """The uniform 64-bit value at position `counter` of stream `key`."""
    return mix64((key + counter * GOLDEN) & MASK64)


def random_unit(key: int, counter: int) -> float:
    """A uniform float in [0, 1) with exactly 53 random bits.

    The top 53 bits of the u64 are scaled by 2^-53. Every step is exact in
    IEEE-754 double precision, so the result is bit-identical on every
    platform.
    """
    return (random_u64(key, counter) >> 11) * 2.0**-53
