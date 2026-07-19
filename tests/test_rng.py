"""Tests for the counter-based RNG.

The pinned hex constants here are the first golden vectors of the project: any
future implementation (Rust, WASM) must reproduce them bit for bit.
"""

from gittins_reference.rng import (
    derive_key,
    fnv1a_64,
    mix64,
    random_u64,
    random_unit,
)

KEY = derive_key("decision-001", "salt-A")


class TestPinnedVectors:
    def test_fnv1a_known_values(self):
        # Published FNV-1a 64 test values.
        assert fnv1a_64(b"") == 0xCBF29CE484222325
        assert fnv1a_64(b"abc") == 0xE71FA2190541574B

    def test_mix64(self):
        assert mix64(1) == 0x5692161D100B05E5

    def test_derive_key(self):
        assert KEY == 0x93A23219EDB6E287
        assert derive_key("decision-001", "salt-B") == 0x93A23319EDB6E43A

    def test_random_u64(self):
        assert random_u64(KEY, 0) == 0xC6608EC67B2DA337
        assert random_u64(KEY, 1) == 0x722B4FD724B7F479
        assert random_u64(KEY, 2) == 0x4A3B5D64C3443582

    def test_random_unit(self):
        assert random_unit(KEY, 0) == 0.7749108538220555
        assert random_unit(KEY, 1) == 0.44597338678860843
        assert random_unit(KEY, 2) == 0.2899683352473097


class TestProperties:
    def test_same_inputs_same_outputs(self):
        assert random_u64(KEY, 7) == random_u64(KEY, 7)

    def test_key_derivation_is_unambiguous(self):
        # Length-prefixing keeps (id, salt) boundaries distinct.
        assert derive_key("ab", "c") != derive_key("a", "bc")

    def test_different_salt_different_stream(self):
        other = derive_key("decision-001", "salt-B")
        draws = [(random_u64(KEY, c), random_u64(other, c)) for c in range(100)]
        assert all(a != b for a, b in draws)

    def test_unit_range_and_granularity(self):
        for counter in range(10_000):
            u = random_unit(KEY, counter)
            assert 0.0 <= u < 1.0
            # Exactly 53 bits: scaling back up must give a whole number.
            assert (u * 2.0**53) == int(u * 2.0**53)

    def test_unit_mean_is_near_half(self):
        n = 10_000
        mean = sum(random_unit(KEY, c) for c in range(n)) / n
        assert abs(mean - 0.5) < 0.01

    def test_output_bits_are_balanced(self):
        # Each of the 64 output bit positions should be set ~50% of the time.
        n = 4_000
        counts = [0] * 64
        for counter in range(n):
            value = random_u64(KEY, counter)
            for bit in range(64):
                counts[bit] += (value >> bit) & 1
        for bit, count in enumerate(counts):
            assert abs(count / n - 0.5) < 0.05, f"bit {bit} is biased"

    def test_adjacent_counters_decorrelated(self):
        # Consecutive outputs should differ in roughly half their bits.
        for counter in range(100):
            diff = random_u64(KEY, counter) ^ random_u64(KEY, counter + 1)
            assert 16 <= diff.bit_count() <= 48
