import math

from gittins_reference.decay import (
    RENORM_LIMIT,
    new_accumulator,
    add,
    read,
    merge,
)
from gittins_reference.detmath import exp2

HOUR = 3600.0
T0 = 1_752_000_000.0  # an arbitrary realistic Unix timestamp


class TestDecaySemantics:
    def test_immediate_read_returns_value(self):
        acc = add(new_accumulator(HOUR, T0), 10.0, T0)
        assert math.isclose(read(acc, T0), 10.0, rel_tol=1e-12)

    def test_one_half_life_halves(self):
        acc = add(new_accumulator(HOUR, T0), 10.0, T0)
        assert math.isclose(read(acc, T0 + HOUR), 5.0, rel_tol=1e-12)

    def test_ten_half_lives(self):
        acc = add(new_accumulator(HOUR, T0), 1024.0, T0)
        assert math.isclose(read(acc, T0 + 10 * HOUR), 1.0, rel_tol=1e-12)

    def test_contributions_decay_independently(self):
        acc = new_accumulator(HOUR, T0)
        acc = add(acc, 8.0, T0)  # will be 2 half-lives old at the read
        acc = add(acc, 8.0, T0 + HOUR)  # will be 1 half-life old at the read
        assert math.isclose(read(acc, T0 + 2 * HOUR), 2.0 + 4.0, rel_tol=1e-12)

    def test_infinite_half_life_never_forgets(self):
        acc = add(new_accumulator(math.inf, T0), 3.0, T0)
        century = 100 * 365 * 24 * HOUR
        assert read(acc, T0 + century) == 3.0

    def test_counting_uses_the_same_machinery(self):
        acc = new_accumulator(HOUR, T0)
        for i in range(4):
            acc = add(acc, 1.0, T0 + i)  # four uses within a few seconds
        expected = sum(exp2(-(3 - i) / HOUR) for i in range(4))  # ~3.9988
        assert math.isclose(read(acc, T0 + 3), expected, rel_tol=1e-12)


class TestLateRewards:
    def test_late_add_equals_on_time_add_bit_for_bit(self):
        # Within one renormalization era, add order is exactly irrelevant:
        # a reward that arrives late (added after later data) produces the
        # same bits as if it had arrived on time.
        on_time = add(add(new_accumulator(HOUR, T0), 1.0, T0), 5.0, T0 + 9 * HOUR)
        late = add(add(new_accumulator(HOUR, T0), 5.0, T0 + 9 * HOUR), 1.0, T0)
        assert on_time == late

    def test_late_add_across_renormalization_is_equivalent(self):
        far = T0 + (RENORM_LIMIT + 10) * HOUR  # this add forces a renorm
        on_time = add(add(new_accumulator(HOUR, T0), 1.0, T0), 5.0, far)
        late = add(add(new_accumulator(HOUR, T0), 5.0, far), 1.0, T0)
        assert math.isclose(read(on_time, far), read(late, far), rel_tol=1e-12)

    def test_very_late_reward_is_fully_forgotten_not_an_error(self):
        acc = add(new_accumulator(HOUR, T0), 5.0, T0)
        acc = add(acc, 100.0, T0 - 1_000_000 * HOUR)  # ancient decision
        assert math.isclose(read(acc, T0), 5.0, rel_tol=1e-12)


class TestRenormalization:
    def test_renorm_preserves_readings(self):
        acc = add(new_accumulator(HOUR, T0), 7.0, T0)
        t1 = T0 + (RENORM_LIMIT + 1) * HOUR
        renormed = add(acc, 0.0, t1)  # forces origin to advance
        assert renormed.origin == t1
        assert math.isclose(read(renormed, t1), read(acc, t1), rel_tol=1e-12)

    def test_stored_total_stays_in_range_over_long_use(self):
        # 10 simulated years of daily adds with a one-hour half-life: the
        # naive weight 2^(t/H) would overflow after ~13 days.
        acc = new_accumulator(HOUR, T0)
        for day in range(3650):
            acc = add(acc, 1.0, T0 + day * 24 * HOUR)
            assert math.isfinite(acc.total)
        # Today's 1.0 plus the geometric residue of every earlier day.
        expected = 1.0 / (1.0 - 2.0**-24)
        assert math.isclose(read(acc, T0 + 3649 * 24 * HOUR), expected, rel_tol=1e-9)


class TestMerge:
    def test_merge_is_commutative_bit_for_bit(self):
        a = add(add(new_accumulator(HOUR, T0), 2.0, T0), 3.0, T0 + HOUR)
        b = add(new_accumulator(HOUR, T0), 4.0, T0 + 5 * HOUR)
        assert merge(a, b) == merge(b, a)

    def test_merge_commutative_across_renorm_eras(self):
        a = add(new_accumulator(HOUR, T0), 2.0, T0)
        b = add(new_accumulator(HOUR, T0), 4.0, T0 + (RENORM_LIMIT + 50) * HOUR)
        assert merge(a, b) == merge(b, a)

    def test_merge_equals_single_accumulator(self):
        # Fleet mode: two agents' accumulators merge into what one central
        # accumulator would have seen.
        a = add(add(new_accumulator(HOUR, T0), 1.0, T0), 2.0, T0 + HOUR)
        b = add(new_accumulator(HOUR, T0), 4.0, T0 + 2 * HOUR)
        central = add(
            add(add(new_accumulator(HOUR, T0), 1.0, T0), 2.0, T0 + HOUR),
            4.0,
            T0 + 2 * HOUR,
        )
        t = T0 + 3 * HOUR
        assert math.isclose(read(merge(a, b), t), read(central, t), rel_tol=1e-12)

    def test_merge_rejects_mismatched_half_lives(self):
        import pytest

        with pytest.raises(ValueError):
            merge(new_accumulator(HOUR, T0), new_accumulator(2 * HOUR, T0))


class TestPinnedVectors:
    # Golden vectors: bit-exact state after a fixed little history.
    def test_state_bits(self):
        acc = new_accumulator(HOUR, T0)
        acc = add(acc, 1.0, T0)
        acc = add(acc, 2.5, T0 + 5400.0)
        acc = add(acc, -0.75, T0 + 2 * HOUR)
        # No add crossed RENORM_LIMIT, so the origin is still creation time.
        assert acc.origin == 1752000000.0
        assert acc.total == 5.0710678118654755
        assert read(acc, T0 + 3 * HOUR) == 0.6338834764831844
