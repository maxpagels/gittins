import math
from collections import Counter

import pytest

from gittins_reference.exploration import (
    FLOOR_MASS,
    apply_floor,
    choose,
    inverse_gap_probabilities,
    sample_index,
)
from gittins_reference.rng import derive_key

KEY = derive_key("decision-0001", "pepper")


class TestInverseGap:
    def test_sums_to_exactly_one(self):
        # The best candidate's probability is defined as 1 minus the others,
        # so the sum is 1.0 in exact float arithmetic, not just approximately.
        p = inverse_gap_probabilities([0.3, -1.0, 0.9, 0.0], gamma=7.0)
        assert sum(p) == 1.0

    def test_best_gets_most_and_gaps_order_the_rest(self):
        p = inverse_gap_probabilities([0.9, 0.5, 0.1], gamma=5.0)
        assert p[0] > p[1] > p[2]

    def test_gamma_zero_is_uniform(self):
        # With no greediness every non-best candidate gets exactly 1/k,
        # and the best keeps the identical remainder.
        assert inverse_gap_probabilities([0.9, 0.5, 0.1, 0.0], gamma=0.0) == [0.25] * 4

    def test_large_gamma_approaches_argmax(self):
        p = inverse_gap_probabilities([1.0, 0.0], gamma=1e9)
        assert p[0] > 1.0 - 1e-8

    def test_non_best_probability_never_exceeds_one_over_k(self):
        # Structural guarantee: gaps are >= 0, so 1/(k + gamma*gap) <= 1/k,
        # which is what keeps the best candidate's remainder non-negative.
        p = inverse_gap_probabilities([0.5, 0.5 - 1e-12, 0.499], gamma=0.001)
        assert all(v <= 1.0 / 3.0 for i, v in enumerate(p) if i != 0)
        assert p[0] >= 1.0 / 3.0

    def test_first_max_wins_ties(self):
        # A candidate tied with the best has gap 0 and gets exactly 1/k;
        # the *first* max holds the remainder. Deterministic, order-fixed.
        p = inverse_gap_probabilities([0.5, -0.25, 0.0, 0.5], gamma=10.0)
        assert p[3] == 0.25
        assert p[0] == 1.0 - (p[1] + p[2] + p[3])

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            inverse_gap_probabilities([], gamma=1.0)
        with pytest.raises(ValueError):
            inverse_gap_probabilities([0.5], gamma=-1.0)
        with pytest.raises(ValueError):
            inverse_gap_probabilities([0.5], gamma=float("nan"))


class TestFloor:
    def test_every_probability_at_least_floor(self):
        # Even a candidate driven to ~zero probability by a huge gamma
        # keeps at least FLOOR_MASS / k after the floor.
        p = apply_floor(inverse_gap_probabilities([1.0, 0.0], gamma=1e9))
        assert min(p) >= FLOOR_MASS / 2

    def test_preserves_total_mass(self):
        p = apply_floor([0.7, 0.2, 0.1])
        assert math.isclose(sum(p), 1.0, rel_tol=1e-15)

    def test_preserves_order(self):
        # The floor is a positive-slope affine map: it never reorders
        # candidates, so the greedy choice is unchanged.
        p = apply_floor([0.7, 0.2, 0.1])
        assert p[0] > p[1] > p[2]

    def test_zero_floor_is_identity(self):
        assert apply_floor([0.7, 0.2, 0.1], floor_mass=0.0) == [0.7, 0.2, 0.1]

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            apply_floor([])
        with pytest.raises(ValueError):
            apply_floor([1.0], floor_mass=1.5)


class TestSample:
    def test_deterministic(self):
        p = [0.5, 0.3, 0.2]
        assert sample_index(p, KEY, 7) == sample_index(p, KEY, 7)

    def test_frequencies_match_probabilities(self):
        p = [0.5, 0.3, 0.2]
        n = 100_000
        counts = Counter(sample_index(p, KEY, c) for c in range(n))
        for i in range(3):
            assert math.isclose(counts[i] / n, p[i], abs_tol=0.01)

    def test_rounding_gap_falls_to_last_index(self):
        # If the cumulative sum falls short of the draw (only possible
        # through rounding), the last index is chosen rather than crashing.
        assert sample_index([0.0, 0.0], KEY, 0) == 1


class TestChoose:
    def test_propensity_is_the_floored_probability_of_the_choice(self):
        est = [0.5, -0.25, 0.0, 0.5]
        p = apply_floor(inverse_gap_probabilities(est, gamma=10.0))
        i, prop = choose(est, gamma=10.0, key=KEY, counter=0)
        assert prop == p[i]

    def test_single_candidate_is_certain(self):
        assert choose([0.42], gamma=3.0, key=KEY, counter=0) == (0, 1.0)


class TestPinnedVectors:
    # Golden vectors: bit-exact distribution and choices for a fixed input.
    # estimates [0.5, -0.25, 0.0, 0.5], gamma 10, key from
    # ("decision-0001", "pepper"), FLOOR_MASS 0.05.
    def test_distribution_bits(self):
        p = apply_floor(inverse_gap_probabilities([0.5, -0.25, 0.0, 0.5], gamma=10.0))
        assert p == [
            0.5368357487922705,
            0.0951086956521739,
            0.11805555555555554,
            0.25,
        ]

    def test_choice_bits(self):
        est = [0.5, -0.25, 0.0, 0.5]
        assert choose(est, 10.0, KEY, 0) == (0, 0.5368357487922705)
        assert choose(est, 10.0, KEY, 1) == (1, 0.0951086956521739)
        assert choose(est, 10.0, KEY, 2) == (0, 0.5368357487922705)
