import math
from collections import Counter

import pytest

from gittins_reference.exploration import (
    DEFAULT_EPSILON,
    choose,
    epsilon_greedy_probabilities,
    sample_index,
)
from gittins_reference.rng import derive_key

KEY = derive_key("decision-0001", "pepper")


class TestEpsilonGreedy:
    def test_greedy_mass_on_the_best(self):
        p = epsilon_greedy_probabilities([0.3, -1.0, 0.9, 0.0], epsilon=0.05)
        assert p[2] == 0.05 / 4 + 0.95
        assert p[0] == p[1] == p[3] == 0.05 / 4

    def test_sums_to_about_one(self):
        p = epsilon_greedy_probabilities([0.3, -1.0, 0.9, 0.0], epsilon=0.05)
        assert math.isclose(sum(p), 1.0, rel_tol=1e-15)

    def test_epsilon_one_is_uniform(self):
        assert epsilon_greedy_probabilities([0.9, 0.5, 0.1, 0.0], epsilon=1.0) == [0.25] * 4

    def test_epsilon_zero_is_argmax(self):
        assert epsilon_greedy_probabilities([0.0, 1.0], epsilon=0.0) == [0.0, 1.0]

    def test_every_probability_at_least_epsilon_over_k(self):
        # Even the worst candidate keeps epsilon / k: importance weights in
        # offline evaluation are bounded by k / epsilon (R3), and every arm
        # keeps accumulating evidence (R2).
        p = epsilon_greedy_probabilities([1.0, -100.0], epsilon=DEFAULT_EPSILON)
        assert min(p) == DEFAULT_EPSILON / 2

    def test_ties_split_the_greedy_mass(self):
        # Exact ties share (1 - epsilon) equally — no first-index favoritism.
        p = epsilon_greedy_probabilities([0.5, -0.25, 0.0, 0.5], epsilon=0.05)
        assert p[0] == p[3] == 0.05 / 4 + 0.95 / 2
        assert p[1] == p[2] == 0.05 / 4

    def test_all_tied_is_uniform(self):
        # A fresh model estimates every candidate at 0.0: the distribution
        # must be uniform, not a point mass on index 0 (cold start).
        p = epsilon_greedy_probabilities([0.0, 0.0, 0.0, 0.0], epsilon=0.05)
        assert p == [0.05 / 4 + 0.95 / 4] * 4
        assert math.isclose(sum(p), 1.0, rel_tol=1e-15)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            epsilon_greedy_probabilities([], epsilon=0.05)
        with pytest.raises(ValueError):
            epsilon_greedy_probabilities([0.5], epsilon=-0.1)
        with pytest.raises(ValueError):
            epsilon_greedy_probabilities([0.5], epsilon=1.5)
        with pytest.raises(ValueError):
            epsilon_greedy_probabilities([0.5], epsilon=float("nan"))


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
    def test_propensity_is_the_probability_of_the_choice(self):
        est = [0.5, -0.25, 0.0, 0.5]
        p = epsilon_greedy_probabilities(est, epsilon=0.05)
        i, prop = choose(est, epsilon=0.05, key=KEY, counter=0)
        assert prop == p[i]

    def test_single_candidate_is_certain(self):
        assert choose([0.42], epsilon=0.05, key=KEY, counter=0) == (0, 1.0)

    def test_mostly_greedy(self):
        est = [0.0, 1.0, 0.5]
        n = 10_000
        counts = Counter(choose(est, DEFAULT_EPSILON, KEY, c)[0] for c in range(n))
        assert math.isclose(counts[1] / n, 1.0 - DEFAULT_EPSILON * 2 / 3, abs_tol=0.01)


class TestPinnedVectors:
    # Golden vectors: bit-exact distribution and choices for a fixed input.
    # estimates [0.5, -0.25, 0.0, 0.5] (a tie for the maximum), epsilon 0.05,
    # key from ("decision-0001", "pepper").
    def test_distribution_bits(self):
        p = epsilon_greedy_probabilities([0.5, -0.25, 0.0, 0.5], epsilon=0.05)
        assert p == [0.4875, 0.0125, 0.0125, 0.4875]

    def test_choice_bits(self):
        est = [0.5, -0.25, 0.0, 0.5]
        assert choose(est, 0.05, KEY, 0) == (0, 0.4875)
        assert choose(est, 0.05, KEY, 1) == (3, 0.4875)
        assert choose(est, 0.05, KEY, 2) == (0, 0.4875)
