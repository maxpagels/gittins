import math
from collections import Counter

import pytest

from gittins_reference.decide import (
    BanditState,
    DecisionRecord,
    candidate_set_hash,
    choose_gamma,
    decide,
    new_bandit,
)
from gittins_reference.exploration import apply_floor, inverse_gap_probabilities
from gittins_reference.model import predict, update

HOUR = 3600.0
DAY = 24 * HOUR
T0 = 1_752_000_000.0

CANDS = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]


def trained_state(n: int = 200) -> BanditState:
    # Arm [1,0] pays 1, arm [0,1] pays 0, observed n times each.
    m = new_bandit(2, HOUR, T0, horizon=DAY).model
    for i in range(n):
        m = update(m, [1.0, 0.0], 1.0, T0 + i)
        m = update(m, [0.0, 1.0], 0.0, T0 + i + 0.5)
    return BanditState(
        model=m, next_seq=0, model_version=2 * n, horizon=DAY, default_reward=0.0, ledger=()
    )


class TestDecide:
    def test_is_deterministic(self):
        s = new_bandit(2, HOUR, T0, horizon=DAY)
        assert decide(s, CANDS, T0, "pepper") == decide(s, CANDS, T0, "pepper")

    def test_decide_never_learns(self):
        # decide never trains (D5: training happens only through the ledger)
        # and never mutates its input; the state changes are the sequence
        # advancing and the record joining the ledger.
        s = new_bandit(2, HOUR, T0, horizon=DAY)
        record, s2 = decide(s, CANDS, T0, "pepper")
        assert s2.model == s.model
        assert s2.model_version == s.model_version
        assert s2.next_seq == s.next_seq + 1
        assert s2.ledger == (record,)
        assert s.next_seq == 0 and s.ledger == ()  # input untouched

    def test_ids_are_unique_and_sequential(self):
        s = new_bandit(2, HOUR, T0, horizon=DAY)
        r1, s = decide(s, CANDS, T0, "pepper")
        r2, s = decide(s, CANDS, T0 + 1, "pepper")
        assert r1.decision_id == "pepper:0"
        assert r2.decision_id == "pepper:1"

    def test_record_is_self_contained(self):
        s = new_bandit(2, HOUR, T0, horizon=DAY)
        record, _ = decide(s, CANDS, T0, "pepper")
        assert record.features == tuple(CANDS[record.chosen])
        assert record.t == T0
        assert record.salt == "pepper"
        assert record.model_version == 0
        assert record.candidate_hash == candidate_set_hash(CANDS)

    def test_propensity_is_the_floored_probability_of_the_choice(self):
        s = trained_state()
        t = T0 + 500.0
        record, _ = decide(s, CANDS, t, "pepper")
        ests, uncs = [], []
        for x in CANDS:
            est, unc = predict(s.model, x, t)
            ests.append(est)
            uncs.append(unc)
        p = apply_floor(inverse_gap_probabilities(ests, choose_gamma(uncs)))
        assert record.propensity == p[record.chosen]

    def test_fresh_state_explores_uniformly(self):
        # A fresh model estimates 0 with equal uncertainty everywhere, so
        # the choice frequencies over many salts are ~uniform.
        s = new_bandit(2, HOUR, T0, horizon=DAY)
        n = 3000
        counts = Counter(decide(s, CANDS, T0, f"salt-{i}")[0].chosen for i in range(n))
        for i in range(3):
            assert math.isclose(counts[i] / n, 1 / 3, abs_tol=0.03)

    def test_trained_state_is_greedy_but_never_certain(self):
        # After clear evidence the good arm dominates, but the bad arm's
        # propensity keeps a floor above FLOOR_MASS / k (R2).
        s = trained_state()
        n = 2000
        cands = [[1.0, 0.0], [0.0, 1.0]]
        records = [decide(s, cands, T0 + 500, f"s{i}")[0] for i in range(n)]
        counts = Counter(r.chosen for r in records)
        assert counts[0] / n > 0.85
        bad = next(r for r in records if r.chosen == 1)
        assert bad.propensity >= 0.05 / 2

    def test_gamma_grows_as_uncertainty_shrinks(self):
        # Same candidates, fresh vs. trained state: the trained state is
        # greedier — the losing arm's propensity has dropped.
        cands = [[1.0, 0.0], [0.0, 1.0]]
        fresh, _ = decide(new_bandit(2, HOUR, T0, horizon=DAY), cands, T0, "z")
        greedy, _ = decide(trained_state(), cands, T0 + 500, "z")
        p_fresh = 1.0 - fresh.propensity if fresh.chosen == 0 else fresh.propensity
        p_greedy = 1.0 - greedy.propensity if greedy.chosen == 0 else greedy.propensity
        assert p_greedy < p_fresh / 3

    def test_rejects_bad_candidates(self):
        s = new_bandit(2, HOUR, T0, horizon=DAY)
        with pytest.raises(ValueError):
            decide(s, [], T0, "pepper")
        with pytest.raises(ValueError):
            decide(s, [[1.0, 0.0, 0.0]], T0, "pepper")


class TestChooseGamma:
    def test_inverse_mean_uncertainty(self):
        assert choose_gamma([0.5, 0.25]) == 1.0 / 0.375

    def test_all_zero_uncertainty_gives_uniform(self):
        assert choose_gamma([0.0, 0.0]) == 0.0


class TestCandidateSetHash:
    def test_sensitive_to_values_order_and_shape(self):
        h = candidate_set_hash([[1.0, 0.0], [0.0, 1.0]])
        assert candidate_set_hash([[1.0, 0.0], [0.0, 1.0]]) == h
        assert candidate_set_hash([[0.0, 1.0], [1.0, 0.0]]) != h
        assert candidate_set_hash([[1.0, 0.5], [0.0, 1.0]]) != h
        # [[1,0],[0,1]] flattened equals [[1],[0,0],[1]] flattened; the
        # per-vector length prefixes must keep them distinct.
        assert candidate_set_hash([[1.0], [0.0, 0.0], [1.0]]) != h


class TestPinnedVectors:
    # Golden vector: bit-exact decision record for a fixed little history.
    # Model dim 2, half-life 3600 s, created at T0; updates ([1,0], 1.0, T0)
    # and ([0,1], -0.5, T0+1800); state at seq 7, version 2; candidates
    # CANDS; decide at T0+3600 with salt "pepper".
    def test_record_bits(self):
        m = new_bandit(2, HOUR, T0, horizon=DAY).model
        m = update(m, [1.0, 0.0], 1.0, T0)
        m = update(m, [0.0, 1.0], -0.5, T0 + 1800.0)
        s = BanditState(
            model=m, next_seq=7, model_version=2, horizon=DAY, default_reward=0.0, ledger=()
        )
        record, s2 = decide(s, CANDS, T0 + 3600.0, "pepper")
        assert record == DecisionRecord(
            decision_id="pepper:7",
            t=1752003600.0,
            candidate_hash=14029619844508309728,
            chosen=0,
            features=(1.0, 0.0),
            propensity=0.4086828696415729,
            model_version=2,
            salt="pepper",
        )
        assert s2.next_seq == 8
