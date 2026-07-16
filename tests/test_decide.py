import math
from collections import Counter

import pytest

from gittins_reference.decide import (
    BanditState,
    DecisionRecord,
    candidate_set_hash,
    decide,
    new_bandit,
)
from gittins_reference.exploration import DEFAULT_EPSILON, epsilon_greedy_probabilities
from gittins_reference.model import predict, update

HOUR = 3600.0
DAY = 24 * HOUR
T0 = 1_752_000_000.0

CANDS = [((0, 1.0),), ((1, 1.0),), ((0, 1.0), (1, 1.0))]


def trained_state(n: int = 200) -> BanditState:
    # The arm on feature 0 pays 1, the arm on feature 1 pays 0, observed
    # n times each.
    m = new_bandit(2, horizon=DAY).model
    for _ in range(n):
        m = update(m, ((0, 1.0),), 1.0)
        m = update(m, ((1, 1.0),), 0.0)
    return BanditState(
        model=m,
        next_seq=0,
        model_version=2 * n,
        horizon=DAY,
        default_reward=0.0,
        epsilon=DEFAULT_EPSILON,
        ledger=(),
    )


class TestDecide:
    def test_is_deterministic(self):
        s = new_bandit(2, horizon=DAY)
        assert decide(s, CANDS, T0, "pepper") == decide(s, CANDS, T0, "pepper")

    def test_decide_never_learns(self):
        # decide never trains (D5: training happens only through the ledger)
        # and never mutates its input; the state changes are the sequence
        # advancing and the record joining the ledger.
        s = new_bandit(2, horizon=DAY)
        record, s2 = decide(s, CANDS, T0, "pepper")
        assert s2.model == s.model
        assert s2.model_version == s.model_version
        assert s2.next_seq == s.next_seq + 1
        assert s2.ledger == (record,)
        assert s.next_seq == 0 and s.ledger == ()  # input untouched

    def test_ids_are_unique_and_sequential(self):
        s = new_bandit(2, horizon=DAY)
        r1, s = decide(s, CANDS, T0, "pepper")
        r2, s = decide(s, CANDS, T0 + 1, "pepper")
        assert r1.decision_id == "pepper:0"
        assert r2.decision_id == "pepper:1"

    def test_record_is_self_contained(self):
        s = new_bandit(2, horizon=DAY)
        record, _ = decide(s, CANDS, T0, "pepper")
        assert record.features == tuple(CANDS[record.chosen])
        assert record.t == T0
        assert record.salt == "pepper"
        assert record.model_version == 0
        assert record.candidate_hash == candidate_set_hash(CANDS, 2)

    def test_propensity_is_the_probability_of_the_choice(self):
        s = trained_state()
        t = T0 + 500.0
        record, _ = decide(s, CANDS, t, "pepper")
        ests = [predict(s.model, x)[0] for x in CANDS]
        p = epsilon_greedy_probabilities(ests, s.epsilon)
        assert record.propensity == p[record.chosen]

    def test_fresh_state_explores_uniformly(self):
        # A fresh model estimates every candidate at 0.0, and the exact-tie
        # split makes the distribution uniform: the choice frequencies over
        # many salts are ~uniform, not a lock on index 0.
        s = new_bandit(2, horizon=DAY)
        n = 3000
        counts = Counter(decide(s, CANDS, T0, f"salt-{i}")[0].chosen for i in range(n))
        for i in range(3):
            assert math.isclose(counts[i] / n, 1 / 3, abs_tol=0.03)

    def test_trained_state_is_greedy_but_never_certain(self):
        # After clear evidence the good arm dominates, but the bad arm's
        # propensity keeps exactly epsilon / k (R2).
        s = trained_state()
        n = 2000
        cands = [((0, 1.0),), ((1, 1.0),)]
        records = [decide(s, cands, T0 + 500, f"s{i}")[0] for i in range(n)]
        counts = Counter(r.chosen for r in records)
        assert counts[0] / n > 0.9
        bad = next(r for r in records if r.chosen == 1)
        assert bad.propensity == DEFAULT_EPSILON / 2

    def test_evidence_concentrates_the_distribution(self):
        # Same candidates, fresh vs. trained state: fresh estimates tie and
        # split the greedy mass (each arm at 1/2); trained evidence drops
        # the losing arm to exactly epsilon / 2.
        cands = [((0, 1.0),), ((1, 1.0),)]
        fresh, _ = decide(new_bandit(2, horizon=DAY), cands, T0, "z")
        greedy, _ = decide(trained_state(), cands, T0 + 500, "z")
        assert fresh.propensity == 0.5
        p_greedy = 1.0 - greedy.propensity if greedy.chosen == 0 else greedy.propensity
        assert p_greedy == DEFAULT_EPSILON / 2

    def test_epsilon_override_shapes_the_distribution(self):
        # epsilon is declared once at construction and every decision
        # spends exactly that mass uniformly.
        s = trained_state()
        s = BanditState(
            model=s.model,
            next_seq=0,
            model_version=s.model_version,
            horizon=DAY,
            default_reward=0.0,
            epsilon=0.2,
            ledger=(),
        )
        cands = [((0, 1.0),), ((1, 1.0),)]
        records = [decide(s, cands, T0 + 500, f"s{i}")[0] for i in range(200)]
        bad = next(r for r in records if r.chosen == 1)
        assert bad.propensity == 0.1

    def test_rejects_bad_candidates(self):
        s = new_bandit(2, horizon=DAY)
        with pytest.raises(ValueError):
            decide(s, [], T0, "pepper")
        with pytest.raises(ValueError):  # index outside the model dimension
            decide(s, [((2, 1.0),)], T0, "pepper")
        with pytest.raises(ValueError):  # indices not strictly increasing
            decide(s, [((1, 1.0), (0, 1.0))], T0, "pepper")
        with pytest.raises(ValueError):  # duplicate index
            decide(s, [((0, 1.0), (0, 1.0))], T0, "pepper")
        with pytest.raises(ValueError):  # explicit zero entry
            decide(s, [((0, 0.0),)], T0, "pepper")

    def test_rejects_bad_epsilon(self):
        with pytest.raises(ValueError):
            new_bandit(2, horizon=DAY, epsilon=-0.1)
        with pytest.raises(ValueError):
            new_bandit(2, horizon=DAY, epsilon=1.5)


class TestCandidateSetHash:
    def test_sensitive_to_values_order_shape_and_dimension(self):
        h = candidate_set_hash([((0, 1.0),), ((1, 1.0),)], 2)
        assert candidate_set_hash([((0, 1.0),), ((1, 1.0),)], 2) == h
        assert candidate_set_hash([((1, 1.0),), ((0, 1.0),)], 2) != h  # order
        assert candidate_set_hash([((0, 1.0), (1, 0.5)), ((1, 1.0),)], 2) != h  # values
        assert candidate_set_hash([((0, 1.0),), ((1, 1.0),)], 4) != h  # dimension
        # The count prefixes keep same-flattening sets distinct: two
        # candidates vs. those entries split across three.
        assert candidate_set_hash([((0, 1.0),), (), ((1, 1.0),)], 2) != h

    def test_matches_the_former_dense_formulation(self):
        # The byte layout is unchanged from the dense-input days: the same
        # logical candidate set (dense [[1,0],[0,1]], dim 2) still hashes
        # to the value pinned in the golden episode's history.
        assert candidate_set_hash(CANDS, 2) == 8340395383735871362


class TestPinnedVectors:
    # Golden vector: bit-exact decision record for a fixed little history.
    # Model dim 2, default forgetting; updates on feature 0 (reward 1.0)
    # then feature 1 (reward -0.5); state at seq 7, version 2; candidates
    # CANDS; decide at T0+3600 with salt "pepper".
    def test_record_bits(self):
        m = new_bandit(2, horizon=DAY).model
        m = update(m, ((0, 1.0),), 1.0)
        m = update(m, ((1, 1.0),), -0.5)
        s = BanditState(
            model=m,
            next_seq=7,
            model_version=2,
            horizon=DAY,
            default_reward=0.0,
            epsilon=DEFAULT_EPSILON,
            ledger=(),
        )
        record, s2 = decide(s, CANDS, T0 + 3600.0, "pepper")
        assert record == DecisionRecord(
            decision_id="pepper:7",
            t=1752003600.0,
            candidate_hash=8340395383735871362,
            chosen=0,
            features=((0, 1.0),),
            propensity=0.9666666666666667,
            model_version=2,
            salt="pepper",
        )
        assert s2.next_seq == 8
