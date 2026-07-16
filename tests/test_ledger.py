from collections import Counter

from gittins_reference.decide import decide, new_bandit
from gittins_reference.ledger import (
    CENSORED,
    EXPIRED,
    REWARDED,
    Resolution,
    censor,
    expire,
    learn,
    take,
)
from gittins_reference.model import predict, update

HOUR = 3600.0
DAY = 24 * HOUR
T0 = 1_752_000_000.0

CANDS = [((0, 1.0),), ((1, 1.0),)]


def fresh(default_reward: float = 0.0):
    return new_bandit(2, horizon=DAY, default_reward=default_reward)


class TestTake:
    def test_removes_exactly_the_named_record(self):
        s = fresh()
        r1, s = decide(s, CANDS, T0, "pepper")
        r2, s = decide(s, CANDS, T0 + 1, "pepper")
        found, rest = take(s.ledger, r1.decision_id)
        assert found == r1
        assert rest == (r2,)

    def test_unknown_id_leaves_ledger_alone(self):
        s = fresh()
        _, s = decide(s, CANDS, T0, "pepper")
        found, rest = take(s.ledger, "nope:0")
        assert found is None
        assert rest == s.ledger


class TestLearn:
    def test_resolves_and_trains_on_the_recorded_features(self):
        s0 = fresh()
        record, s1 = decide(s0, CANDS, T0, "pepper")
        resolution, s2 = learn(s1, record.decision_id, 1.0)
        assert resolution == Resolution(record.decision_id, REWARDED, 1.0)
        assert s2.ledger == ()
        assert s2.model_version == s1.model_version + 1
        # The training update is exactly one model update on the record's
        # features — the record is self-contained.
        assert s2.model == update(s1.model, record.features, 1.0)

    def test_out_of_order_rewards_resolve_safely(self):
        # Two open decisions resolved in either order: both orders fully
        # resolve the ledger with nothing lost or double-counted. The
        # *models* differ (training is order-dependent under per-update
        # forgetting — a late reward counts slightly less), which is the
        # documented trade for dropping wall-clock decay.
        s = fresh()
        r1, s = decide(s, CANDS, T0, "pepper")
        r2, s = decide(s, CANDS, T0 + 9 * HOUR, "pepper")
        _, a = learn(s, r1.decision_id, 1.0)
        _, a = learn(a, r2.decision_id, -0.5)
        _, b = learn(s, r2.decision_id, -0.5)
        _, b = learn(b, r1.decision_id, 1.0)
        assert a.ledger == () and b.ledger == ()
        assert a.model_version == b.model_version == 2
        assert a.model != b.model

    def test_duplicate_report_is_ignored(self):
        s = fresh()
        record, s = decide(s, CANDS, T0, "pepper")
        _, s1 = learn(s, record.decision_id, 1.0)
        resolution, s2 = learn(s1, record.decision_id, 1.0)
        assert resolution is None
        assert s2 == s1

    def test_conflicting_duplicate_is_also_ignored(self):
        # Only the first report of an outcome counts; a later, different
        # value finds the decision already spent.
        s = fresh()
        record, s = decide(s, CANDS, T0, "pepper")
        _, s1 = learn(s, record.decision_id, 1.0)
        resolution, s2 = learn(s1, record.decision_id, 0.0)
        assert resolution is None
        assert s2 == s1

    def test_unknown_id_is_a_no_op(self):
        s = fresh()
        resolution, s2 = learn(s, "never-made:0", 1.0)
        assert resolution is None
        assert s2 == s


class TestCensor:
    def test_excludes_from_training_but_logs_the_exclusion(self):
        s = fresh()
        record, s1 = decide(s, CANDS, T0, "pepper")
        resolution, s2 = censor(s1, record.decision_id)
        assert resolution == Resolution(record.decision_id, CENSORED, None)
        assert s2.ledger == ()
        assert s2.model == s1.model  # nothing trained
        assert s2.model_version == s1.model_version

    def test_censored_decision_cannot_be_rewarded_later(self):
        s = fresh()
        record, s = decide(s, CANDS, T0, "pepper")
        _, s = censor(s, record.decision_id)
        resolution, s2 = learn(s, record.decision_id, 1.0)
        assert resolution is None
        assert s2 == s


class TestExpire:
    def test_nothing_due_is_a_no_op(self):
        s = fresh()
        record, s = decide(s, CANDS, T0, "pepper")
        resolutions, s2 = expire(s, record.t + DAY - 1.0)
        assert resolutions == ()
        assert s2 == s

    def test_due_exactly_at_the_horizon(self):
        s = fresh(default_reward=0.25)
        record, s1 = decide(s, CANDS, T0, "pepper")
        resolutions, s2 = expire(s1, record.t + DAY)
        assert resolutions == (Resolution(record.decision_id, EXPIRED, 0.25),)
        assert s2.ledger == ()
        assert s2.model_version == s1.model_version + 1
        # Trains with the declared default reward.
        assert s2.model == update(s1.model, record.features, 0.25)

    def test_expires_only_the_due_in_ledger_order(self):
        s = fresh()
        r1, s = decide(s, CANDS, T0, "pepper")
        r2, s = decide(s, CANDS, T0 + 10.0, "pepper")
        r3, s = decide(s, CANDS, T0 + 20.0, "pepper")
        resolutions, s2 = expire(s, T0 + DAY + 15.0)
        assert [r.decision_id for r in resolutions] == [r1.decision_id, r2.decision_id]
        assert s2.ledger == (r3,)

    def test_ledger_is_bounded_by_the_horizon(self):
        # Under a sweep at each event's time, the ledger never holds a
        # decision older than the horizon (design doc section 6).
        s = fresh()
        step = DAY / 4
        for i in range(20):
            t = T0 + i * step
            _, s = expire(s, t)
            _, s = decide(s, CANDS, t, "pepper")
            assert all(r.t + DAY > t for r in s.ledger)
            assert len(s.ledger) <= 4

    def test_reward_after_expiry_is_ignored(self):
        # The horizon is the declared cutoff: a reward arriving after its
        # decision expired finds nothing and changes nothing.
        s = fresh()
        record, s = decide(s, CANDS, T0, "pepper")
        _, s = expire(s, T0 + 2 * DAY)
        resolution, s2 = learn(s, record.decision_id, 1.0)
        assert resolution is None
        assert s2 == s


class TestFullLoop:
    def test_bandit_learns_through_the_ledger(self):
        # End-to-end: decide -> reward via learn, arm [1,0] pays 1 and
        # arm [0,1] pays 0. The engine starts uniform and ends greedy on
        # the paying arm, all through decision records — no hand-fed
        # (features, reward) pairs anywhere.
        s = fresh()
        chosen = []
        for i in range(300):
            t = T0 + float(i)
            record, s = decide(s, CANDS, t, "loop")
            _, s = learn(s, record.decision_id, 1.0 if record.chosen == 0 else 0.0)
            chosen.append(record.chosen)
        late = Counter(chosen[-100:])
        assert late[0] > 80


class TestPinnedVectors:
    # Golden vector: dim 2, default forgetting, horizon one day, default
    # reward 0.25. Decisions at T0, T0+600, T0+1200 (salt "pepper") over
    # CANDS; the second rewarded 1.0; expiry swept at T0+600+DAY. Chosen
    # arms are pinned along the way, so the trained feature sequence is
    # fully determined.
    def test_resolution_and_prediction_bits(self):
        s = new_bandit(2, horizon=DAY, default_reward=0.25)
        r1, s = decide(s, CANDS, T0, "pepper")
        r2, s = decide(s, CANDS, T0 + 600.0, "pepper")
        r3, s = decide(s, CANDS, T0 + 1200.0, "pepper")
        assert (r1.chosen, r2.chosen, r3.chosen) == (0, 0, 1)
        resolution, s = learn(s, r2.decision_id, 1.0)
        assert resolution == Resolution("pepper:1", REWARDED, 1.0)
        resolutions, s = expire(s, T0 + 600.0 + DAY)
        assert resolutions == (Resolution("pepper:0", EXPIRED, 0.25),)
        assert [r.decision_id for r in s.ledger] == ["pepper:2"]
        assert s.model_version == 2
        est, unc = predict(s.model, ((0, 1.0), (1, -1.0)))
        assert est == 0.41647215738579535
        assert unc == 1.1547486659415682
