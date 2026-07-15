import math

import pytest

from gittins_reference.decay import RENORM_LIMIT, vector_read
from gittins_reference.decide import new_bandit, decide
from gittins_reference.ledger import learn
from gittins_reference.merge import merge_states
from gittins_reference.model import LinearModel, predict, update

HOUR = 3600.0
DAY = 24 * HOUR
T0 = 1_752_000_000.0

# Dense candidates on purpose: one-hot features would land agents' evidence
# in disjoint sum entries and hide all rounding interplay.
CANDS = [[1.0, 0.5], [0.5, -1.0]]


def run_agent(salt, half_life, times, rewards):
    """One agent: decide+learn at each time, rewards fed positionally.
    Returns (state, resolved (t, features, reward) events)."""
    s = new_bandit(2, half_life, T0, horizon=1e12)
    events = []
    for t, r in zip(times, rewards, strict=True):
        record, s = decide(s, CANDS, t, salt)
        _, s = learn(s, record.decision_id, r)
        events.append((t, record.features, r))
    return s, events


def max_relerr(a: LinearModel, b: LinearModel, t: float) -> float:
    worst = 0.0
    for va, vb in [(a.xx, b.xx), (a.xy, b.xy)]:
        for x, y in zip(vector_read(va, t), vector_read(vb, t), strict=True):
            worst = max(worst, abs(x - y) / max(abs(x), abs(y), 1e-12))
    return worst


class TestMergeStates:
    def test_commutative_bit_for_bit(self):
        a, _ = run_agent("agent-a", HOUR, [T0, T0 + 60], [1.0, -0.5])
        b, _ = run_agent("agent-b", HOUR, [T0 + 30], [0.25])
        assert merge_states(a, b) == merge_states(b, a)

    def test_merged_state_is_pooled_knowledge_only(self):
        # Knowledge merges (sums, version); operational identity does not:
        # open decisions stay with their agent, and a merged state that will
        # decide needs a fresh salt, so its sequence starts at 0.
        a, _ = run_agent("agent-a", HOUR, [T0], [1.0])
        b, _ = run_agent("agent-b", HOUR, [T0 + 60], [0.0])
        _, a = decide(a, CANDS, T0 + 120, "agent-a")  # left open
        merged = merge_states(a, b)
        assert merged.model_version == a.model_version + b.model_version
        assert merged.ledger == ()
        assert merged.next_seq == 0
        assert merged.horizon == a.horizon
        assert merged.default_reward == a.default_reward

    def test_merging_a_fresh_state_changes_no_evidence(self):
        a, _ = run_agent("agent-a", HOUR, [T0, T0 + 60], [1.0, -0.5])
        fresh = new_bandit(2, HOUR, T0, horizon=1e12)
        merged = merge_states(a, fresh)
        assert merged.model.xx.values == a.model.xx.values
        assert merged.model.xy.values == a.model.xy.values

    def test_rejects_mismatched_applications(self):
        base = new_bandit(2, HOUR, T0, horizon=DAY)
        with pytest.raises(ValueError):
            merge_states(base, new_bandit(2, HOUR, T0, horizon=2 * DAY))
        with pytest.raises(ValueError):
            merge_states(base, new_bandit(2, HOUR, T0, horizon=DAY, default_reward=1.0))
        with pytest.raises(ValueError):
            merge_states(base, new_bandit(3, HOUR, T0, horizon=DAY))  # dim
        with pytest.raises(ValueError):
            merge_states(base, new_bandit(2, 2 * HOUR, T0, horizon=DAY))  # half-life


class TestLateRewardMergeCommutation:
    # Design doc section 13 risk 3: applying a late reward at its
    # origin-time weight must commute with timestamp-aligned merging, or
    # fleet mode silently miscounts. Property: resolve-then-merge equals
    # merge-then-(same update on the pooled model), across half-lives
    # (including "never forget") and across a renormalization era. The
    # guarantee is rounding-level (float addition is commutative but not
    # associative); in practice it is usually exact.
    @pytest.mark.parametrize("half_life", [HOUR, math.inf])
    @pytest.mark.parametrize("gap_hours", [0.0, 9.0, RENORM_LIMIT + 9.0])
    def test_resolve_then_merge_equals_merge_then_resolve(self, half_life, gap_hours):
        gap = gap_hours * HOUR
        a, _ = run_agent(
            "agent-a", half_life, [T0 + i * 30.0 for i in range(4)], [0.7 * i - 1.0 for i in range(4)]
        )
        record, a = decide(a, CANDS, T0 + 200.0, "agent-a")  # left open
        b, _ = run_agent(
            "agent-b", half_life, [T0 + gap + i * 60.0 for i in range(3)], [-0.5, 0.5, 1.5]
        )
        _, a_resolved = learn(a, record.decision_id, 1.0)
        resolve_then_merge = merge_states(a_resolved, b).model
        merge_then_resolve = update(
            merge_states(a, b).model, list(record.features), 1.0, record.t
        )
        assert max_relerr(resolve_then_merge, merge_then_resolve, T0 + gap + 100.0) < 1e-12


class TestMergeEqualsCentral:
    def test_fleet_of_three_matches_one_bandit_that_saw_everything(self):
        # Three agents on interleaved schedules; a central model receives
        # the same resolved events in global time order. Pooling by merge
        # must match the central bandit to rounding.
        agents = []
        events = []
        for k, salt in enumerate(["agent-a", "agent-b", "agent-c"]):
            times = [T0 + i * 60.0 + k * 20.0 for i in range(30)]
            s = new_bandit(2, HOUR, T0, horizon=1e12)
            for t in times:
                record, s = decide(s, CANDS, t, salt)
                reward = 1.0 if record.chosen == 0 else 0.0
                _, s = learn(s, record.decision_id, reward)
                events.append((t, record.features, reward))
            agents.append(s)
        merged = merge_states(merge_states(agents[0], agents[1]), agents[2])
        central = new_bandit(2, HOUR, T0, horizon=1e12).model
        for t, features, reward in sorted(events):
            central = update(central, list(features), reward, t)
        assert merged.model_version == 90
        t_query = T0 + 3000.0
        assert max_relerr(merged.model, central, t_query) < 1e-12
        for probe in [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]:
            est_m, unc_m = predict(merged.model, probe, t_query)
            est_c, unc_c = predict(central, probe, t_query)
            assert math.isclose(est_m, est_c, rel_tol=1e-12, abs_tol=1e-12)
            assert math.isclose(unc_m, unc_c, rel_tol=1e-12)

    def test_merge_is_associative_to_rounding(self):
        a, _ = run_agent("agent-a", HOUR, [T0 + i * 60.0 for i in range(5)], [1.0] * 5)
        b, _ = run_agent("agent-b", HOUR, [T0 + 20 + i * 60.0 for i in range(5)], [-0.5] * 5)
        c, _ = run_agent("agent-c", HOUR, [T0 + 40 + i * 60.0 for i in range(5)], [0.25] * 5)
        left = merge_states(merge_states(a, b), c).model
        right = merge_states(a, merge_states(b, c)).model
        assert max_relerr(left, right, T0 + 400.0) < 1e-12


class TestPinnedVectors:
    # Golden vector: two one-decision agents (chosen arms pinned), merged;
    # bit-exact pooled prediction.
    def test_merged_prediction_bits(self):
        a = new_bandit(2, HOUR, T0, horizon=DAY)
        b = new_bandit(2, HOUR, T0, horizon=DAY)
        ra, a = decide(a, CANDS, T0, "agent-a")
        _, a = learn(a, ra.decision_id, 1.0)
        rb, b = decide(b, CANDS, T0 + 1800.0, "agent-b")
        _, b = learn(b, rb.decision_id, -0.5)
        assert (ra.chosen, rb.chosen) == (0, 0)
        merged = merge_states(a, b)
        assert merged.model_version == 2
        est, unc = predict(merged.model, [1.0, -1.0], T0 + 3600.0)
        assert est == 0.02918561399511608
        assert unc == 1.3710276201822664
