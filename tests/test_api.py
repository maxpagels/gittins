import pytest

from gittins_reference import api
from gittins_reference.decide import decide as decide_encoded
from gittins_reference.decide import new_bandit
from gittins_reference.encoding import encode

CATALOG = [("basic", {"price": 3.0}), ("plus", {"price": 9.0, "trial": True}), ("free", {})]


class TestCreate:
    def test_declares_bits_not_dim(self):
        state = api.create(bits=5, horizon=10.0)
        assert state.model.dim == 32
        assert api.model_bits(state) == 5

    def test_bits_out_of_range(self):
        for bits in (0, 25, -1):
            with pytest.raises(ValueError, match="bits"):
                api.create(bits, horizon=10.0)

    def test_passes_through_expert_overrides(self):
        state = api.create(4, horizon=7.0, default_reward=-1.0, epsilon=0.2, forgetting=0.5)
        assert (state.horizon, state.default_reward, state.epsilon) == (7.0, -1.0, 0.2)
        assert state.model.forgetting == 0.5


class TestDecide:
    def test_is_exactly_the_layered_path(self):
        # The facade must add nothing: same record, same state, bit for bit,
        # as encoding by hand and deciding over the sparse pairs.
        context = {"seg": "a", "hour": 13}
        facade_state = api.create(4, horizon=10.0)
        layered_state = api.create(4, horizon=10.0)
        for t in (0.0, 1.0, 2.0):
            record, facade_state = api.decide(facade_state, context, CATALOG, t, "s")
            encoded = [encode(context, arm, action, 4) for arm, action in CATALOG]
            expected, layered_state = decide_encoded(layered_state, encoded, t, "s")
            assert record == expected
        assert facade_state == layered_state

    def test_rejects_states_not_built_by_create(self):
        state = new_bandit(5, horizon=10.0)  # dim 5: not a power of two
        with pytest.raises(ValueError, match="2\\*\\*bits"):
            api.decide(state, {}, CATALOG, 0.0, "s")

    def test_empty_candidates_rejected(self):
        with pytest.raises(ValueError, match="candidate"):
            api.decide(api.create(4, horizon=10.0), {}, [], 0.0, "s")


class TestFullCycle:
    def test_decide_learn_expire_serialize_through_the_facade(self):
        # The whole lifecycle without ever touching encoding or sparse
        # pairs: the surface a binding exposes is sufficient on its own.
        state = api.create(4, horizon=5.0)
        record, state = api.decide(state, {"seg": "a"}, CATALOG, 0.0, "s")
        resolution, state = api.learn(state, record.decision_id, 1.0)
        assert resolution.kind == "rewarded" and state.model_version == 1

        record, state = api.decide(state, {"seg": "b"}, CATALOG, 1.0, "s")
        expired, state = api.expire(state, record.t + 5.0)
        assert [r.decision_id for r in expired] == [record.decision_id]
        assert not state.ledger

        assert api.deserialize(api.serialize(state)) == state

    def test_learning_moves_later_decisions(self):
        # Same salt and context, deterministic replay: after strong rewards
        # for one arm, the greedy mass should move to it. Train through the
        # facade only, then check the trained model prefers the arm.
        state = api.create(6, horizon=10.0, forgetting=1.0)
        for i in range(30):
            record, state = api.decide(state, {}, CATALOG, float(i), "train")
            reward = 1.0 if CATALOG[record.chosen][0] == "plus" else 0.0
            _, state = api.learn(state, record.decision_id, reward)
        record, state = api.decide(state, {}, CATALOG, 1000.0, "probe")
        assert CATALOG[record.chosen][0] == "plus"
        assert record.propensity > 0.9  # the greedy share, not an epsilon draw
