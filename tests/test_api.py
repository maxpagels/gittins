import pytest

from gittins_reference import api
from gittins_reference import state as state_module
from gittins_reference.decide import DecisionRecord
from gittins_reference.decide import decide as decide_encoded
from gittins_reference.decide import new_bandit
from gittins_reference.encoding import encode

CATALOG = [("basic", {"price": 3.0}), ("plus", {"price": 9.0, "trial": True}), ("free", {})]


class TestCreate:
    def test_declares_bits_not_dim(self):
        state = api.create(bits=5, horizon=10.0)
        assert api.model_bits(state) == 5

    def test_bits_out_of_range(self):
        for bits in (0, 25, -1):
            with pytest.raises(ValueError, match="bits"):
                api.create(bits, horizon=10.0)

    def test_passes_through_expert_overrides(self):
        state = api.create(4, horizon=7.0, default_reward=-1.0, epsilon=0.2, forgetfulness=0.5)
        inner = state._state
        assert (inner.horizon, inner.default_reward, inner.epsilon) == (7.0, -1.0, 0.2)
        assert inner.model.forgetting == 0.5


class TestDecide:
    def test_is_exactly_the_layered_path(self):
        # The facade must add nothing: same record, same state, bit for bit,
        # as encoding by hand and deciding over the sparse pairs — the
        # handle updating in place while the layered path stays functional.
        context = {"seg": "a", "hour": 13}
        facade = api.create(4, horizon=10.0)
        layered_state = new_bandit(1 << 4, horizon=10.0)
        for t in (0.0, 1.0, 2.0):
            record = api.decide(facade, context, CATALOG, t, "s")
            encoded = [encode(context, arm, action, 4) for arm, action in CATALOG]
            expected, layered_state = decide_encoded(layered_state, encoded, t, "s")
            assert record == expected
        assert facade._state == layered_state

    def test_handle_updates_in_place(self):
        # The uniform cross-binding convention: results come back, the
        # handle mutates, and every alias observes the current state.
        state = api.create(4, horizon=10.0)
        alias = state
        record = api.decide(state, {}, CATALOG, 0.0, "s")
        assert isinstance(record, DecisionRecord)
        assert api.learn(alias, record.decision_id, 1.0).kind == "rewarded"
        assert api.serialize(alias) == api.serialize(state)

    def test_rejects_states_not_built_by_create(self):
        # dim 5 is not a power of two; reaching the facade with one is only
        # possible via the layered API, and decide refuses it.
        state = api.deserialize(state_module.serialize(new_bandit(5, horizon=10.0)).hex())
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
        record = api.decide(state, {"seg": "a"}, CATALOG, 0.0, "s")
        resolution = api.learn(state, record.decision_id, 1.0)
        assert resolution.kind == "rewarded"
        assert api.learn(state, record.decision_id, 1.0) is None  # exactly once

        record = api.decide(state, {"seg": "b"}, CATALOG, 1.0, "s")
        expired = api.expire(state, record.t + 5.0)
        assert [r.decision_id for r in expired] == [record.decision_id]
        assert api.expire(state, record.t + 5.0) == ()

        data = api.serialize(state)
        assert isinstance(data, str) and data.startswith("67697474696e7300")  # "gittins\0"
        assert api.serialize(api.deserialize(data)) == data
        assert api.serialize(api.deserialize(data.upper())) == data  # either case in, lowercase out

    def test_deserialize_rejects_non_hex(self):
        good = api.serialize(api.create(4, horizon=5.0))
        for bad in (good[:-1], good + "zz", good.replace(good[:2], "6 "), "", "banana"):
            with pytest.raises(ValueError):
                api.deserialize(bad)

    def test_learning_moves_later_decisions(self):
        # Same salt and context, deterministic replay: after strong rewards
        # for one arm, the greedy mass should move to it. Train through the
        # facade only, then check the trained model prefers the arm.
        state = api.create(6, horizon=10.0, forgetfulness=1.0)
        for i in range(30):
            record = api.decide(state, {}, CATALOG, float(i), "train")
            reward = 1.0 if CATALOG[record.chosen][0] == "plus" else 0.0
            api.learn(state, record.decision_id, reward)
        record = api.decide(state, {}, CATALOG, 1000.0, "probe")
        assert CATALOG[record.chosen][0] == "plus"
        assert record.propensity > 0.9  # the greedy share, not an epsilon draw
