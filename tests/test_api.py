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
        # (The public record additionally carries the inputs; the compact
        # fields are the layered record's, exactly.)
        context = {"seg": "a", "hour": 13}
        facade = api.create(4, horizon=10.0)
        layered_state = new_bandit(1 << 4, horizon=10.0)
        for t in (0.0, 1.0, 2.0):
            record = api.decide(facade, context, CATALOG, t, "s")
            encoded = [encode(context, arm, action, 4) for arm, action in CATALOG]
            expected, layered_state = decide_encoded(layered_state, encoded, t, "s")
            compact = DecisionRecord(**{
                f: getattr(record, f)
                for f in ("decision_id", "t", "candidate_hash", "chosen",
                          "features", "propensity", "model_version", "salt")
            })
            assert compact == expected
            assert record.bits == 4
            assert record.context is context and record.candidates is CATALOG
        assert facade._state == layered_state

    def test_handle_updates_in_place(self):
        # The uniform cross-binding convention: results come back, the
        # handle mutates, and every alias observes the current state.
        state = api.create(4, horizon=10.0)
        alias = state
        record = api.decide(state, {}, CATALOG, 0.0, "s")
        assert isinstance(record, api.DecisionRecord)
        assert api.learn(alias, record.decision_id, 1.0, 1.0).kind == "rewarded"
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
        resolution = api.learn(state, record.decision_id, 1.0, 1.0)
        assert resolution.kind == "rewarded"
        assert api.learn(state, record.decision_id, 1.0, 1.0) is None  # exactly once

        record = api.decide(state, {"seg": "b"}, CATALOG, 1.0, "s")
        expired = api.expire(state, record.t + 5.0)
        assert [r.decision_id for r in expired] == [record.decision_id]
        assert api.expire(state, record.t + 5.0) == ()

        data = api.serialize(state)
        assert isinstance(data, str) and data.startswith("67697474696e7300")  # "gittins\0"
        assert api.serialize(api.deserialize(data)) == data
        assert api.serialize(api.deserialize(data.upper())) == data  # either case in, lowercase out

    def test_late_reward_learns_as_expired(self):
        # The horizon is enforced at learn time: a reward arriving at or
        # past record.t + horizon trains as default_reward, exactly as an
        # expire sweep would have resolved it.
        state = api.create(4, horizon=5.0, default_reward=-1.0)
        record = api.decide(state, {}, CATALOG, 0.0, "s")
        resolution = api.learn(state, record.decision_id, 1.0, 5.0)
        assert resolution.kind == "expired" and resolution.reward == -1.0
        assert api.learn(state, record.decision_id, 1.0, 5.0) is None  # still exactly once

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
            api.learn(state, record.decision_id, reward, float(i))
        record = api.decide(state, {}, CATALOG, 1000.0, "probe")
        assert CATALOG[record.chosen][0] == "plus"
        assert record.propensity > 0.9  # the greedy share, not an epsilon draw


class TestByo:
    # The public BYO surface: score/explore on decide, train
    # on learn/expire. A BYO model is score + train; a BYO exploration
    # rule is explore; the engine keeps encoding, the RNG draw, the
    # record, propensity logging, and the exactly-once ledger.

    def test_score_receives_the_callers_objects(self):
        state = api.create(4, horizon=10.0)
        context = {"seg": "a"}
        seen = []

        def score(ctx, cands):
            seen.append((ctx, cands))
            return [1.0, 0.0, 0.0]

        record = api.decide(state, context, CATALOG, 0.0, "s", score=score)
        assert seen == [(context, CATALOG)]
        assert seen[0][0] is context and seen[0][1] is CATALOG
        assert record.chosen == 0

    def test_explore_propensity_is_logged(self):
        state = api.create(4, horizon=10.0)
        p = [0.25, 0.25, 0.5]
        record = api.decide(state, {}, CATALOG, 0.0, "s", explore=lambda est, eps: p)
        assert record.propensity == p[record.chosen]

    def test_train_replaces_the_builtin_update(self):
        state = api.create(4, horizon=10.0)
        record = api.decide(state, {"seg": "a"}, CATALOG, 0.0, "s")
        model_before = state._state.model
        trained = []
        resolution = api.learn(
            state, record.decision_id, 0.75, 1.0, train=lambda rec, r: trained.append((rec, r))
        )
        assert resolution.kind == "rewarded" and resolution.reward == 0.75
        assert state._state.model is model_before  # built-in model untouched
        assert state._state.model_version == 1  # the observation still counts
        [(rec, r)] = trained
        assert (rec.decision_id, rec.features, r) == (record.decision_id, record.features, 0.75)
        # The train-side record is the compact one: the inputs live in the
        # log written at decide time, never in the state.
        assert rec.bits is None and rec.context is None and rec.candidates is None
        # Exactly once: the retry is a no-op and the callback stays quiet.
        assert api.learn(state, record.decision_id, 0.75, 1.0, train=lambda rec, r: trained.append((rec, r))) is None
        assert len(trained) == 1

    def test_late_learn_with_train_fires_with_the_default_reward(self):
        # The classification happens before the training tap: a late
        # reward reaches the BYO model as default_reward, kind expired.
        state = api.create(4, horizon=5.0, default_reward=-1.0)
        record = api.decide(state, {}, CATALOG, 0.0, "s")
        trained = []
        resolution = api.learn(
            state, record.decision_id, 1.0, 5.0, train=lambda rec, r: trained.append(r)
        )
        assert resolution.kind == "expired" and resolution.reward == -1.0
        assert trained == [-1.0]

    def test_expire_with_train(self):
        state = api.create(4, horizon=5.0, default_reward=-1.0)
        records = [api.decide(state, {}, CATALOG, float(t), "s") for t in range(2)]
        model_before = state._state.model
        trained = []
        resolutions = api.expire(state, 100.0, train=lambda rec, r: trained.append((rec, r)))
        assert [r.decision_id for r in resolutions] == [r.decision_id for r in records]
        assert all(r.kind == "expired" and r.reward == -1.0 for r in resolutions)
        assert [(rec.decision_id, r) for rec, r in trained] == [
            (records[0].decision_id, -1.0),
            (records[1].decision_id, -1.0),
        ]
        assert state._state.model is model_before
        assert state._state.model_version == 2

    def test_train_fires_after_the_commit(self):
        # A raising callback loses its one example loudly but can never
        # double-train: the record is spent before the callback runs.
        state = api.create(4, horizon=10.0)
        record = api.decide(state, {}, CATALOG, 0.0, "s")

        def boom(rec, r):
            raise RuntimeError("trainer crashed")

        with pytest.raises(RuntimeError, match="trainer crashed"):
            api.learn(state, record.decision_id, 1.0, 1.0, train=boom)
        assert state._state.model_version == 1
        assert api.learn(state, record.decision_id, 1.0, 1.0) is None  # already spent

    def test_expire_sweep_is_per_record_under_a_raising_train(self):
        state = api.create(4, horizon=5.0)
        records = [api.decide(state, {}, CATALOG, float(t), "s") for t in range(3)]
        calls = []

        def boom(rec, r):
            calls.append(rec.decision_id)
            if len(calls) == 2:
                raise RuntimeError("flaky trainer")

        with pytest.raises(RuntimeError, match="flaky trainer"):
            api.expire(state, 100.0, train=boom)
        # Records 0 and 1 committed (1's example was lost loudly); 2 is
        # still open for the next sweep.
        assert calls == [records[0].decision_id, records[1].decision_id]
        remaining = api.expire(state, 100.0, train=lambda rec, r: calls.append(rec.decision_id))
        assert [r.decision_id for r in remaining] == [records[2].decision_id]

    def test_validation_messages_reach_the_public_surface(self):
        state = api.create(4, horizon=10.0)
        with pytest.raises(ValueError, match="score must return one finite estimate"):
            api.decide(state, {}, CATALOG, 0.0, "s", score=lambda c, k: ["high"] * len(k))
        with pytest.raises(ValueError, match="finite, nonnegative, and sum to 1"):
            api.decide(state, {}, CATALOG, 0.0, "s", explore=lambda e, eps: [0.9] * len(e))

    def test_byo_state_round_trips_like_any_other(self):
        # Callbacks are per-call values: nothing about them is stored, so
        # a BYO agent's state serializes and restores exactly as usual.
        state = api.create(4, horizon=10.0)
        record = api.decide(
            state, {}, CATALOG, 0.0, "s",
            score=lambda c, k: [1.0, 0.0, 0.0],
            explore=lambda e, eps: [0.5, 0.25, 0.25],
        )
        api.learn(state, record.decision_id, 1.0, 1.0, train=lambda rec, r: None)
        data = api.serialize(state)
        assert api.serialize(api.deserialize(data)) == data


class TestLogLine:
    # Assembly-free logging: the record decide returns
    # carries its inputs, and log_line emits exactly the experience-log
    # line verify/evaluate/replay consume.

    def test_appended_lines_are_a_valid_verified_log(self):
        from gittins_reference import ope

        state = api.create(4, horizon=10.0)
        lines = []
        record = api.decide(state, {"seg": "a"}, CATALOG, 0.0, "s")
        lines.append(api.log_line(record))
        lines.append(api.log_line(api.learn(state, record.decision_id, 1.0, 0.5)))
        record = api.decide(state, {"seg": "b"}, CATALOG, 1.0, "s")
        lines.append(api.log_line(record))
        lines.extend(api.log_line(r) for r in api.expire(state, 100.0))
        events = tuple(ope.parse_log(lines))
        assert ope.verify(events) == ()
        report = ope.evaluate(events, bits=4)
        assert report.decisions == 2 and report.resolved == 2

    def test_lines_are_compact_single_line_json(self):
        import json

        state = api.create(4, horizon=10.0)
        record = api.decide(state, {"seg": "a"}, CATALOG, 0.0, "s")
        line = api.log_line(record)
        assert "\n" not in line and ": " not in line and ", " not in line
        parsed = json.loads(line)
        assert list(parsed) == ["event", "bits", "context", "candidates", "record"]
        assert parsed["record"]["decision_id"] == record.decision_id
        # Feature dicts keep insertion order — the order they encoded with.
        assert list(parsed["context"]) == ["seg"]

    def test_refuses_records_without_inputs(self):
        # A train callback's record is the compact one; its inputs are
        # already in the log, written at decide time.
        state = api.create(4, horizon=10.0)
        record = api.decide(state, {}, CATALOG, 0.0, "s")
        seen = []
        api.learn(state, record.decision_id, 1.0, 1.0, train=lambda rec, r: seen.append(rec))
        with pytest.raises(ValueError, match="record decide returned"):
            api.log_line(seen[0])
        with pytest.raises(ValueError, match="DecisionRecord or Resolution"):
            api.log_line({"not": "a record"})
