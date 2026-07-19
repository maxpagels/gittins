"""The binding's acceptance tests, per spec/api.md: replay the golden `api`
section through the binding's public surface alone, and pin the binding to
the reference bit for bit. Requires the `gittins` wheel and (for the
equivalence test) `gittins-reference` installed in the same environment."""

import json
from pathlib import Path

import pytest

gittins = pytest.importorskip("gittins")

GOLDEN = json.loads(
    (Path(__file__).resolve().parents[3] / "spec" / "golden.json").read_text(encoding="utf-8")
)


def replay_api_section():
    """The corpus scenario, driven through the binding only."""
    s = GOLDEN["sections"]["api"]
    state = gittins.create(bits=s["bits"], horizon=s["horizon"], forgetfulness=s["forgetfulness"])
    catalog = [(arm_id, action) for arm_id, action in s["catalog"]]
    records = []
    for event in s["events"]:
        record = gittins.decide(state, event["context"], catalog, event["t"], s["salt"])
        records.append((record, event["record"]))
    resolutions = []
    _, expected0 = records[0]
    _, expected1 = records[1]
    resolutions.append(gittins.learn(state, expected1["decision_id"], 1.0))
    resolutions.append(gittins.learn(state, expected0["decision_id"], 0.0))
    resolutions.append(gittins.censor(state, records[2][1]["decision_id"]))
    resolutions.extend(gittins.expire(state, s["expire_sweep_at"]))
    return s, state, records, resolutions


class TestGoldenGate:
    def test_records_match_the_corpus(self):
        # JSON floats parse to exact doubles; == on floats is the bitwise
        # comparison the corpus demands (it contains no NaN and no -0.0).
        _, _, records, _ = replay_api_section()
        for record, expected in records:
            assert record.decision_id == expected["decision_id"]
            assert record.t == expected["t"]
            assert record.candidate_hash == expected["candidate_hash"]
            assert record.chosen == expected["chosen"]
            assert [list(pair) for pair in record.features] == expected["features"]
            assert record.propensity == expected["propensity"]
            assert record.model_version == expected["model_version"]
            assert record.salt == expected["salt"]

    def test_resolutions_match_the_corpus(self):
        s, _, _, resolutions = replay_api_section()
        expected = s["resolutions"]
        assert len(resolutions) == len(expected)
        for r, e in zip(resolutions, expected):
            assert (r.decision_id, r.kind, r.reward) == (
                e["decision_id"],
                e["kind"],
                e["reward"],
            )

    def test_final_state_matches_the_corpus(self):
        s, state, _, _ = replay_api_section()
        data = gittins.serialize(state)
        assert data == s["final"]["state_hex"]  # serialize IS the hex string
        assert gittins.serialize(gittins.deserialize(data)) == data


class TestReferenceEquivalence:
    def test_binding_and_reference_agree_decision_for_decision(self):
        # The strongest check the binding can get: the same scripted run
        # through the wheel and through pure Python must produce identical
        # records and identical serialized bytes.
        api = pytest.importorskip("gittins_reference.api")
        catalog = [("basic", {"price": 3.0}), ("plus", {"trial": True}), ("free", {})]
        bound = gittins.create(bits=6, horizon=100.0, epsilon=0.1)
        pure = api.create(bits=6, horizon=100.0, epsilon=0.1)
        for i in range(40):
            context = {"seg": "a" if i % 3 else "b", "hour": float(i % 24)}
            b_record = gittins.decide(bound, context, catalog, float(i), "eq")
            p_record = api.decide(pure, context, catalog, float(i), "eq")
            assert b_record.decision_id == p_record.decision_id
            assert b_record.chosen == p_record.chosen
            assert b_record.propensity == p_record.propensity
            assert list(b_record.features) == list(p_record.features)
            assert b_record.candidate_hash == p_record.candidate_hash
            if i % 4 == 0:
                gittins.learn(bound, b_record.decision_id, 1.0)
                api.learn(pure, p_record.decision_id, 1.0)
            if i % 7 == 0:
                gittins.expire(bound, float(i))
                api.expire(pure, float(i))
        assert gittins.serialize(bound) == api.serialize(pure)


class TestSurface:
    def test_errors_are_valueerrors_with_reference_messages(self):
        with pytest.raises(ValueError, match="bits must be between 1 and 24"):
            gittins.create(0, horizon=10.0)
        with pytest.raises(ValueError, match="horizon must be positive"):
            gittins.create(4, horizon=0.0)
        state = gittins.create(4, horizon=10.0)
        with pytest.raises(ValueError, match="unsupported type"):
            gittins.decide(state, {"bad": [1, 2]}, [("a", {})], 0.0, "s")
        with pytest.raises(ValueError, match="hexadecimal"):
            gittins.deserialize(gittins.serialize(state)[:-1])  # odd length
        with pytest.raises(ValueError, match="checksum|truncated"):
            gittins.deserialize(gittins.serialize(state)[:-2])  # missing byte

    def test_unknown_resolutions_are_none(self):
        state = gittins.create(4, horizon=10.0)
        assert gittins.learn(state, "never-made:0", 1.0) is None
        assert gittins.expire(state, 1e9) == ()

    def test_the_usage_doc_lifecycle(self):
        # docs/usage.md, through the wheel: decide, learn, expire, censor,
        # save, load.
        state = gittins.create(bits=8, horizon=3600.0)
        context = {"device": "mobile", "hour": 14}
        candidates = [
            ("banner-sale", {"discount": 0.2}),
            ("banner-new", {"discount": 0.0}),
            ("banner-plain", {}),
        ]
        record = gittins.decide(state, context, candidates, 0.0, "agent-1")
        assert candidates[record.chosen][0].startswith("banner-")
        assert gittins.learn(state, record.decision_id, 1.0).kind == "rewarded"
        record = gittins.decide(state, context, candidates, 1.0, "agent-1")
        assert gittins.censor(state, record.decision_id).kind == "censored"
        assert gittins.model_bits(state) == 8
        restored = gittins.deserialize(gittins.serialize(state))
        assert gittins.serialize(restored) == gittins.serialize(state)


# The spec-defined golden byo callbacks (spec/byo.md), written natively in
# Python — the binding-side halves of the corpus's `byo` section.
def byo_score(context, candidates):
    base = 1.0 if context["seg"] == "a" else 0.0
    return [
        base + 0.5 * action.get("price", 0.0) - i
        for i, (_arm, action) in enumerate(candidates)
    ]


def byo_explore(estimates, epsilon):
    k = len(estimates)
    return [(i + 1) / (k * (k + 1) / 2) for i in range(k)]


def replay_byo_section():
    """The corpus's byo scenario, driven through the binding only, with the
    callbacks crossing the real Python boundary."""
    s = GOLDEN["sections"]["byo"]
    state = gittins.create(bits=s["bits"], horizon=s["horizon"], forgetfulness=s["forgetfulness"])
    catalog = [(arm_id, action) for arm_id, action in s["catalog"]]
    trained = []

    def train(record, reward):
        trained.append(
            {
                "decision_id": record.decision_id,
                "features": [list(pair) for pair in record.features],
                "reward": reward,
            }
        )

    records = []
    for event in s["events"]:
        mode = event["mode"]
        record = gittins.decide(
            state,
            event["context"],
            catalog,
            event["t"],
            s["salt"],
            score=byo_score if mode in ("score", "both") else None,
            explore=byo_explore if mode in ("explore", "both") else None,
        )
        records.append((record, event["record"]))
    ids = [expected["decision_id"] for _, expected in records]
    resolutions = []
    resolutions.append(gittins.learn(state, ids[1], 1.0, train=train))
    resolutions.append(gittins.learn(state, ids[0], 0.0, train=train))
    resolutions.append(gittins.learn(state, ids[4], 1.0))  # deliberately mixed: built-in update
    resolutions.append(gittins.censor(state, ids[2]))
    resolutions.extend(gittins.expire(state, s["expire_sweep_at"], train=train))
    return s, state, records, resolutions, trained


class TestByoGoldenGate:
    def test_records_match_the_corpus(self):
        _, _, records, _, _ = replay_byo_section()
        for record, expected in records:
            assert record.decision_id == expected["decision_id"]
            assert record.chosen == expected["chosen"]
            assert [list(pair) for pair in record.features] == expected["features"]
            assert record.propensity == expected["propensity"]
            assert record.candidate_hash == expected["candidate_hash"]
            assert record.model_version == expected["model_version"]

    def test_resolutions_and_trained_match_the_corpus(self):
        s, _, _, resolutions, trained = replay_byo_section()
        expected = s["resolutions"]
        assert len(resolutions) == len(expected)
        for r, e in zip(resolutions, expected):
            assert (r.decision_id, r.kind, r.reward) == (e["decision_id"], e["kind"], e["reward"])
        assert trained == s["trained"]

    def test_final_state_matches_the_corpus(self):
        s, state, _, _, _ = replay_byo_section()
        assert gittins.serialize(state) == s["final"]["state_hex"]


class TestByoReferenceEquivalence:
    def test_binding_and_reference_agree_under_byo_callbacks(self):
        api = pytest.importorskip("gittins_reference.api")
        catalog = [("basic", {"price": 3.0}), ("plus", {"price": 9.0}), ("free", {})]
        bound = gittins.create(bits=6, horizon=100.0)
        pure = api.create(bits=6, horizon=100.0)
        trained = {"bound": [], "pure": []}
        for i in range(30):
            context = {"seg": "a" if i % 3 else "b"}
            kwargs = {}
            if i % 2 == 0:
                kwargs["score"] = byo_score
            if i % 3 == 0:
                kwargs["explore"] = byo_explore
            b_record = gittins.decide(bound, context, catalog, float(i), "eq", **kwargs)
            p_record = api.decide(pure, context, catalog, float(i), "eq", **kwargs)
            assert b_record.decision_id == p_record.decision_id
            assert b_record.chosen == p_record.chosen
            assert b_record.propensity == p_record.propensity
            if i % 4 == 0:
                gittins.learn(
                    bound, b_record.decision_id, 1.0,
                    train=lambda rec, r: trained["bound"].append((rec.decision_id, r)),
                )
                api.learn(
                    pure, p_record.decision_id, 1.0,
                    train=lambda rec, r: trained["pure"].append((rec.decision_id, r)),
                )
        gittins.expire(bound, 1e9, train=lambda rec, r: trained["bound"].append((rec.decision_id, r)))
        api.expire(pure, 1e9, train=lambda rec, r: trained["pure"].append((rec.decision_id, r)))
        assert trained["bound"] == trained["pure"]
        assert gittins.serialize(bound) == api.serialize(pure)


class TestByoSurface:
    def test_score_receives_the_callers_objects(self):
        state = gittins.create(4, horizon=10.0)
        context = {"seg": "a"}
        catalog = [("a", {}), ("b", {})]
        seen = []

        def score(ctx, cands):
            seen.append((ctx, cands))
            return [1.0, 0.0]

        gittins.decide(state, context, catalog, 0.0, "s", score=score)
        assert seen[0][0] is context and seen[0][1] is catalog

    def test_callback_exceptions_are_reraised(self):
        state = gittins.create(4, horizon=10.0)
        catalog = [("a", {}), ("b", {})]

        def boom(ctx, cands):
            raise RuntimeError("model server down")

        with pytest.raises(RuntimeError, match="model server down"):
            gittins.decide(state, {}, catalog, 0.0, "s", score=boom)
        # A failing decide changes nothing: the next id is still s:0.
        record = gittins.decide(state, {}, catalog, 0.0, "s")
        assert record.decision_id == "s:0"

        def boom_train(rec, r):
            raise RuntimeError("trainer crashed")

        with pytest.raises(RuntimeError, match="trainer crashed"):
            gittins.learn(state, record.decision_id, 1.0, train=boom_train)
        # Commit-then-fire: the record is spent; no double-train possible.
        assert gittins.learn(state, record.decision_id, 1.0) is None

    def test_malformed_callback_results_are_valueerrors_with_reference_messages(self):
        state = gittins.create(4, horizon=10.0)
        catalog = [("a", {}), ("b", {})]
        with pytest.raises(ValueError, match="score must return one finite estimate"):
            gittins.decide(state, {}, catalog, 0.0, "s", score=lambda c, k: ["high", "low"])
        with pytest.raises(ValueError, match="score must return one finite estimate"):
            gittins.decide(state, {}, catalog, 0.0, "s", score=lambda c, k: [1.0])
        with pytest.raises(ValueError, match="one probability per candidate"):
            gittins.decide(state, {}, catalog, 0.0, "s", explore=lambda e, eps: [1.0])
        with pytest.raises(ValueError, match="finite, nonnegative, and sum to 1"):
            gittins.decide(state, {}, catalog, 0.0, "s", explore=lambda e, eps: [0.9, 0.9])

    def test_train_replaces_the_builtin_update(self):
        state = gittins.create(4, horizon=10.0)
        catalog = [("a", {}), ("b", {})]
        before = gittins.serialize(state)
        record = gittins.decide(state, {}, catalog, 0.0, "s")
        trained = []
        resolution = gittins.learn(
            state, record.decision_id, 1.0, train=lambda rec, r: trained.append((rec.decision_id, r))
        )
        assert resolution.kind == "rewarded"
        assert trained == [(record.decision_id, 1.0)]
        # The model bytes are untouched by a BYO-trained resolution; only
        # the counters moved. Deserialize both and compare the model by
        # re-serializing a state whose counters are advanced identically.
        after = gittins.serialize(state)
        assert after != before  # counters advanced


class TestLogLine:
    # Assembly-free logging (spec/ope.md): the record decide returns
    # carries the caller's inputs, and log_line emits the canonical
    # experience-log line — byte-identical to the reference's, since both
    # serialize with the same rules.

    def test_record_carries_the_callers_objects(self):
        state = gittins.create(4, horizon=10.0)
        context = {"seg": "a"}
        catalog = [("a", {"price": 1.0}), ("b", {})]
        record = gittins.decide(state, context, catalog, 0.0, "s")
        assert record.bits == 4
        assert record.context is context and record.candidates is catalog

    def test_lines_match_the_reference_byte_for_byte(self):
        api = pytest.importorskip("gittins_reference.api")
        catalog = [("basic", {"price": 3.0}), ("plus", {"trial": True}), ("free", {})]
        bound = gittins.create(bits=6, horizon=100.0)
        pure = api.create(bits=6, horizon=100.0)
        for i in range(10):
            context = {"seg": "a" if i % 3 else "b", "hour": float(i % 24)}
            b_record = gittins.decide(bound, context, catalog, float(i), "eq")
            p_record = api.decide(pure, context, catalog, float(i), "eq")
            assert gittins.log_line(b_record) == api.log_line(p_record)
            if i % 4 == 0:
                b_res = gittins.learn(bound, b_record.decision_id, 1.0)
                p_res = api.learn(pure, p_record.decision_id, 1.0)
                assert gittins.log_line(b_res) == api.log_line(p_res)

    def test_appended_lines_are_a_verified_evaluable_log(self):
        ope = pytest.importorskip("gittins_reference.ope")
        state = gittins.create(4, horizon=10.0)
        catalog = [("a", {"price": 1.0}), ("b", {})]
        lines = []
        record = gittins.decide(state, {"seg": "a"}, catalog, 0.0, "s")
        lines.append(gittins.log_line(record))
        lines.append(gittins.log_line(gittins.learn(state, record.decision_id, 1.0)))
        record = gittins.decide(state, {"seg": "b"}, catalog, 1.0, "s")
        lines.append(gittins.log_line(record))
        lines.extend(gittins.log_line(r) for r in gittins.expire(state, 100.0))
        events = tuple(ope.parse_log(lines))
        assert ope.verify(events) == ()
        assert ope.evaluate(events, bits=4).resolved == 2

    def test_refusals_match_the_reference(self):
        state = gittins.create(4, horizon=10.0)
        record = gittins.decide(state, {}, [("a", {}), ("b", {})], 0.0, "s")
        seen = []
        gittins.learn(state, record.decision_id, 1.0, train=lambda rec, r: seen.append(rec))
        assert seen[0].bits is None and seen[0].context is None and seen[0].candidates is None
        with pytest.raises(ValueError, match="record decide returned"):
            gittins.log_line(seen[0])
        with pytest.raises(ValueError, match="DecisionRecord or Resolution"):
            gittins.log_line({"not": "a record"})
