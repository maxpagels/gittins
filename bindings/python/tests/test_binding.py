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
    state = gittins.create(bits=s["bits"], horizon=s["horizon"], forgetting=s["forgetting"])
    catalog = [(arm_id, action) for arm_id, action in s["catalog"]]
    records = []
    for event in s["events"]:
        record, state = gittins.decide(state, event["context"], catalog, event["t"], s["salt"])
        records.append((record, event["record"]))
    resolutions = []
    _, expected0 = records[0]
    _, expected1 = records[1]
    r, state = gittins.learn(state, expected1["decision_id"], 1.0)
    resolutions.append(r)
    r, state = gittins.learn(state, expected0["decision_id"], 0.0)
    resolutions.append(r)
    r, state = gittins.censor(state, records[2][1]["decision_id"])
    resolutions.append(r)
    expired, state = gittins.expire(state, s["expire_sweep_at"])
    resolutions.extend(expired)
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

    def test_final_state_bytes_match_the_corpus(self):
        s, state, _, _ = replay_api_section()
        data = gittins.serialize(state)
        assert data.hex() == s["final"]["state_hex"]
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
            b_record, bound = gittins.decide(bound, context, catalog, float(i), "eq")
            p_record, pure = api.decide(pure, context, catalog, float(i), "eq")
            assert b_record.decision_id == p_record.decision_id
            assert b_record.chosen == p_record.chosen
            assert b_record.propensity == p_record.propensity
            assert list(b_record.features) == list(p_record.features)
            assert b_record.candidate_hash == p_record.candidate_hash
            if i % 4 == 0:
                _, bound = gittins.learn(bound, b_record.decision_id, 1.0)
                _, pure = api.learn(pure, p_record.decision_id, 1.0)
            if i % 7 == 0:
                _, bound = gittins.expire(bound, float(i))
                _, pure = api.expire(pure, float(i))
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
        with pytest.raises(ValueError, match="checksum|truncated"):
            gittins.deserialize(gittins.serialize(state)[:-1])

    def test_unknown_resolutions_are_none(self):
        state = gittins.create(4, horizon=10.0)
        resolution, state = gittins.learn(state, "never-made:0", 1.0)
        assert resolution is None
        resolutions, state = gittins.expire(state, 1e9)
        assert resolutions == ()

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
        record, state = gittins.decide(state, context, candidates, 0.0, "agent-1")
        assert candidates[record.chosen][0].startswith("banner-")
        resolution, state = gittins.learn(state, record.decision_id, 1.0)
        assert resolution.kind == "rewarded"
        record, state = gittins.decide(state, context, candidates, 1.0, "agent-1")
        resolution, state = gittins.censor(state, record.decision_id)
        assert resolution.kind == "censored"
        assert gittins.model_bits(state) == 8
        restored = gittins.deserialize(gittins.serialize(state))
        assert gittins.serialize(restored) == gittins.serialize(state)
