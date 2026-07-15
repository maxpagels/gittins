from collections import Counter

import pytest

from gittins_reference.decide import decide, new_bandit
from gittins_reference.encoding import encode, feature_tokens
from gittins_reference.ledger import learn

HOUR = 3600.0
DAY = 24 * HOUR
T0 = 1_752_000_000.0


class TestFeatureTokens:
    def test_value_type_picks_the_token_shape(self):
        # Strings put the value in the token (categorical, contribution 1);
        # numbers keep the name-only token and contribute their value.
        assert feature_tokens("c", {"device": "mobile"}) == [("c|device=mobile", 1.0)]
        assert feature_tokens("c", {"hour": 14}) == [("c|hour", 14.0)]
        assert feature_tokens("c", {"clicked": True}) == [("c|clicked", 1.0)]

    def test_none_means_absent(self):
        assert feature_tokens("c", {"hour": None}) == []

    def test_sorted_token_order(self):
        assert feature_tokens("c", {"b": 1.0, "a": 2.0}) == [("c|a", 2.0), ("c|b", 1.0)]

    def test_unsupported_type_is_an_error(self):
        with pytest.raises(ValueError):
            feature_tokens("c", {"bad": [1, 2]})


class TestEncode:
    def test_shape_and_determinism(self):
        v = encode({"hour": 14}, "arm", {"price": 1.0}, 6)
        assert len(v) == 64
        assert v == encode({"hour": 14}, "arm", {"price": 1.0}, 6)

    def test_independent_of_dict_insertion_order(self):
        # Tokens are processed sorted, so the accumulation order — and every
        # output bit — ignores how the caller built the dicts.
        a = encode({"h": 1.0, "d": 2.0}, "z", {"p": 3.0, "q": 4.0}, 6)
        b = encode({"d": 2.0, "h": 1.0}, "z", {"q": 4.0, "p": 3.0}, 6)
        assert a == b

    def test_everything_is_open_world(self):
        # No schema: never-seen field names, values, and arms all encode
        # without registration; empty dicts still give bias + identity.
        v = encode({}, "brand-new-arm", {}, 10)
        assert sorted(abs(x) for x in v if x != 0.0) == [1.0, 1.0]
        w = encode({"invented-just-now": "novel-value"}, "brand-new-arm", {}, 10)
        assert sorted(abs(x) for x in w if x != 0.0) == [1.0] * 4

    def test_outer_product_structure(self):
        # ({c:2}, arm, {a:3}): left tokens (bias, c|c) x right tokens
        # (bias, a|a, i|z) = 6 contributions: 1*1, 1*3, 1*1, 2*1, 2*3, 2*1.
        v = encode({"c": 2.0}, "z", {"a": 3.0}, 10)
        assert sorted(abs(x) for x in v if x != 0.0) == [1.0, 1.0, 2.0, 2.0, 3.0, 6.0]

    def test_identity_and_categorical_values_distinguish(self):
        assert encode({}, "arm-1", {"p": 1.0}, 8) != encode({}, "arm-2", {"p": 1.0}, 8)
        assert encode({}, "z", {"color": "red"}, 8) != encode({}, "z", {"color": "blue"}, 8)

    def test_collisions_add_instead_of_crashing(self):
        # bits=1 forces all six contributions into two slots; they sum.
        assert encode({"c": 2.0}, "z", {"a": 3.0}, 1) == [-4.0, -5.0]

    def test_rejects_bad_bits(self):
        with pytest.raises(ValueError):
            encode({}, "z", {}, 0)
        with pytest.raises(ValueError):
            encode({}, "z", {}, 25)

    def test_pinned_vector(self):
        # Golden vector, bits=4: the design doc's shop example, bit for
        # bit. The 1.5 cell is a live collision (1.0 + 0.5 sharing a slot)
        # — pinned deliberately: collision behavior is part of the spec.
        v = encode(
            {"hour": 0.5, "device": "mobile"}, "banner-sale", {"color": "red", "price": 1}, 4
        )
        assert v == [
            1.0, 0.5, 1.5, 0.0, 0.0, 0.0, 0.5, 0.0,
            1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0,
        ]


class TestPersonalization:
    def test_bandit_learns_per_context_preferences_end_to_end(self):
        # The whole stack: hashed encode -> decide -> ledger. Arms "x" and
        # "y" have no descriptive features — identity tokens only — and pay
        # oppositely in segments "a" and "b": learning this requires the
        # identity tokens *and* the hashed context-action interactions.
        # bits=5 (dim 32) keeps the reference solve fast; collisions and
        # all, the preference is learned.
        bits = 5
        state = new_bandit(1 << bits, DAY, T0, horizon=DAY)
        per_seg = {"a": [], "b": []}
        for i in range(300):
            seg = "a" if i % 2 == 0 else "b"
            cands = [encode({"seg": seg}, arm, {}, bits) for arm in ["x", "y"]]
            record, state = decide(state, cands, T0 + float(i), "loop")
            arm = ["x", "y"][record.chosen]
            reward = 1.0 if (arm == "x") == (seg == "a") else 0.0
            _, state = learn(state, record.decision_id, reward)
            per_seg[seg].append(record.chosen)
        assert Counter(per_seg["a"][-50:])[0] > 40  # segment a wants arm x
        assert Counter(per_seg["b"][-50:])[1] > 40  # segment b wants arm y
