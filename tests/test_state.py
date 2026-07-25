import pytest

from gittins_reference.decide import decide, new_bandit
from gittins_reference.encoding import encode
from gittins_reference.ledger import learn
from gittins_reference.rng import fnv1a_64
from gittins_reference.state import FORMAT_VERSION, MAGIC, deserialize, serialize


def worked_state():
    """A state with history: trained model, advanced counters, two open
    ledger records — every field of the format populated."""
    state = new_bandit(1 << 4, horizon=120.0, default_reward=-0.25, epsilon=0.1)
    for i in range(6):
        candidates = [encode({"seg": str(i % 2)}, arm, {}, 4) for arm in ("x", "y")]
        record, state = decide(state, candidates, 1000.0 + i, "salty")
        if i < 4:  # the last two stay open
            _, state = learn(state, record.decision_id, 1.0 if i % 2 else -1.0, 1000.0 + i)
    return state


class TestRoundTrip:
    def test_deserialize_inverts_serialize_bit_for_bit(self):
        state = worked_state()
        assert deserialize(serialize(state)) == state

    def test_serialize_inverts_deserialize_byte_for_byte(self):
        data = serialize(worked_state())
        assert serialize(deserialize(data)) == data

    def test_fresh_state_round_trips(self):
        state = new_bandit(2, horizon=1.0)
        assert deserialize(serialize(state)) == state

    def test_serialization_is_deterministic(self):
        assert serialize(worked_state()) == serialize(worked_state())


class TestRejection:
    """Every malformed input raises ValueError; nothing is best-effort."""

    def test_bad_magic(self):
        data = bytearray(serialize(worked_state()))
        data[0] ^= 0xFF
        with pytest.raises(ValueError, match="magic|checksum"):
            deserialize(bytes(data))

    def test_unsupported_version(self):
        # Rebuild with a bumped version and a valid checksum: the version
        # check itself must reject it, not the checksum.
        data = bytearray(serialize(worked_state()))[:-8]
        data[8:16] = (FORMAT_VERSION + 1).to_bytes(8, "little")
        data += fnv1a_64(bytes(data)).to_bytes(8, "little")
        with pytest.raises(ValueError, match="version"):
            deserialize(bytes(data))

    def test_corrupted_payload_fails_the_checksum(self):
        data = bytearray(serialize(worked_state()))
        data[40] ^= 0x01  # somewhere inside the model sums
        with pytest.raises(ValueError, match="checksum"):
            deserialize(bytes(data))

    def test_truncation(self):
        data = serialize(worked_state())
        for cut in (0, 7, len(MAGIC) + 15, len(data) // 2, len(data) - 1):
            with pytest.raises(ValueError):
                deserialize(data[:cut])

    def test_trailing_bytes(self):
        data = serialize(worked_state())[:-8] + b"\x00"
        data += fnv1a_64(data).to_bytes(8, "little")
        with pytest.raises(ValueError, match="trailing|truncated"):
            deserialize(data)

    def test_constructor_invariants_are_revalidated(self):
        # Field offsets: dim at 16, ridge 24, forgetting 32, scale 40.
        for offset, bad, message in [
            (16, (0).to_bytes(8, "little"), "dim"),
            (24, b"\x00" * 8, "ridge"),  # ridge = 0.0
            (32, b"\x00" * 8, "forgetting"),  # forgetting = 0.0
            (40, b"\x00\x00\x00\x00\x00\x00\x00\x40", "scale"),  # scale = 2.0
        ]:
            data = bytearray(serialize(new_bandit(1, horizon=1.0)))[:-8]
            data[offset : offset + 8] = bad
            data += fnv1a_64(bytes(data)).to_bytes(8, "little")
            with pytest.raises(ValueError, match=message):
                deserialize(bytes(data))

    def test_record_features_are_revalidated(self):
        # Corrupt an open record's feature index to sit outside the model
        # dimension: structural validation must catch what the checksum
        # cannot (a "valid" file written by broken code). One decision at
        # dim 4 with salt "s" puts the record's first feature index at a
        # fixed offset: 6 u64 header/model scalars + 2*dim sums + 5 bandit
        # scalars + ledger count + id (8 + 3 bytes) + t + hash + chosen +
        # feature count.
        state = new_bandit(4, horizon=1.0)
        _, state = decide(state, [((0, 1.0),), ((1, 1.0),)], 0.0, "s")
        offset = 8 * 6 + 16 * 4 + 8 * 5 + 8 + (8 + 3) + 8 * 4
        data = bytearray(serialize(state))[:-8]
        assert data[offset : offset + 8] == state.ledger[0].features[0][0].to_bytes(8, "little")
        data[offset : offset + 8] = (9).to_bytes(8, "little")  # 9 >= dim
        data += fnv1a_64(bytes(data)).to_bytes(8, "little")
        with pytest.raises(ValueError, match="features"):
            deserialize(bytes(data))
