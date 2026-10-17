import dataclasses
import struct

import pytest

from gittins_reference.decide import decide, new_bandit
from gittins_reference.encoding import encode
from gittins_reference.ledger import learn
from gittins_reference.rng import fnv1a_64
from gittins_reference.state import (
    FORMAT_VERSION,
    MAGIC,
    deserialize,
    serialize,
)


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


def with_slots(state, slots):
    """`state` with its model sums replaced: every slot +0.0 except
    `slots`, a {index: (xx, xy)} map."""
    m = state.model
    xx = [0.0] * m.dim
    xy = [0.0] * m.dim
    for j, (u, w) in slots.items():
        xx[j], xy[j] = u, w
    model = dataclasses.replace(m, xx=tuple(xx), xy=tuple(xy))
    return dataclasses.replace(state, model=model)


def resealed(payload):
    return bytes(payload) + fnv1a_64(bytes(payload)).to_bytes(8, "little")


def slot_list_bytes(state, entries):
    """`state`'s serialization with its slot list replaced by `entries`,
    raw (index, xx, xy) triples written as given, checksum recomputed."""
    head = serialize(with_slots(state, {}))[:-8]
    cut = 8 * 6  # magic, version, dim, ridge, forgetting, scale
    body = (len(entries)).to_bytes(8, "little")
    for j, u, w in entries:
        body += j.to_bytes(8, "little") + struct.pack("<dd", u, w)
    return resealed(head[:cut] + body + head[cut + 8 :])


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


class TestSparseSums:
    def test_size_tracks_learned_slots_not_dim(self):
        small = new_bandit(1 << 4, horizon=1.0)
        large = new_bandit(1 << 20, horizon=1.0)
        assert len(serialize(small)) == len(serialize(large))
        trained = with_slots(large, {7: (1.0, 0.5), 1 << 19: (2.0, -1.0)})
        assert len(serialize(trained)) == len(serialize(large)) + 2 * 24

    def test_negative_zero_is_stored_and_round_trips(self):
        state = with_slots(new_bandit(4, horizon=1.0), {1: (0.0, -0.0), 2: (-0.0, 0.0)})
        data = serialize(state)
        assert len(data) == len(serialize(new_bandit(4, horizon=1.0))) + 2 * 24
        loaded = deserialize(data)
        assert struct.pack("<d", loaded.model.xy[1]) == struct.pack("<d", -0.0)
        assert struct.pack("<d", loaded.model.xx[2]) == struct.pack("<d", -0.0)
        assert serialize(loaded) == data

    def test_version_1_is_rejected(self):
        data = bytearray(serialize(worked_state()))[:-8]
        data[8:16] = (1).to_bytes(8, "little")
        with pytest.raises(ValueError, match="unsupported state format version 1"):
            deserialize(resealed(data))

    @pytest.mark.parametrize(
        "entries",
        [
            [(1, 0.0, 0.0)],  # +0.0 in both: must be left out instead
            [(2, 1.0, 1.0), (1, 1.0, 1.0)],  # decreasing
            [(1, 1.0, 1.0), (1, 2.0, 2.0)],  # repeated
            [(4, 1.0, 1.0)],  # outside dim 4
        ],
    )
    def test_non_canonical_slot_lists_are_rejected(self, entries):
        data = slot_list_bytes(new_bandit(4, horizon=1.0), entries)
        with pytest.raises(ValueError, match="slots"):
            deserialize(data)

    def test_dim_above_the_cap_is_rejected_before_allocating(self):
        data = bytearray(serialize(new_bandit(1, horizon=1.0)))[:-8]
        data[16:24] = (1 << 62).to_bytes(8, "little")
        with pytest.raises(ValueError, match="dim"):
            deserialize(resealed(data))


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
        data[40] ^= 0x01  # inside the model's scale
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
        # fixed offset: 6 u64 header/model scalars + an empty slot list
        # (nothing learned yet) + 5 bandit scalars + ledger count + id
        # (8 + 3 bytes) + t + hash + chosen + feature count.
        state = new_bandit(4, horizon=1.0)
        _, state = decide(state, [((0, 1.0),), ((1, 1.0),)], 0.0, "s")
        offset = 8 * 6 + 8 + 8 * 5 + 8 + (8 + 3) + 8 * 4
        data = bytearray(serialize(state))[:-8]
        assert data[offset : offset + 8] == state.ledger[0].features[0][0].to_bytes(8, "little")
        data[offset : offset + 8] = (9).to_bytes(8, "little")  # 9 >= dim
        data += fnv1a_64(bytes(data)).to_bytes(8, "little")
        with pytest.raises(ValueError, match="features"):
            deserialize(bytes(data))
