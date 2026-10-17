"""Canonical state serialization (design doc section 7).

The engine's state is an explicit value, and this module is what makes that
operational: `serialize` turns a BanditState into one self-contained byte
string — suitable for a flat file, a version-controlled artifact, or a
browser's localStorage — and `deserialize` turns it back, rejecting anything
malformed. Both are pure functions; serializing the same state produces the
same bytes on every platform and in every implementation, so the byte
strings themselves are golden-pinned (`spec/golden.json`, section
"serialization") and the compiled core must reproduce them exactly.

**Layout (format version 2).** Everything is little-endian, in the style of
`candidate_set_hash`: integers as 8-byte unsigned, floats as 8-byte IEEE-754
doubles, strings as an 8-byte length followed by UTF-8 bytes, lists as an
8-byte count followed by the elements.

    magic            8 bytes, b"gittins\\x00"
    format version   u64 (this layout is version 2)
    model            dim u64, ridge f64, forgetting f64, scale f64,
                     slot count u64, then per stored slot:
                       index u64, xx f64, xy f64
    bandit           next_seq u64, model_version u64, horizon f64,
                     default_reward f64, epsilon f64
    ledger           count u64, then per open decision record:
                       decision_id string, t f64, candidate_hash u64,
                       chosen u64, features (count u64, then per pair:
                       index u64, value f64), propensity f64,
                       model_version u64, salt string
    checksum         u64, FNV-1a over every preceding byte

**The model sums are sparse.** An update touches only its features'
slots and renormalization multiplies zeros to zeros, so almost every slot
of a hashed model stays exactly zero; writing all 2 * dim sums made a
state's size track `bits` rather than what it has learned. A slot is
stored exactly when its xx or its xy is not +0.0 *bitwise* — a -0.0 is
stored, since it is a different double — in strictly increasing index
order, and every slot not stored loads as +0.0 in both sums. Stored slots
that are +0.0 in both are rejected, so the encoding stays canonical.

Version 1 wrote the sums densely (xx[dim] f64, then xy[dim] f64, in
place of the slot list); it is no longer accepted.

**Deserialization validates.** The checksum rejects truncation and
corruption up front; the parse then re-checks every invariant the
constructors enforce (dim, forgetting, ridge, horizon, epsilon ranges;
`scale` in (0, 1] — the update rule keeps it there; each stored slot
strictly increasing, in-dimension, not +0.0 in both sums; each record's features
strictly increasing, in-dimension, nonzero) and requires the payload to be
exactly consumed. A deserialized state is therefore as trustworthy as a
constructed one; anything else raises ValueError. Model sums (`xx`, `xy`)
and record floats are *not* range-policed beyond structure — the engine
itself places no bounds on them.

`deserialize(serialize(state))` reproduces the state bit for bit, and
`serialize(deserialize(data))` reproduces accepted bytes exactly — the
format has one canonical encoding per state, no optional parts.
"""

import math
import struct

from gittins_reference.decide import BanditState, DecisionRecord
from gittins_reference.model import MAX_DIM, LinearModel
from gittins_reference.rng import fnv1a_64

MAGIC = b"gittins\x00"
FORMAT_VERSION = 2


def _u64(n: int) -> bytes:
    return n.to_bytes(8, "little")


def _f64(x: float) -> bytes:
    return struct.pack("<d", x)


def _is_positive_zero(x: float) -> bool:
    return x == 0.0 and math.copysign(1.0, x) > 0.0


def _string(s: str) -> bytes:
    encoded = s.encode("utf-8")
    return _u64(len(encoded)) + encoded


def serialize(state: BanditState) -> bytes:
    """The canonical byte string for one state; see the module docstring
    for the exact layout."""
    m = state.model
    out = bytearray()
    out += MAGIC
    out += _u64(FORMAT_VERSION)
    out += _u64(m.dim)
    out += _f64(m.ridge)
    out += _f64(m.forgetting)
    out += _f64(m.scale)
    stored = [
        j
        for j in range(m.dim)
        if not (_is_positive_zero(m.xx[j]) and _is_positive_zero(m.xy[j]))
    ]
    out += _u64(len(stored))
    for j in stored:
        out += _u64(j)
        out += _f64(m.xx[j])
        out += _f64(m.xy[j])
    out += _u64(state.next_seq)
    out += _u64(state.model_version)
    out += _f64(state.horizon)
    out += _f64(state.default_reward)
    out += _f64(state.epsilon)
    out += _u64(len(state.ledger))
    for r in state.ledger:
        out += _string(r.decision_id)
        out += _f64(r.t)
        out += _u64(r.candidate_hash)
        out += _u64(r.chosen)
        out += _u64(len(r.features))
        for j, v in r.features:
            out += _u64(j)
            out += _f64(v)
        out += _f64(r.propensity)
        out += _u64(r.model_version)
        out += _string(r.salt)
    out += _u64(fnv1a_64(bytes(out)))
    return bytes(out)


class _Reader:
    """A bounds-checked cursor over the payload (checksum already verified
    and stripped)."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise ValueError("truncated state")
        chunk = self.data[self.pos : self.pos + n]
        self.pos += n
        return chunk

    def u64(self) -> int:
        return int.from_bytes(self.take(8), "little")

    def f64(self) -> float:
        return struct.unpack("<d", self.take(8))[0]

    def string(self) -> str:
        length = self.u64()
        try:
            return self.take(length).decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("state string is not valid UTF-8") from None


def deserialize(data: bytes) -> BanditState:
    """Parse and validate one serialized state; raises ValueError on
    anything malformed."""
    if len(data) < len(MAGIC) + 16:  # magic + version + checksum, at minimum
        raise ValueError("truncated state")
    if data[: len(MAGIC)] != MAGIC:
        raise ValueError("not a gittins state (bad magic)")
    if int.from_bytes(data[-8:], "little") != fnv1a_64(data[:-8]):
        raise ValueError("state checksum mismatch")
    r = _Reader(data[:-8])
    r.take(len(MAGIC))
    version = r.u64()
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported state format version {version}")

    dim = r.u64()
    ridge = r.f64()
    forgetting = r.f64()
    scale = r.f64()
    if dim < 1:
        raise ValueError("dim must be at least 1")
    if dim > MAX_DIM:
        raise ValueError("dim must be at most 2**24")
    if not (0.0 < forgetting <= 1.0):
        raise ValueError("forgetting must be in (0, 1] (1.0 = never forget)")
    if not (ridge > 0.0):
        raise ValueError("ridge must be positive")
    if not (0.0 < scale <= 1.0):
        raise ValueError("scale must be in (0, 1]")
    xx = [0.0] * dim
    xy = [0.0] * dim
    prev = -1
    for _ in range(r.u64()):
        j = r.u64()
        u = r.f64()
        w = r.f64()
        if not (prev < j < dim) or (_is_positive_zero(u) and _is_positive_zero(w)):
            raise ValueError(
                "model slots must be in strictly increasing index order "
                "within the model dimension, never +0.0 in both sums"
            )
        xx[j] = u
        xy[j] = w
        prev = j
    xx = tuple(xx)
    xy = tuple(xy)

    next_seq = r.u64()
    model_version = r.u64()
    horizon = r.f64()
    default_reward = r.f64()
    epsilon = r.f64()
    if not (horizon > 0.0):
        raise ValueError("horizon must be positive")
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon must be in [0, 1]")

    ledger = []
    for _ in range(r.u64()):
        decision_id = r.string()
        t = r.f64()
        candidate_hash = r.u64()
        chosen = r.u64()
        features = []
        prev = -1
        for _ in range(r.u64()):
            j = r.u64()
            v = r.f64()
            if not (prev < j < dim) or v == 0.0:
                raise ValueError(
                    "record features must be (index, value) pairs in strictly "
                    "increasing index order within the model dimension, values nonzero"
                )
            features.append((j, v))
            prev = j
        propensity = r.f64()
        record_version = r.u64()
        salt = r.string()
        ledger.append(
            DecisionRecord(
                decision_id=decision_id,
                t=t,
                candidate_hash=candidate_hash,
                chosen=chosen,
                features=tuple(features),
                propensity=propensity,
                model_version=record_version,
                salt=salt,
            )
        )
    if r.pos != len(r.data):
        raise ValueError("trailing bytes in state")

    return BanditState(
        model=LinearModel(
            dim=dim, ridge=ridge, forgetting=forgetting, scale=scale, xx=xx, xy=xy
        ),
        next_seq=next_seq,
        model_version=model_version,
        horizon=horizon,
        default_reward=default_reward,
        epsilon=epsilon,
        ledger=tuple(ledger),
    )
