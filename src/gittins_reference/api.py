"""The public API: the dict-shaped facade every binding mirrors.

This module is the engine's complete public surface — the layer the Python
and browser bindings expose, name for name and shape for shape, and nothing
else. The rule it exists to enforce: the public API is specified once, here
in the reference, and every binding mirrors it exactly; a binding's CI gate
is the golden `api` section replayed through these functions alone.

The surface is eight names:

    create(bits, horizon, ...)          -> BanditState (a handle)
    decide(state, context, candidates, t, salt) -> DecisionRecord
    learn(state, decision_id, reward)   -> Resolution | None
    censor(state, decision_id)          -> Resolution | None
    expire(state, t)                    -> (Resolution, ...)
    serialize(state)                    -> str (hex)
    deserialize(data)                   -> BanditState
    model_bits(state)                   -> int

**The state is an opaque handle, updated in place.** `create` and
`deserialize` return one; `decide`/`learn`/`censor`/`expire` mutate it and
return only their results. This is the uniform convention across every
implementation (a JS binding cannot return (result, state) tuples
idiomatically, so nothing does). It is purely a facade choice: the
reference's internal layers (decide.py, ledger.py, state.py) remain pure
functions over immutable values, and the handle is a one-field cell that
swaps which immutable value it holds — snapshotting, rollback, and replay
stay one `serialize` away.

`create` is `new_bandit` with the single encoding declaration, `bits`, in
place of a raw model dimension: the model is 2**bits-dimensional, and the
declaration is recoverable from any state built here (`model_bits`), so no
new state format and no serialization change is needed. `decide` takes what
the caller actually has — a context dict and candidates as (arm_id,
action dict) pairs — and folds the hashed encoding in, so hashing never
crosses the public boundary; everything else about the decision path
(scoring, epsilon-greedy, the record, the ledger) is decide.py unchanged.

Feature dicts follow encoding.py's contract: string values are categorical
tokens, numbers (bools included) are numeric contributions, None means
absent. Duplicate candidates are allowed and score identically, exactly as
duplicate encodings do in the layered API.
"""

from gittins_reference import ledger as _ledger
from gittins_reference import state as _state
from gittins_reference.decide import DecisionRecord, new_bandit
from gittins_reference.decide import decide as _decide_encoded
from gittins_reference.encoding import encode
from gittins_reference.exploration import DEFAULT_EPSILON
from gittins_reference.ledger import Resolution
from gittins_reference.model import DEFAULT_FORGETTING

__all__ = [
    "BanditState",
    "create",
    "decide",
    "learn",
    "censor",
    "expire",
    "serialize",
    "deserialize",
    "model_bits",
]


class BanditState:
    """An opaque handle on the bandit's state, updated in place by
    decide/learn/censor/expire. Snapshot or persist it with `serialize`;
    the value inside stays immutable, so aliases of the handle always
    observe the current state."""

    __slots__ = ("_state",)

    def __init__(self, state):
        self._state = state


def create(
    bits: int,
    horizon: float,
    default_reward: float = 0.0,
    epsilon: float = DEFAULT_EPSILON,
    forgetting: float = DEFAULT_FORGETTING,
) -> BanditState:
    """A fresh bandit whose model spans the 2**bits hashed feature space —
    `bits` is the one encoding declaration (encoding.py); every other
    parameter is new_bandit's, with the same meanings and defaults."""
    if not (1 <= bits <= 24):
        raise ValueError("bits must be between 1 and 24")
    return BanditState(
        new_bandit(
            1 << bits,
            horizon=horizon,
            default_reward=default_reward,
            epsilon=epsilon,
            forgetting=forgetting,
        )
    )


def model_bits(state: BanditState) -> int:
    """The `bits` declaration recovered from the model dimension. States
    built by `create` always satisfy dim == 2**bits; anything else was
    built against the layered API and has no public encoding space."""
    dim = state._state.model.dim
    bits = dim.bit_length() - 1
    if dim != 1 << bits or not (1 <= bits <= 24):
        raise ValueError(
            "model dimension is not 2**bits for bits in [1, 24]; "
            "the state was not built by create()"
        )
    return bits


def decide(
    state: BanditState,
    context: dict,
    candidates: "list[tuple[str, dict]]",
    t: float,
    salt: str,
) -> DecisionRecord:
    """Score, explore, choose, and record one decision over dict-shaped
    inputs: `context` is one feature dict, each candidate is an
    (arm_id, action feature dict) pair, encoded here in candidate order.
    The record returned and every state change are decide.py's, unchanged."""
    bits = model_bits(state)
    encoded = [encode(context, arm_id, action, bits) for arm_id, action in candidates]
    record, state._state = _decide_encoded(state._state, encoded, t, salt)
    return record


def learn(state: BanditState, decision_id: str, reward: float) -> "Resolution | None":
    """Resolve an open decision as rewarded; None (a no-op) if the id is
    unknown or already resolved."""
    resolution, state._state = _ledger.learn(state._state, decision_id, reward)
    return resolution


def censor(state: BanditState, decision_id: str) -> "Resolution | None":
    """Resolve an open decision as censored (excluded from training, on
    record); None (a no-op) if the id is unknown or already resolved."""
    resolution, state._state = _ledger.censor(state._state, decision_id)
    return resolution


def expire(state: BanditState, t: float) -> "tuple[Resolution, ...]":
    """Resolve every open decision past its horizon at time `t` as expired,
    in ledger order."""
    resolutions, state._state = _ledger.expire(state._state, t)
    return resolutions


def serialize(state: BanditState) -> str:
    """The current state as one plain string: the canonical byte layout
    (state.py), hex-encoded, lowercase. A string goes anywhere text goes —
    a file, a database column, localStorage, version control — so no
    caller, in any language, ever handles raw bytes or picks an encoding."""
    return _state.serialize(state._state).hex()


def deserialize(data: str) -> BanditState:
    """A handle on a state parsed and validated from a `serialize` string;
    raises ValueError on anything malformed. Hex digits of either case are
    accepted; the canonical form `serialize` emits is lowercase."""
    try:
        raw = bytes.fromhex(data)
    except (ValueError, TypeError):
        raise ValueError("state must be a hexadecimal string") from None
    if len(data) != 2 * len(raw):  # fromhex tolerates whitespace; the API does not
        raise ValueError("state must be a hexadecimal string")
    return BanditState(_state.deserialize(raw))
