"""The public API: the dict-shaped facade every binding mirrors.

This module is the engine's complete public surface — the layer PR 20/21's
Python and browser bindings expose, name for name, and nothing else. The
rule it exists to enforce: the public API is specified once, here in the
reference, and every binding mirrors it exactly; a binding's CI gate is the
golden `api` section replayed through these functions alone.

The surface is eight names:

    create(bits, horizon, ...)          -> BanditState
    decide(state, context, candidates, t, salt)
                                        -> (DecisionRecord, BanditState)
    learn(state, decision_id, reward)   -> (Resolution | None, BanditState)
    censor(state, decision_id)          -> (Resolution | None, BanditState)
    expire(state, t)                    -> ((Resolution, ...), BanditState)
    serialize(state)                    -> bytes
    deserialize(data)                   -> BanditState
    model_bits(state)                   -> int

`create` is `new_bandit` with the single encoding declaration, `bits`, in
place of a raw model dimension: the model is 2**bits-dimensional, and the
declaration is recoverable from any state built here (`model_bits`), so no
new state type and no serialization change is needed. `decide` takes what
the caller actually has — a context dict and candidates as (arm_id,
action dict) pairs — and folds the hashed encoding in, so hashing never
crosses the public boundary; everything else about the decision path
(scoring, epsilon-greedy, the record, the ledger) is decide.py unchanged.
Resolution and serialization pass through as they are.

Feature dicts follow encoding.py's contract: string values are categorical
tokens, numbers (bools included) are numeric contributions, None means
absent. Duplicate candidates are allowed and score identically, exactly as
duplicate encodings do in the layered API.
"""

from gittins_reference.decide import BanditState, DecisionRecord, new_bandit
from gittins_reference.decide import decide as decide_encoded
from gittins_reference.encoding import encode
from gittins_reference.exploration import DEFAULT_EPSILON
from gittins_reference.ledger import censor, expire, learn  # re-exported unchanged
from gittins_reference.model import DEFAULT_FORGETTING
from gittins_reference.state import deserialize, serialize  # re-exported unchanged

__all__ = [
    "create",
    "decide",
    "learn",
    "censor",
    "expire",
    "serialize",
    "deserialize",
    "model_bits",
]


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
    return new_bandit(
        1 << bits,
        horizon=horizon,
        default_reward=default_reward,
        epsilon=epsilon,
        forgetting=forgetting,
    )


def model_bits(state: BanditState) -> int:
    """The `bits` declaration recovered from the model dimension. States
    built by `create` always satisfy dim == 2**bits; anything else was
    built against the layered API and has no public encoding space."""
    dim = state.model.dim
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
) -> "tuple[DecisionRecord, BanditState]":
    """Score, explore, choose, and record one decision over dict-shaped
    inputs: `context` is one feature dict, each candidate is an
    (arm_id, action feature dict) pair, encoded here in candidate order.
    Everything returned and every state change is decide.py's, unchanged."""
    bits = model_bits(state)
    encoded = [encode(context, arm_id, action, bits) for arm_id, action in candidates]
    return decide_encoded(state, encoded, t, salt)
