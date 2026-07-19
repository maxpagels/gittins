"""The public API: the dict-shaped facade every binding mirrors.

This module is the engine's complete public surface — the layer the Python
and browser bindings expose, name for name and shape for shape, and nothing
else. The rule it exists to enforce: the public API is specified once, here
in the reference, and every binding mirrors it exactly; a binding's CI gate
is the golden `api` section replayed through these functions alone.

The surface is nine names:

    create(bits, horizon, ...)          -> BanditState (a handle)
    decide(state, context, candidates, t, salt,
           score=None, explore=None)    -> DecisionRecord
    learn(state, decision_id, reward, train=None) -> Resolution | None
    censor(state, decision_id)          -> Resolution | None
    expire(state, t, train=None)        -> (Resolution, ...)
    serialize(state)                    -> str (hex)
    deserialize(data)                   -> BanditState
    model_bits(state)                   -> int
    log_line(record | resolution)       -> str (one experience-log line)

**Assembly-free logging.** The record `decide` returns
carries the inputs the decision was made over — `bits`, `context`,
`candidates`, attached at the only moment they exist — and `log_line`
turns it (or any resolution) into one canonical experience-log line, so
building an OPE-ready log is appending what each call returns:

    f.write(log_line(record) + "\n")      # after decide
    f.write(log_line(resolution) + "\n")  # after learn / censor / expire

The three input fields live only on records returned by `decide`: the
ledger keeps the compact record (fixed memory, unchanged serialization),
so a record that crossed the state boundary — the one a `train` callback
receives — carries None there. The log, not the state, is where inputs
persist; `candidate_hash` is the commitment binding the two.

**Bring your own model / exploration.** The three
optional callbacks are the whole BYO surface, each replacing exactly one
built-in component while everything else — encoding, the RNG draw, the
record, propensity logging, the ledger's exactly-once join, OPE — is
inherited unchanged:

    score(context, candidates)  -> one reward estimate per candidate, called
                                   with exactly the objects the caller passed
    explore(estimates, epsilon) -> one probability per candidate; the engine
                                   still draws from it deterministically
    train(record, reward)       -> the user model's training tap: called with
                                   the resolved decision's record and the
                                   reward, exactly once per resolution, in
                                   place of the built-in model's update
                                   (model_version still advances)

A BYO model is `score` + `train`; a BYO exploration rule is `explore`.
Callbacks are per-call values, never stored, so the state stays plain
serializable data. `train` fires *after* its resolution commits: a raising
callback loses that one training example loudly, but can never cause a
double-train — the record is already spent.

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

import json as _json
from dataclasses import dataclass
from dataclasses import replace as _replace

from gittins_reference import ledger as _ledger
from gittins_reference import state as _state
from gittins_reference.decide import new_bandit
from gittins_reference.decide import decide as _decide_encoded
from gittins_reference.encoding import encode
from gittins_reference.exploration import DEFAULT_EPSILON
from gittins_reference.ledger import Resolution
from gittins_reference.model import DEFAULT_FORGETTING

__all__ = [
    "BanditState",
    "DecisionRecord",
    "create",
    "decide",
    "learn",
    "censor",
    "expire",
    "serialize",
    "deserialize",
    "model_bits",
    "log_line",
]


@dataclass(frozen=True)
class DecisionRecord:
    """The public decision record: decide.py's compact record plus, on
    records returned by `decide`, the inputs the decision was made over
    (`bits`, `context`, `candidates` — the caller's own objects, attached
    at the only moment they exist), so a record logs as-is (`log_line`).
    A record that crossed the state boundary — the one a `train` callback
    receives — carries None in those three fields: the ledger keeps only
    the compact record, and the log is where inputs persist."""

    decision_id: str
    t: float
    candidate_hash: int
    chosen: int
    features: tuple[tuple[int, float], ...]
    propensity: float
    model_version: int
    salt: str
    bits: "int | None" = None
    context: "dict | None" = None
    candidates: "list[tuple[str, dict]] | None" = None


def _public_record(record, bits=None, context=None, candidates=None) -> DecisionRecord:
    return DecisionRecord(
        decision_id=record.decision_id,
        t=record.t,
        candidate_hash=record.candidate_hash,
        chosen=record.chosen,
        features=record.features,
        propensity=record.propensity,
        model_version=record.model_version,
        salt=record.salt,
        bits=bits,
        context=context,
        candidates=candidates,
    )


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
    forgetfulness: float = DEFAULT_FORGETTING,
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
            forgetting=forgetfulness,
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
    score=None,
    explore=None,
) -> DecisionRecord:
    """Score, explore, choose, and record one decision over dict-shaped
    inputs: `context` is one feature dict, each candidate is an
    (arm_id, action feature dict) pair, encoded here in candidate order.
    The record returned and every state change are decide.py's, unchanged.

    `score` and `explore` are the BYO callbacks (module docstring);
    `score` is called once with the very `context` and
    `candidates` passed here.

    The returned record carries `bits` and the very `context` and
    `candidates` passed in (no copies), so it logs as-is via
    `log_line`."""
    bits = model_bits(state)
    encoded = [encode(context, arm_id, action, bits) for arm_id, action in candidates]
    record, state._state = _decide_encoded(
        state._state,
        encoded,
        t,
        salt,
        score=None if score is None else (lambda _encoded: score(context, candidates)),
        explore=explore,
    )
    return _public_record(record, bits=bits, context=context, candidates=candidates)


def learn(
    state: BanditState, decision_id: str, reward: float, train=None
) -> "Resolution | None":
    """Resolve an open decision as rewarded; None (a no-op) if the id is
    unknown or already resolved.

    With `train` (the BYO model's training tap), the built-in
    model is left untouched: the resolution commits — the record leaves the
    ledger and model_version advances — and then `train(record, reward)`
    fires, exactly once per decision."""
    if train is None:
        resolution, state._state = _ledger.learn(state._state, decision_id, reward)
        return resolution
    record, rest = _ledger.take(state._state.ledger, decision_id)
    if record is None:
        return None
    state._state = _replace(
        state._state, model_version=state._state.model_version + 1, ledger=rest
    )
    train(_public_record(record), reward)
    return Resolution(decision_id, _ledger.REWARDED, reward)


def censor(state: BanditState, decision_id: str) -> "Resolution | None":
    """Resolve an open decision as censored (excluded from training, on
    record); None (a no-op) if the id is unknown or already resolved."""
    resolution, state._state = _ledger.censor(state._state, decision_id)
    return resolution


def expire(state: BanditState, t: float, train=None) -> "tuple[Resolution, ...]":
    """Resolve every open decision past its horizon at time `t` as expired,
    in ledger order.

    With `train` each expiry commits and then fires
    `train(record, default_reward)` — one record at a time, so a raising
    callback leaves earlier records resolved and later ones open for the
    next sweep, and no record ever trains twice."""
    if train is None:
        resolutions, state._state = _ledger.expire(state._state, t)
        return resolutions
    due = [r for r in state._state.ledger if r.t + state._state.horizon <= t]
    resolutions = []
    for record in due:
        taken, rest = _ledger.take(state._state.ledger, record.decision_id)
        if taken is None:  # a re-entrant callback resolved it already
            continue
        state._state = _replace(
            state._state, model_version=state._state.model_version + 1, ledger=rest
        )
        resolutions.append(
            Resolution(record.decision_id, _ledger.EXPIRED, state._state.default_reward)
        )
        train(_public_record(record), state._state.default_reward)
    return tuple(resolutions)


def log_line(item) -> str:
    """One canonical experience-log line for a decision
    record or a resolution — exactly what `verify`/`evaluate`/`replay`
    consume, so building an OPE-ready log is appending this, verbatim,
    after each call. Compact JSON, canonical field order, feature dicts
    kept in their own (insertion) order — the order they encoded with.

    Decision records must be the ones `decide` returned: those carry the
    inputs. A record from a `train` callback does not (its inputs are
    already in your log, written at decide time) and is refused."""
    if isinstance(item, DecisionRecord):
        if item.bits is None or item.context is None or item.candidates is None:
            raise ValueError("log_line needs the record decide returned (it carries the inputs)")
        return _json.dumps(
            {
                "event": "decision",
                "bits": item.bits,
                "context": item.context,
                "candidates": [[arm_id, action] for arm_id, action in item.candidates],
                "record": {
                    "decision_id": item.decision_id,
                    "t": item.t,
                    "candidate_hash": item.candidate_hash,
                    "chosen": item.chosen,
                    "features": [[j, v] for j, v in item.features],
                    "propensity": item.propensity,
                    "model_version": item.model_version,
                    "salt": item.salt,
                },
            },
            separators=(",", ":"),
        )
    if isinstance(item, Resolution):
        return _json.dumps(
            {
                "event": "resolution",
                "decision_id": item.decision_id,
                "kind": item.kind,
                "reward": item.reward,
            },
            separators=(",", ":"),
        )
    raise ValueError("log_line takes a DecisionRecord or Resolution")


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
