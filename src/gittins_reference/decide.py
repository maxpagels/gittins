"""Decision records and the decide layer (design doc D1, D5).

`decide` is a plain function from (state, candidates, time, salt) to a
decision record plus a new state. It never touches the disk, the network,
the clock, or a stateful random generator (D1): time comes in as `t`, and
randomness comes from the counter RNG keyed by the decision's identity.

`decide()` does not return a bare action (D5). It returns a **decision
record** — the unit the rest of the engine is built around: `learn`
(ledger.py) accepts only a record plus an outcome, and the append-only log
of records *is* the offline-evaluation dataset (R3). The record carries:

    decision_id    — unique; see below
    t              — the caller-supplied decision time
    candidate_hash — 64-bit hash of the full candidate set, so replay and
                     offline evaluation can verify they see what was scored
    chosen         — index into the candidate list
    features       — the chosen candidate's sparse (index, value) pairs
                     (what `learn` will train on; the record is
                     self-contained, and O(nonzeros) on disk)
    propensity     — the probability the choice was made with
    model_version  — how many observations the model had absorbed
                     (bumped by the ledger's trained resolutions)
    salt           — the RNG salt, so the draw is exactly replayable

Decision IDs are `"{salt}:{seq}"` where `seq` is a monotone counter in the
state. Uniqueness is structural, not probabilistic: within one agent the
counter never repeats, and across a fleet each agent must be given its own
salt (the same rule that keeps their random streams distinct). No hashing,
no collision math.

At this layer a candidate *is* its sparse feature pairs in the model's
space — (index, value) tuples in strictly increasing index order, values
nonzero; folding a context and an action description into those pairs is
the feature encoder's job (encoding.py), and the dict-shaped public API
arrives with it.

**Where epsilon comes from.** The exploration rule (exploration.py) takes
its uniform-exploration mass `epsilon` from the state: declared once at
construction, defaulting to `DEFAULT_EPSILON` — an expert override like
`forgetting`, not a routine knob (R2). The rule needs only the reward
estimates, so scoring is one dot product per candidate — no uncertainties,
no sqrt, no schedule. A fresh model estimates every candidate at 0.0 and
the exact-tie split makes the first distributions uniform; the epsilon
mass guarantees exploration (and re-exploration after a world shift) ever
after. Epsilon-greedy replaced the SquareCB rule and its uncertainty-driven
gamma schedule here — see exploration.py for the reasoning and the price.

RNG counter allocation: counter 0 of the decision's stream is the sampling
draw. Later counters are reserved for future per-decision randomness.
"""

import struct
from dataclasses import dataclass

from gittins_reference.exploration import (
    DEFAULT_EPSILON,
    epsilon_greedy_probabilities,
    sample_index,
)
from gittins_reference.model import (
    DEFAULT_FORGETTING,
    Features,
    LinearModel,
    estimate_factored,
    factorize,
    new_model,
)
from gittins_reference.rng import derive_key, fnv1a_64


@dataclass(frozen=True)
class BanditState:
    model: LinearModel
    next_seq: int  # sequence number of the next decision
    model_version: int  # observations absorbed so far (bumped by the ledger)
    horizon: float  # seconds an unresolved decision waits before expiring
    default_reward: float  # the reward an expired decision trains with
    epsilon: float  # probability mass spent uniformly per decision
    ledger: "tuple[DecisionRecord, ...]"  # open decisions awaiting resolution


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    t: float
    candidate_hash: int
    chosen: int
    features: tuple[tuple[int, float], ...]  # the chosen candidate's sparse pairs
    propensity: float
    model_version: int
    salt: str


def new_bandit(
    dim: int,
    horizon: float,
    default_reward: float = 0.0,
    epsilon: float = DEFAULT_EPSILON,
    forgetting: float = DEFAULT_FORGETTING,
) -> BanditState:
    """`horizon` and `default_reward` are the application's reward-handling
    declaration (design doc section 6), made once, up front: a decision with
    no reward after `horizon` seconds resolves as expired(`default_reward`).
    `epsilon` (uniform-exploration mass) and `forgetting` (the model's
    per-update tracking rate) are expert overrides, not default-API knobs;
    the right values for a deployment are selected offline from the decision
    log (see exploration.py, model.py)."""
    if not (horizon > 0.0):
        raise ValueError("horizon must be positive")
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon must be in [0, 1]")
    return BanditState(
        model=new_model(dim, forgetting),
        next_seq=0,
        model_version=0,
        horizon=horizon,
        default_reward=default_reward,
        epsilon=epsilon,
        ledger=(),
    )


def candidate_set_hash(candidates: "list[Features]", dim: int) -> int:
    """64-bit FNV-1a over the canonical sparse encoding of the candidate
    set: the count, then each candidate as the model dimension, its
    nonzero-entry count, and its (index, value) pairs in index order —
    integers as 8 little-endian bytes, values as little-endian IEEE-754
    doubles. Candidates already *are* that canonical form (sorted nonzero
    pairs), so hashing is O(nonzeros) with no conversion; the byte layout
    is unchanged from the dense-input formulation, so the same logical
    candidate set hashes to the same value it always did. Order-, value-,
    and shape-sensitive, identical on every platform."""
    data = bytearray(len(candidates).to_bytes(8, "little"))
    for x in candidates:
        data += dim.to_bytes(8, "little")
        data += len(x).to_bytes(8, "little")
        for i, v in x:
            data += i.to_bytes(8, "little")
            data += struct.pack("<d", v)
    return fnv1a_64(bytes(data))


def decide(
    state: BanditState, candidates: "list[Features]", t: float, salt: str
) -> tuple[DecisionRecord, BanditState]:
    """Score, explore, choose, and record one decision. Each candidate is
    sparse (index, value) pairs in strictly increasing index order, values
    nonzero, indices inside the model dimension (encoding.py's output).

    The model is untouched (learning happens only through the ledger's
    resolutions, see ledger.py); the state changes are the decision counter
    advancing and the record joining the ledger of open decisions.
    """
    if len(candidates) < 1:
        raise ValueError("need at least one candidate")
    dim = state.model.dim
    for x in candidates:
        prev = -1
        for j, v in x:
            if not (prev < j < dim):
                raise ValueError(
                    "candidate features must be (index, value) pairs in strictly "
                    "increasing index order within the model dimension"
                )
            if v == 0.0:
                raise ValueError("candidate feature values must be nonzero")
            prev = j

    # The weights depend on the model only, so they are solved once and
    # shared by every candidate.
    factored = factorize(state.model)
    estimates = [estimate_factored(factored, x) for x in candidates]
    p = epsilon_greedy_probabilities(estimates, state.epsilon)

    decision_id = f"{salt}:{state.next_seq}"
    key = derive_key(decision_id, salt)
    chosen = sample_index(p, key, 0)

    record = DecisionRecord(
        decision_id=decision_id,
        t=t,
        candidate_hash=candidate_set_hash(candidates, dim),
        chosen=chosen,
        features=tuple((j, v) for j, v in candidates[chosen]),
        propensity=p[chosen],
        model_version=state.model_version,
        salt=salt,
    )
    new_state = BanditState(
        model=state.model,
        next_seq=state.next_seq + 1,
        model_version=state.model_version,
        horizon=state.horizon,
        default_reward=state.default_reward,
        epsilon=state.epsilon,
        ledger=state.ledger + (record,),
    )
    return record, new_state
