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
    features       — the chosen candidate's feature vector (what `learn`
                     will train on; the record is self-contained)
    propensity     — the floored probability the choice was made with
    model_version  — how many observations the model had absorbed
                     (bumped by the ledger's trained resolutions)
    salt           — the RNG salt, so the draw is exactly replayable

Decision IDs are `"{salt}:{seq}"` where `seq` is a monotone counter in the
state. Uniqueness is structural, not probabilistic: within one agent the
counter never repeats, and across a fleet each agent must be given its own
salt (the same rule that keeps their random streams distinct). No hashing,
no collision math.

At this layer a candidate *is* its feature vector in the model's space;
folding a context and an action description into that vector is the feature
encoder's job (PR 8), and the dict-shaped public API arrives with it.

**Where gamma comes from.** The exploration rule (PR 5) left its greediness
`gamma` to this layer. It is derived from the model's own uncertainty:

    gamma = GAMMA_SCALE / mean(uncertainty over the candidates)

A fresh model has uncertainty ~1 on unit-scale features, so gamma ~1 and
the distribution is near-uniform; uncertainty shrinks like 1/sqrt(n) as
evidence accumulates, so gamma grows like sqrt(n) — the schedule SquareCB's
guarantee wants — with no clock or knob. Because decayed sums bound the
effective sample size, gamma is bounded too: the system never becomes fully
greedy (R2), and after a world shift uncertainty regrows and gamma falls
back on its own. GAMMA_SCALE is a fixed engine constant; its value (and
whether the mean is the right aggregate) is provisional until the Phase 2
battery, but the *interface* — gamma is computed here, never asked of the
user — is settled.

RNG counter allocation: counter 0 of the decision's stream is the sampling
draw. Later counters are reserved for future per-decision randomness.
"""

import struct
from dataclasses import dataclass

from gittins_reference.exploration import (
    apply_floor,
    inverse_gap_probabilities,
    sample_index,
)
from gittins_reference.model import LinearModel, factorize, new_model, predict_factored
from gittins_reference.rng import derive_key, fnv1a_64

# Fixed scale of the uncertainty-driven gamma schedule; not a user knob.
GAMMA_SCALE = 1.0


@dataclass(frozen=True)
class BanditState:
    model: LinearModel
    next_seq: int  # sequence number of the next decision
    model_version: int  # observations absorbed so far (bumped by the ledger)
    horizon: float  # seconds an unresolved decision waits before expiring
    default_reward: float  # the reward an expired decision trains with
    ledger: "tuple[DecisionRecord, ...]"  # open decisions awaiting resolution


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    t: float
    candidate_hash: int
    chosen: int
    features: tuple[float, ...]
    propensity: float
    model_version: int
    salt: str


def new_bandit(
    dim: int, half_life: float, t: float, horizon: float, default_reward: float = 0.0
) -> BanditState:
    """`horizon` and `default_reward` are the application's reward-handling
    declaration (design doc section 6), made once, up front: a decision with
    no reward after `horizon` seconds resolves as expired(`default_reward`)."""
    if not (horizon > 0.0):
        raise ValueError("horizon must be positive")
    return BanditState(
        model=new_model(dim, half_life, t),
        next_seq=0,
        model_version=0,
        horizon=horizon,
        default_reward=default_reward,
        ledger=(),
    )


def candidate_set_hash(candidates: list[list[float]]) -> int:
    """64-bit FNV-1a over a canonical encoding of the candidate set: the
    count, then each vector as its length followed by little-endian IEEE-754
    doubles. Order- and value-sensitive, identical on every platform."""
    data = len(candidates).to_bytes(8, "little")
    for x in candidates:
        data += len(x).to_bytes(8, "little")
        for v in x:
            data += struct.pack("<d", v)
    return fnv1a_64(data)


def choose_gamma(uncertainties: list[float]) -> float:
    """The uncertainty-driven greediness schedule (see module docstring)."""
    total = 0.0
    for u in uncertainties:
        total += u
    mean = total / len(uncertainties)
    if mean <= 0.0:  # every candidate a zero vector; nothing to be greedy about
        return 0.0
    return GAMMA_SCALE / mean


def decide(
    state: BanditState, candidates: list[list[float]], t: float, salt: str
) -> tuple[DecisionRecord, BanditState]:
    """Score, explore, choose, and record one decision.

    The model is untouched (learning happens only through the ledger's
    resolutions, see ledger.py); the state changes are the decision counter
    advancing and the record joining the ledger of open decisions.
    """
    if len(candidates) < 1:
        raise ValueError("need at least one candidate")
    for x in candidates:
        if len(x) != state.model.dim:
            raise ValueError("candidate feature length does not match model dim")

    # The ridge system depends on (model, t) only, so one factorization
    # serves every candidate: O(dim^3 + k * dim^2) per decision, not
    # O(k * dim^3).
    factored = factorize(state.model, t)
    estimates = []
    uncertainties = []
    for x in candidates:
        est, unc = predict_factored(factored, x)
        estimates.append(est)
        uncertainties.append(unc)

    gamma = choose_gamma(uncertainties)
    p = apply_floor(inverse_gap_probabilities(estimates, gamma))

    decision_id = f"{salt}:{state.next_seq}"
    key = derive_key(decision_id, salt)
    chosen = sample_index(p, key, 0)

    record = DecisionRecord(
        decision_id=decision_id,
        t=t,
        candidate_hash=candidate_set_hash(candidates),
        chosen=chosen,
        features=tuple(candidates[chosen]),
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
        ledger=state.ledger + (record,),
    )
    return record, new_state
