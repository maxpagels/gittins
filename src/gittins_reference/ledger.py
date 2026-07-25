"""The decision ledger: safe reward handling (design doc section 6).

The engine owns the join between decisions and rewards. The state's ledger
holds every *open* decision record — made by `decide`, not yet resolved —
and the only way to train the model is to resolve one of them. Every open
decision resolves in exactly one of two ways, each a deliberate event
returned to the caller for logging:

    learn(state, decision_id, reward, t)  ->  rewarded(reward), or
                                              expired(default_reward) when t
                                              says the horizon already passed
    expire(state, t)                      ->  expired(default_reward), for every
                                              decision past its horizon

Three properties, by construction rather than by discipline:

- **Idempotent** — resolving removes the record from the ledger, so a
  duplicate (or conflicting, or bogus) report finds nothing and is a no-op.
  There is one record per decision and it can be spent once.
- **Late rewards are safe, in arrival order** — rewards may arrive late,
  out of order, or never, and every case resolves through the same two
  paths with nothing lost or double-counted. Training applies at the
  *resolution's* position in the update sequence with the model's current
  forgetting weight (model.py has no notion of time), so a late reward
  counts slightly less than an on-time one would have — the model is
  simply trained in the order the world reported outcomes. Replaying a
  model therefore requires the ordered resolution sequence, which the
  decision log provides.
- **The horizon is enforced by the engine, not the sweep** — `learn`
  receives the caller's time and checks it against the open record's own
  decision time: a reward arriving at or past `record.t + horizon` is not
  a reward at all but an expiry that hadn't been swept yet, and it trains
  as `default_reward` exactly as `expire` would have. The declared cutoff
  therefore holds even if the caller's sweep cadence lags.
- **The classic silent bug is unrepresentable** — "reward hasn't arrived
  yet" cannot be read as "reward was zero", because an open decision is not
  training data and no code path learns from one. Absence becomes a zero (or
  whatever the application declared) only through `expire`, an explicit
  event at a declared horizon.

The ledger is bounded by the horizon: a decision is due `horizon` seconds
after it was made, so under a regular `expire` sweep (the caller's event
loop passes each event's time) the ledger never holds more than one
horizon's worth of open decisions. A reward that arrives after its decision
expired finds the ledger empty and is ignored — the horizon is the
application's declared cutoff, not a tunable.

Resolution order within one `expire` sweep is ledger (insertion) order,
fixed, so a sweep is bit-reproducible like everything else.
"""

from dataclasses import dataclass, replace

from gittins_reference.decide import BanditState, DecisionRecord
from gittins_reference.model import update

REWARDED = "rewarded"
EXPIRED = "expired"


@dataclass(frozen=True)
class Resolution:
    """One deliberate, loggable resolution event. `reward` is the value the
    model trained with."""

    decision_id: str
    kind: str  # REWARDED | EXPIRED
    reward: float


def take(
    ledger: "tuple[DecisionRecord, ...]", decision_id: str
) -> "tuple[DecisionRecord | None, tuple[DecisionRecord, ...]]":
    """The open record with this ID (or None) and the ledger without it."""
    for i in range(len(ledger)):
        if ledger[i].decision_id == decision_id:
            return ledger[i], ledger[:i] + ledger[i + 1 :]
    return None, ledger


def learn(
    state: BanditState, decision_id: str, reward: float, t: float
) -> "tuple[Resolution | None, BanditState]":
    """Resolve an open decision at time `t`: rewarded(reward) inside the
    horizon, expired(default_reward) at or past it — the same boundary
    `expire` uses, so a late reward can never train as a timely one.
    Unknown or already resolved IDs are a no-op: (None, unchanged state)."""
    record, rest = take(state.ledger, decision_id)
    if record is None:
        return None, state
    if record.t + state.horizon <= t:
        kind, trained = EXPIRED, state.default_reward
    else:
        kind, trained = REWARDED, reward
    new_state = replace(
        state,
        model=update(state.model, record.features, trained),
        model_version=state.model_version + 1,
        ledger=rest,
    )
    return Resolution(decision_id, kind, trained), new_state


def expire(state: BanditState, t: float) -> "tuple[tuple[Resolution, ...], BanditState]":
    """Resolve every open decision whose horizon has passed at time `t` as
    expired(default_reward), in ledger order. Callers sweep this with each
    event's time; that sweep is what keeps the ledger bounded."""
    resolutions = []
    remaining = []
    model = state.model
    for record in state.ledger:
        if record.t + state.horizon <= t:
            model = update(model, record.features, state.default_reward)
            resolutions.append(Resolution(record.decision_id, EXPIRED, state.default_reward))
        else:
            remaining.append(record)
    if not resolutions:
        return (), state
    new_state = replace(
        state,
        model=model,
        model_version=state.model_version + len(resolutions),
        ledger=tuple(remaining),
    )
    return tuple(resolutions), new_state
