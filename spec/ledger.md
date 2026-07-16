# Spec: the decision ledger

Design doc reference: section 6 (delayed, missing, and duplicate rewards),
R4 (safe reward handling), D5 (learn accepts only a decision record plus an
outcome). Implemented in `src/gittins_reference/ledger.py`, on top
of the decide layer (`spec/decide.md`) and the model (`spec/model.md`).

## State

The `BanditState.ledger` is the tuple of *open* decision records — made by
`decide`, not yet resolved — in decision (insertion) order. Two companion
fields form the application's reward-handling declaration, made once at
`new_bandit`:

- `horizon` (seconds, > 0) — how long an unresolved decision waits before
  it expires.
- `default_reward` — the reward an expired decision trains with (e.g.
  "no click = 0"). Default 0.0.

## Resolutions

Every open decision resolves in exactly one of three ways. Each resolver
returns a `Resolution(decision_id, kind, reward)` — a deliberate, loggable
event — alongside the new state; `reward` is the value trained with, or
None for censored.

**learn(state, decision_id, reward) → rewarded.** Removes the record from
the ledger, trains the model on `(record.features, reward)` — one ordinary
model update at the resolution's position in the update sequence — and
bumps `model_version`.

**expire(state, t) → expired, for every due decision.** A decision is due
when `record.t + horizon <= t`. Each due record trains with
`default_reward` and is removed; resolutions happen in ledger order,
fixed. Callers sweep `expire` with each event's time; that sweep is what
keeps the ledger bounded to one horizon's worth of open decisions.

**censor(state, decision_id) → censored.** Removes the record *without*
training and without bumping `model_version`; the returned resolution is
the on-record exclusion that offline evaluation needs (R3).

`learn` and `censor` against an ID not in the ledger — never made, already
resolved, or expired — return `(None, state)` unchanged.

## The three properties, by construction

- **Idempotent.** Resolving removes the record; there is one record per
  decision and it can be spent once. A duplicate report — same value,
  different value, or entirely bogus — finds nothing and is a no-op.
- **Late rewards are safe, in arrival order.** Rewards may arrive late,
  out of order, more than once, or never; every case resolves through the
  same three paths with nothing lost or double-counted. Training applies
  in resolution order with the model's current forgetting weight
  (`spec/model.md` — the model has no notion of time), so a late reward
  counts slightly less than an on-time one would have. Replaying a model
  therefore requires the ordered resolution sequence, which the decision
  log provides.
- **The silent bug is unrepresentable.** "Reward hasn't arrived yet"
  cannot be read as "reward was zero": an open decision is not training
  data, and no code path learns from one. Absence becomes the declared
  default only through `expire` — an explicit event at the declared
  horizon. A reward arriving *after* its decision expired is ignored: the
  horizon is the application's declared cutoff, not a tunable.

Combined with D5 (there is no API taking a hand-assembled
(features, reward) pair in the online path), constructing invalid training
data is impossible by design (R4).

## Golden vectors

Bandit dim 2, default forgetting (0.999), horizon 86400 s, default reward
0.25, T0 = 1752000000.0. Decisions over `[[1,0], [0,1]]` at T0, T0+600,
T0+1200 with salt `"pepper"` choose arms (0, 0, 1). Then
`learn("pepper:1", 1.0)` and `expire` at T0+600+86400:

    expire resolves  ("pepper:0", expired, 0.25)   only
    ledger holds     ["pepper:2"]
    model_version    2
    predict([1,-1]) = (0.4164721573857953, 1.1547486659415682)
