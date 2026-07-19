//! The decision ledger: safe reward handling — the port of `ledger.py`.
//!
//! Every open decision resolves in exactly one of three ways — rewarded,
//! expired(default_reward) at the horizon, or censored — each a deliberate,
//! loggable event. Resolving removes the record, so duplicates are
//! structural no-ops; no code path learns from an open decision. Resolution
//! order within one `expire` sweep is ledger (insertion) order.

use crate::decide::{BanditState, DecisionRecord};
use crate::model::update;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Rewarded,
    Expired,
    Censored,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Rewarded => "rewarded",
            Kind::Expired => "expired",
            Kind::Censored => "censored",
        }
    }
}

/// One deliberate, loggable resolution event. `reward` is the value the
/// model trained with; `None` for censored (excluded from training).
#[derive(Clone, Debug, PartialEq)]
pub struct Resolution {
    pub decision_id: String,
    pub kind: Kind,
    pub reward: Option<f64>,
}

/// The open record with this ID, removed from the ledger; `None` (ledger
/// untouched) if it is unknown or already resolved.
fn take(ledger: &mut Vec<DecisionRecord>, decision_id: &str) -> Option<DecisionRecord> {
    let i = ledger.iter().position(|r| r.decision_id == decision_id)?;
    Some(ledger.remove(i))
}

/// Resolve an open decision as rewarded(reward). Unknown or already
/// resolved IDs are a no-op returning `None`.
pub fn learn(state: &mut BanditState, decision_id: &str, reward: f64) -> Option<Resolution> {
    let record = take(&mut state.ledger, decision_id)?;
    update(&mut state.model, &record.features, reward);
    state.model_version += 1;
    Some(Resolution {
        decision_id: record.decision_id,
        kind: Kind::Rewarded,
        reward: Some(reward),
    })
}

/// Resolve an open decision as censored: removed from the ledger without
/// training, the exclusion itself returned for the log.
pub fn censor(state: &mut BanditState, decision_id: &str) -> Option<Resolution> {
    let record = take(&mut state.ledger, decision_id)?;
    Some(Resolution {
        decision_id: record.decision_id,
        kind: Kind::Censored,
        reward: None,
    })
}

/// Resolve every open decision whose horizon has passed at time `t` as
/// expired(default_reward), in ledger order. Callers sweep this with each
/// event's time; that sweep is what keeps the ledger bounded.
pub fn expire(state: &mut BanditState, t: f64) -> Vec<Resolution> {
    let mut resolutions = Vec::new();
    let mut remaining = Vec::with_capacity(state.ledger.len());
    for record in std::mem::take(&mut state.ledger) {
        if record.t + state.horizon <= t {
            update(&mut state.model, &record.features, state.default_reward);
            resolutions.push(Resolution {
                decision_id: record.decision_id,
                kind: Kind::Expired,
                reward: Some(state.default_reward),
            });
        } else {
            remaining.push(record);
        }
    }
    state.ledger = remaining;
    state.model_version += resolutions.len() as u64;
    resolutions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decide::{decide, new_bandit};
    use crate::exploration::DEFAULT_EPSILON;
    use crate::model::DEFAULT_FORGETTING;

    /// The golden episode resolves each decision exactly once (its
    /// `rejected` attempts cover the no-op paths through the final-state
    /// comparison); this pins the same exactly-once property directly, the
    /// way the reference's pytest suite does: every second resolution —
    /// duplicate, conflicting, post-expiry, post-censor, unknown — returns
    /// None and leaves every bit of the state alone.
    #[test]
    fn resolutions_are_exactly_once() {
        let mut state = new_bandit(4, 10.0, 0.0, DEFAULT_EPSILON, DEFAULT_FORGETTING).unwrap();
        let candidates = vec![vec![(0, 1.0)], vec![(1, 1.0)]];
        let a = decide(&mut state, &candidates, 0.0, "test", None, None).unwrap();
        let b = decide(&mut state, &candidates, 1.0, "test", None, None).unwrap();
        let c = decide(&mut state, &candidates, 2.0, "test", None, None).unwrap();

        assert!(learn(&mut state, &a.decision_id, 1.0).is_some());
        assert!(censor(&mut state, &b.decision_id).is_some());
        let expired = expire(&mut state, 12.0); // c is due at exactly 12.0
        assert!(expired.len() == 1 && expired[0].decision_id == c.decision_id);
        assert!(state.ledger.is_empty() && state.model_version == 2);

        let before = state.clone();
        assert!(learn(&mut state, &a.decision_id, 0.0).is_none()); // conflicting duplicate
        assert!(learn(&mut state, &b.decision_id, 1.0).is_none()); // reward after censor
        assert!(learn(&mut state, &c.decision_id, 1.0).is_none()); // reward after expiry
        assert!(censor(&mut state, &a.decision_id).is_none()); // censor after reward
        assert!(learn(&mut state, "never-made:0", 1.0).is_none()); // unknown id
        assert!(expire(&mut state, 100.0).is_empty()); // nothing due
        assert!(state == before, "a rejected resolution moved the state");
    }
}
