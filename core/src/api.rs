//! The public API — the port of `api.py`, the dict-shaped facade every
//! binding mirrors. This module is the complete binding
//! surface: PR 20/21 expose exactly these names and nothing else. Every
//! function is the layered API unchanged, with the hashed encoding folded
//! inside `decide` so sparse pairs never cross the public boundary; the
//! facade adds no randomness, no reordering, and no arithmetic, so a
//! decision through it is bit-identical to `encode` + `decide` by hand.

use crate::decide::{decide as decide_encoded, new_bandit, BanditState, DecisionRecord};
use crate::decide::{Explore, Score};
use crate::encoding::{encode, Features, Value};
use crate::error::Error;
use crate::ledger;

// Censor and the resolution types pass through unchanged; re-exported so
// the binding surface is this one module. `learn` and `expire` gain the
// BYO `train` callback below.
pub use crate::ledger::{censor, Kind, Resolution};

/// A BYO train callback: the resolved decision's record and
/// the reward it resolved with — the exactly-once, correctly-joined
/// observation the user's model trains on in place of the built-in update.
pub type Train<'a> = &'a mut dyn FnMut(&DecisionRecord, f64) -> Result<(), Error>;

/// Resolve an open decision at time `t`: rewarded(reward) inside the
/// horizon, expired(default_reward) at or past it — the engine enforces
/// the declared cutoff against the ledger record's own decision time, so
/// a late reward can never train as a timely one. `Ok(None)` (a no-op) if
/// the id is unknown or already resolved. With `train` the built-in model
/// is left untouched: the resolution commits — the record leaves the
/// ledger and model_version advances — and then `train(record, trained)`
/// fires with the same classified reward, exactly once per decision; its
/// error propagates after the commit, so a failing callback loses that
/// one example loudly but can never double-train.
pub fn learn(
    state: &mut BanditState,
    decision_id: &str,
    reward: f64,
    t: f64,
    train: Option<Train>,
) -> Result<Option<Resolution>, Error> {
    let Some(train) = train else {
        return Ok(ledger::learn(state, decision_id, reward, t));
    };
    let Some(i) = state.ledger.iter().position(|r| r.decision_id == decision_id) else {
        return Ok(None);
    };
    let record = state.ledger.remove(i);
    let (kind, trained) = if record.t + state.horizon <= t {
        (Kind::Expired, state.default_reward)
    } else {
        (Kind::Rewarded, reward)
    };
    state.model_version += 1;
    train(&record, trained)?;
    Ok(Some(Resolution {
        decision_id: record.decision_id,
        kind,
        reward: Some(trained),
    }))
}

/// Resolve every open decision past its horizon at time `t` as expired, in
/// ledger order. With `train` each expiry commits and then fires
/// `train(record, default_reward)` — one record at a time, so an error
/// leaves earlier records resolved and later ones open for the next sweep,
/// and no record ever trains twice. The resolutions already
/// made stand when an error is returned.
pub fn expire(
    state: &mut BanditState,
    t: f64,
    train: Option<Train>,
) -> Result<Vec<Resolution>, Error> {
    let Some(train) = train else {
        return Ok(ledger::expire(state, t));
    };
    let mut resolutions = Vec::new();
    let mut i = 0;
    while i < state.ledger.len() {
        if state.ledger[i].t + state.horizon <= t {
            let record = state.ledger.remove(i);
            state.model_version += 1;
            resolutions.push(Resolution {
                decision_id: record.decision_id.clone(),
                kind: Kind::Expired,
                reward: Some(state.default_reward),
            });
            train(&record, state.default_reward)?;
        } else {
            i += 1;
        }
    }
    Ok(resolutions)
}

/// The current state as one plain string: the canonical byte layout
/// (state.rs), hex-encoded, lowercase. A string
/// goes anywhere text goes — a file, a database column, localStorage,
/// version control — so no caller, in any language, ever handles raw
/// bytes or picks an encoding.
pub fn serialize(state: &BanditState) -> String {
    let bytes = crate::state::serialize(state);
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(char::from_digit((b >> 4) as u32, 16).unwrap());
        out.push(char::from_digit((b & 0xf) as u32, 16).unwrap());
    }
    out
}

/// A state parsed and validated from a `serialize` string. Hex digits of
/// either case are accepted; the canonical form `serialize` emits is
/// lowercase.
pub fn deserialize(data: &str) -> Result<BanditState, Error> {
    if data.len() % 2 != 0 {
        return Err(Error::new("state must be a hexadecimal string"));
    }
    let digits = data.as_bytes();
    let mut bytes = Vec::with_capacity(digits.len() / 2);
    for pair in digits.chunks(2) {
        let hi = (pair[0] as char).to_digit(16);
        let lo = (pair[1] as char).to_digit(16);
        match (hi, lo) {
            (Some(hi), Some(lo)) => bytes.push(((hi << 4) | lo) as u8),
            _ => return Err(Error::new("state must be a hexadecimal string")),
        }
    }
    crate::state::deserialize(&bytes)
}

/// A fresh bandit whose model spans the 2^bits hashed feature space —
/// `bits` is the one encoding declaration; every other parameter is
/// `new_bandit`'s (bindings supply the reference's defaults).
pub fn create(
    bits: u32,
    horizon: f64,
    default_reward: f64,
    epsilon: f64,
    forgetfulness: f64,
) -> Result<BanditState, Error> {
    if !(1..=24).contains(&bits) {
        return Err(Error::new("bits must be between 1 and 24"));
    }
    new_bandit(1usize << bits, horizon, default_reward, epsilon, forgetfulness)
}

/// The `bits` declaration recovered from the model dimension. States built
/// by `create` always satisfy dim == 2^bits; anything else was built
/// against the layered API and has no public encoding space.
pub fn model_bits(state: &BanditState) -> Result<u32, Error> {
    let dim = state.model.dim;
    let bits = dim.trailing_zeros();
    if dim.count_ones() != 1 || !(1..=24).contains(&bits) {
        return Err(Error::new(
            "model dimension is not 2**bits for bits in [1, 24]; \
             the state was not built by create()",
        ));
    }
    Ok(bits)
}

/// Score, explore, choose, and record one decision over dict-shaped
/// inputs: `context` is one feature list, each candidate is an
/// (arm_id, action features) pair, encoded here in candidate order.
/// Everything returned and every state change is `decide`'s, unchanged.
/// `score` and `explore` are the BYO callbacks; bindings
/// build them over the caller's own context/candidate objects.
pub fn decide(
    state: &mut BanditState,
    context: &[(String, Value)],
    candidates: &[(String, Vec<(String, Value)>)],
    t: f64,
    salt: &str,
    score: Option<Score>,
    explore: Option<Explore>,
) -> Result<DecisionRecord, Error> {
    let bits = model_bits(state)?;
    let encoded: Vec<Features> = candidates
        .iter()
        .map(|(arm_id, action)| encode(context, arm_id, action, bits))
        .collect::<Result<_, _>>()?;
    decide_encoded(state, &encoded, t, salt, score, explore)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exploration::DEFAULT_EPSILON;
    use crate::model::DEFAULT_FORGETTING;

    fn catalog() -> Vec<(String, Vec<(String, Value)>)> {
        vec![
            ("basic".into(), vec![("price".into(), Value::Num(3.0))]),
            ("plus".into(), vec![("price".into(), Value::Num(9.0))]),
            ("free".into(), vec![]),
        ]
    }

    /// The facade must add nothing: same record, same state, bit for bit,
    /// as encoding by hand and deciding over the sparse pairs.
    #[test]
    fn decide_is_exactly_the_layered_path() {
        let context = vec![("seg".to_string(), Value::Str("a".to_string()))];
        let mut facade = create(4, 10.0, 0.0, DEFAULT_EPSILON, DEFAULT_FORGETTING).unwrap();
        let mut layered = facade.clone();
        for t in [0.0, 1.0, 2.0] {
            let record = decide(&mut facade, &context, &catalog(), t, "s", None, None).unwrap();
            let encoded: Vec<Features> = catalog()
                .iter()
                .map(|(arm, action)| encode(&context, arm, action, 4).unwrap())
                .collect();
            let expected = decide_encoded(&mut layered, &encoded, t, "s", None, None).unwrap();
            assert!(record == expected);
        }
        assert!(facade == layered);
    }

    #[test]
    fn rejects_bits_and_dimensions_outside_the_declaration() {
        for bits in [0, 25] {
            assert!(create(bits, 10.0, 0.0, DEFAULT_EPSILON, DEFAULT_FORGETTING).is_err());
        }
        // dim 5: not a power of two — built against the layered API.
        let mut state = new_bandit(5, 10.0, 0.0, DEFAULT_EPSILON, DEFAULT_FORGETTING).unwrap();
        assert!(decide(&mut state, &[], &catalog(), 0.0, "s", None, None)
            .unwrap_err()
            .message()
            .contains("2**bits"));
    }

    /// The BYO training tap: train replaces the built-in
    /// update, model_version still advances, the callback fires exactly
    /// once per decision, and the commit precedes the callback so a
    /// failing trainer can never cause a double-train.
    #[test]
    fn train_replaces_the_builtin_update_exactly_once() {
        let mut state = create(4, 10.0, 0.0, DEFAULT_EPSILON, DEFAULT_FORGETTING).unwrap();
        let record = decide(&mut state, &[], &catalog(), 0.0, "s", None, None).unwrap();
        let model_before = state.model.clone();
        let trained: std::cell::RefCell<Vec<(String, f64)>> = std::cell::RefCell::new(Vec::new());
        let mut train = |r: &DecisionRecord, reward: f64| {
            trained.borrow_mut().push((r.decision_id.clone(), reward));
            Ok(())
        };
        let resolution = learn(&mut state, &record.decision_id, 0.75, 1.0, Some(&mut train))
            .unwrap()
            .unwrap();
        assert!(resolution.kind == Kind::Rewarded && resolution.reward == Some(0.75));
        assert!(state.model == model_before, "train must not touch the built-in model");
        assert!(state.model_version == 1, "the observation still counts");
        assert!(*trained.borrow() == vec![(record.decision_id.clone(), 0.75)]);
        // Exactly once: a retry is a no-op and the callback stays quiet.
        assert!(learn(&mut state, &record.decision_id, 0.75, 1.0, Some(&mut train))
            .unwrap()
            .is_none());
        assert!(trained.borrow().len() == 1);
    }

    #[test]
    fn expire_with_train_commits_per_record() {
        let mut state = create(4, 5.0, -1.0, DEFAULT_EPSILON, DEFAULT_FORGETTING).unwrap();
        let ids: Vec<String> = (0..3)
            .map(|t| {
                decide(&mut state, &[], &catalog(), t as f64, "s", None, None)
                    .unwrap()
                    .decision_id
            })
            .collect();
        let model_before = state.model.clone();
        // The trainer fails on the second record: the first two commit
        // (the second's example is lost loudly), the third stays open.
        let mut calls: Vec<String> = Vec::new();
        let mut flaky = |r: &DecisionRecord, reward: f64| {
            assert!(reward == -1.0);
            calls.push(r.decision_id.clone());
            if calls.len() == 2 {
                return Err(Error::new("flaky trainer"));
            }
            Ok(())
        };
        assert!(expire(&mut state, 100.0, Some(&mut flaky)).is_err());
        assert!(calls == ids[..2].to_vec());
        assert!(state.model_version == 2);
        let mut tap = |r: &DecisionRecord, _: f64| {
            calls.push(r.decision_id.clone());
            Ok(())
        };
        let rest = expire(&mut state, 100.0, Some(&mut tap)).unwrap();
        assert!(rest.len() == 1 && rest[0].decision_id == ids[2]);
        assert!(state.model == model_before && state.ledger.is_empty());
    }
}
