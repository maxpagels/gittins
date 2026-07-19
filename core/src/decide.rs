//! Decision records and the decide layer — the port of `decide.py`.
//!
//! `decide` scores every candidate with one shared lazy factorization,
//! builds the epsilon-greedy distribution, draws with counter 0 of the
//! decision's RNG stream, and returns a self-contained decision record;
//! the state changes are the decision counter advancing and the record
//! joining the ledger of open decisions. Decision IDs are `"{salt}:{seq}"`
//! — uniqueness is structural. This core mutates the state in place; the
//! record returned is a clone of the one the ledger holds.
//!
//! Bring your own model / exploration (R7, spec/byo.md): `decide` takes
//! two optional callbacks, each replacing exactly one built-in component —
//! `score(candidates)` the built-in estimates, `explore(estimates,
//! epsilon)` epsilon-greedy. Outputs are validated here (the reference's
//! rules and messages, exactly); the draw, the record, the propensity, and
//! the ledger are the shared path, so a BYO decision is as deterministic
//! and replayable as a built-in one. A callback error propagates before
//! any state change.

use crate::encoding::Features;
use crate::error::Error;
use crate::exploration::{epsilon_greedy_probabilities, sample_index};
use crate::model::{estimate_factored, factorize, new_model, LinearModel};
use crate::rng::{derive_key, fnv1a_64};

#[derive(Clone, Debug, PartialEq)]
pub struct BanditState {
    pub model: LinearModel,
    pub next_seq: u64,       // sequence number of the next decision
    pub model_version: u64,  // observations absorbed so far (bumped by the ledger)
    pub horizon: f64,        // seconds an unresolved decision waits before expiring
    pub default_reward: f64, // the reward an expired decision trains with
    pub epsilon: f64,        // probability mass spent uniformly per decision
    pub ledger: Vec<DecisionRecord>, // open decisions awaiting resolution
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionRecord {
    pub decision_id: String,
    pub t: f64,
    pub candidate_hash: u64,
    pub chosen: usize,
    pub features: Features, // the chosen candidate's sparse pairs
    pub propensity: f64,
    pub model_version: u64,
    pub salt: String,
}

/// `horizon` and `default_reward` are the application's reward-handling
/// declaration; `epsilon` and `forgetting` are expert overrides
/// (`DEFAULT_EPSILON`, `DEFAULT_FORGETTING`), not default-API knobs.
pub fn new_bandit(
    dim: usize,
    horizon: f64,
    default_reward: f64,
    epsilon: f64,
    forgetting: f64,
) -> Result<BanditState, Error> {
    if !(horizon > 0.0) {
        return Err(Error::new("horizon must be positive"));
    }
    if !(0.0..=1.0).contains(&epsilon) {
        return Err(Error::new("epsilon must be in [0, 1]"));
    }
    Ok(BanditState {
        model: new_model(dim, forgetting, 1.0)?,
        next_seq: 0,
        model_version: 0,
        horizon,
        default_reward,
        epsilon,
        ledger: Vec::new(),
    })
}

/// 64-bit FNV-1a over the canonical sparse encoding of the candidate set:
/// the count, then each candidate as the model dimension, its nonzero-entry
/// count, and its (index, value) pairs in index order — integers as 8
/// little-endian bytes, values as little-endian IEEE-754 doubles.
pub fn candidate_set_hash(candidates: &[Features], dim: usize) -> u64 {
    let mut data = Vec::new();
    data.extend_from_slice(&(candidates.len() as u64).to_le_bytes());
    for x in candidates {
        data.extend_from_slice(&(dim as u64).to_le_bytes());
        data.extend_from_slice(&(x.len() as u64).to_le_bytes());
        for &(i, v) in x {
            data.extend_from_slice(&(i as u64).to_le_bytes());
            data.extend_from_slice(&v.to_le_bytes());
        }
    }
    fnv1a_64(&data)
}

/// A BYO score callback: the candidates' estimates, in candidate order
/// (spec/byo.md). The `Err` side carries a binding's failure across the
/// boundary; validation of the `Ok` side happens in `decide`.
pub type Score<'a> = &'a mut dyn FnMut(&[Features]) -> Result<Vec<f64>, Error>;

/// A BYO explore callback: (estimates, epsilon) to one probability per
/// candidate; the engine draws from it (spec/byo.md).
pub type Explore<'a> = &'a mut dyn FnMut(&[f64], f64) -> Result<Vec<f64>, Error>;

/// How far a BYO explore distribution's sum may sit from 1.0 before it is
/// rejected: generous against benign rounding, unforgiving of real errors
/// — a wrong sum poisons every logged propensity (spec/byo.md).
pub const PROBABILITY_TOLERANCE: f64 = 1e-9;

fn validated_estimates(estimates: Vec<f64>, k: usize) -> Result<Vec<f64>, Error> {
    if estimates.len() != k || estimates.iter().any(|v| !v.is_finite()) {
        return Err(Error::new("score must return one finite estimate per candidate"));
    }
    Ok(estimates)
}

fn validated_probabilities(p: Vec<f64>, k: usize) -> Result<Vec<f64>, Error> {
    if p.len() != k {
        return Err(Error::new("explore must return one probability per candidate"));
    }
    let mut total = 0.0;
    for &v in &p {
        if !(v.is_finite() && v >= 0.0) {
            return Err(Error::new(
                "explore probabilities must be finite, nonnegative, and sum to 1",
            ));
        }
        total += v;
    }
    if !((total - 1.0).abs() <= PROBABILITY_TOLERANCE) {
        return Err(Error::new(
            "explore probabilities must be finite, nonnegative, and sum to 1",
        ));
    }
    Ok(p)
}

/// Score, explore, choose, and record one decision. The model is untouched
/// (learning happens only through the ledger's resolutions). `score` and
/// `explore` are the BYO callbacks (module docs, spec/byo.md).
pub fn decide(
    state: &mut BanditState,
    candidates: &[Features],
    t: f64,
    salt: &str,
    score: Option<Score>,
    explore: Option<Explore>,
) -> Result<DecisionRecord, Error> {
    if candidates.is_empty() {
        return Err(Error::new("need at least one candidate"));
    }
    let dim = state.model.dim;
    for x in candidates {
        let mut prev: i64 = -1;
        for &(j, v) in x {
            if !(prev < j as i64 && j < dim) {
                return Err(Error::new(
                    "candidate features must be (index, value) pairs in strictly \
                     increasing index order within the model dimension",
                ));
            }
            if v == 0.0 {
                return Err(Error::new("candidate feature values must be nonzero"));
            }
            prev = j as i64;
        }
    }

    let estimates: Vec<f64> = match score {
        None => {
            // The weights depend on the model only, so they are solved
            // once and shared by every candidate.
            let mut factored = factorize(&state.model);
            candidates
                .iter()
                .map(|x| estimate_factored(&mut factored, x))
                .collect()
        }
        Some(score) => validated_estimates(score(candidates)?, candidates.len())?,
    };
    let p = match explore {
        None => epsilon_greedy_probabilities(&estimates, state.epsilon)?,
        Some(explore) => {
            validated_probabilities(explore(&estimates, state.epsilon)?, candidates.len())?
        }
    };

    let decision_id = format!("{salt}:{}", state.next_seq);
    let key = derive_key(&decision_id, salt);
    let chosen = sample_index(&p, key, 0);

    let record = DecisionRecord {
        decision_id,
        t,
        candidate_hash: candidate_set_hash(candidates, dim),
        chosen,
        features: candidates[chosen].clone(),
        propensity: p[chosen],
        model_version: state.model_version,
        salt: salt.to_string(),
    };
    state.next_seq += 1;
    state.ledger.push(record.clone());
    Ok(record)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exploration::DEFAULT_EPSILON;
    use crate::model::DEFAULT_FORGETTING;

    fn state() -> BanditState {
        new_bandit(4, 10.0, 0.0, DEFAULT_EPSILON, DEFAULT_FORGETTING).unwrap()
    }

    fn cands() -> Vec<Features> {
        vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(2, 1.0)]]
    }

    /// A BYO explore callback that computes exactly the built-in
    /// distribution must yield the identical record and state: the draw,
    /// the record, and the ledger append are one shared path.
    #[test]
    fn byo_explore_reproducing_the_builtin_rule_is_bit_identical() {
        let mut builtin = state();
        let mut byo = state();
        let builtin_record = decide(&mut builtin, &cands(), 0.0, "s", None, None).unwrap();
        let mut explore = |estimates: &[f64], epsilon: f64| {
            epsilon_greedy_probabilities(estimates, epsilon)
        };
        let byo_record =
            decide(&mut byo, &cands(), 0.0, "s", None, Some(&mut explore)).unwrap();
        assert!(byo_record == builtin_record);
        assert!(byo == builtin);
    }

    #[test]
    fn byo_score_and_explore_flow_into_the_record() {
        let mut s = state();
        let mut score = |candidates: &[Features]| {
            Ok((0..candidates.len()).map(|i| i as f64).collect())
        };
        let mut explore = |_: &[f64], _: f64| Ok(vec![0.25, 0.25, 0.5]);
        let record =
            decide(&mut s, &cands(), 0.0, "s", Some(&mut score), Some(&mut explore)).unwrap();
        assert!(record.propensity == [0.25, 0.25, 0.5][record.chosen]);
        // Score alone: the greedy mass lands on the callback's maximum.
        let record = decide(&mut s, &cands(), 1.0, "s", Some(&mut score), None).unwrap();
        assert!(record.chosen == 2 || record.propensity == DEFAULT_EPSILON / 3.0);
    }

    #[test]
    fn byo_outputs_are_validated_and_leave_the_state_unchanged() {
        let mut s = state();
        let before = s.clone();
        let cases: Vec<(Vec<f64>, &str)> = vec![
            (vec![0.0, 0.0], "score must return one finite estimate per candidate"),
            (vec![0.0, 0.0, f64::NAN], "score must return one finite estimate per candidate"),
        ];
        for (bad, message) in cases {
            let mut score = |_: &[Features]| Ok(bad.clone());
            let err = decide(&mut s, &cands(), 0.0, "s", Some(&mut score), None).unwrap_err();
            assert!(err.message() == message, "got {:?}", err.message());
        }
        let cases: Vec<(Vec<f64>, &str)> = vec![
            (vec![0.5, 0.5], "explore must return one probability per candidate"),
            (vec![0.6, 0.5, -0.1], "explore probabilities must be finite, nonnegative, and sum to 1"),
            (vec![0.5, 0.5, f64::NAN], "explore probabilities must be finite, nonnegative, and sum to 1"),
            (vec![0.3, 0.3, 0.3], "explore probabilities must be finite, nonnegative, and sum to 1"),
        ];
        for (bad, message) in cases {
            let mut explore = |_: &[f64], _: f64| Ok(bad.clone());
            let err = decide(&mut s, &cands(), 0.0, "s", None, Some(&mut explore)).unwrap_err();
            assert!(err.message() == message, "got {:?}", err.message());
        }
        // A callback error propagates before any state change.
        let mut boom = |_: &[Features]| Err(Error::new("model server down"));
        assert!(decide(&mut s, &cands(), 0.0, "s", Some(&mut boom), None).is_err());
        assert!(s == before, "a rejected BYO decision moved the state");
        // Within tolerance passes.
        let mut close = |_: &[f64], _: f64| Ok(vec![0.1, 0.2, 0.7 + 1e-10]);
        assert!(decide(&mut s, &cands(), 0.0, "s", None, Some(&mut close)).is_ok());
    }
}
