//! Decision records and the decide layer — the port of `decide.py`.
//!
//! `decide` scores every candidate with one shared lazy factorization,
//! builds the epsilon-greedy distribution, draws with counter 0 of the
//! decision's RNG stream, and returns a self-contained decision record;
//! the state changes are the decision counter advancing and the record
//! joining the ledger of open decisions. Decision IDs are `"{salt}:{seq}"`
//! — uniqueness is structural. This core mutates the state in place; the
//! record returned is a clone of the one the ledger holds.

use crate::encoding::Features;
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
) -> BanditState {
    assert!(horizon > 0.0, "horizon must be positive");
    assert!((0.0..=1.0).contains(&epsilon), "epsilon must be in [0, 1]");
    BanditState {
        model: new_model(dim, forgetting, 1.0),
        next_seq: 0,
        model_version: 0,
        horizon,
        default_reward,
        epsilon,
        ledger: Vec::new(),
    }
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

/// Score, explore, choose, and record one decision. The model is untouched
/// (learning happens only through the ledger's resolutions).
pub fn decide(
    state: &mut BanditState,
    candidates: &[Features],
    t: f64,
    salt: &str,
) -> DecisionRecord {
    assert!(!candidates.is_empty(), "need at least one candidate");
    let dim = state.model.dim;
    for x in candidates {
        let mut prev: i64 = -1;
        for &(j, v) in x {
            assert!(
                prev < j as i64 && j < dim,
                "candidate features must be (index, value) pairs in strictly \
                 increasing index order within the model dimension"
            );
            assert!(v != 0.0, "candidate feature values must be nonzero");
            prev = j as i64;
        }
    }

    // The weights depend on the model only, so they are solved once and
    // shared by every candidate.
    let mut factored = factorize(&state.model);
    let estimates: Vec<f64> = candidates
        .iter()
        .map(|x| estimate_factored(&mut factored, x))
        .collect();
    let p = epsilon_greedy_probabilities(&estimates, state.epsilon);

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
    record
}
