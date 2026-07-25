//! The comparator policies — the port of `sim/policies.py`. `gittins` is
//! the engine on the real public path (encode -> decide -> ledger); the
//! baselines share its model and encoding and differ only in exploration.

use gittins_core::decide::{decide, new_bandit, BanditState};
use gittins_core::encoding::{encode, Features};
use gittins_core::exploration::DEFAULT_EPSILON;
use gittins_core::ledger::{expire, learn};
use gittins_core::model::{factorize, new_model, predict_factored, update, LinearModel, DEFAULT_FORGETTING};

use crate::environments::Round;
use crate::rand::{randint, stream, uniform};

pub trait Policy {
    fn name(&self) -> &str;
    /// Reset all run state; called once before each run.
    fn begin(&mut self, seed: u64);
    /// (chosen candidate index, per-candidate reward estimates or None).
    fn choose(&mut self, rd: &Round, seed: u64) -> (usize, Option<Vec<f64>>);
    /// The reward for the most recent `choose`; no-op for model-free policies.
    fn observe(&mut self, _reward: f64) {}
}

/// First maximum — the same tie-breaking rule as the reference's argmax.
fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    best
}

/// Upper bound; the only policy allowed to read Round.means.
pub struct OraclePolicy;

impl Policy for OraclePolicy {
    fn name(&self) -> &str {
        "oracle"
    }
    fn begin(&mut self, _seed: u64) {}
    fn choose(&mut self, rd: &Round, _seed: u64) -> (usize, Option<Vec<f64>>) {
        (argmax(&rd.means), None)
    }
}

/// Lower bound: uniform over the candidates, no model.
pub struct UniformPolicy;

impl Policy for UniformPolicy {
    fn name(&self) -> &str {
        "uniform"
    }
    fn begin(&mut self, _seed: u64) {}
    fn choose(&mut self, rd: &Round, seed: u64) -> (usize, Option<Vec<f64>>) {
        let key = stream("uniform", seed, &format!("choose:{:?}", rd.t));
        (randint(key, 0, rd.arm_ids.len()), None)
    }
}

/// Shared machinery for the model-based baselines: the same per-coordinate
/// forgetting ridge on the same hashed encoding as the engine.
struct ModelCore {
    bits: u32,
    model: LinearModel,
    chosen_x: Features,
}

impl ModelCore {
    fn new(bits: u32) -> ModelCore {
        ModelCore {
            bits,
            model: new_model(1 << bits, DEFAULT_FORGETTING, 1.0).unwrap(),
            chosen_x: Vec::new(),
        }
    }

    fn begin(&mut self) {
        self.model = new_model(1 << self.bits, DEFAULT_FORGETTING, 1.0).unwrap();
    }

    /// (sparse candidates, their estimates) for one round.
    fn estimates(&self, rd: &Round) -> (Vec<Features>, Vec<f64>) {
        let candidates: Vec<Features> = (0..rd.arm_ids.len())
            .map(|i| encode(&rd.context, &rd.arm_ids[i], &rd.actions[i], self.bits).unwrap())
            .collect();
        let mut f = factorize(&self.model);
        let estimates = candidates.iter().map(|x| predict_factored(&mut f, x).0).collect();
        (candidates, estimates)
    }

    fn observe(&mut self, reward: f64) {
        update(&mut self.model, &self.chosen_x, reward);
    }
}

/// epsilon = 0, first-max ties: always the argmax estimate.
pub struct GreedyPolicy {
    core: ModelCore,
}

impl GreedyPolicy {
    pub fn new(bits: u32) -> Self {
        GreedyPolicy { core: ModelCore::new(bits) }
    }
}

impl Policy for GreedyPolicy {
    fn name(&self) -> &str {
        "greedy"
    }
    fn begin(&mut self, _seed: u64) {
        self.core.begin();
    }
    fn choose(&mut self, rd: &Round, _seed: u64) -> (usize, Option<Vec<f64>>) {
        let (candidates, estimates) = self.core.estimates(rd);
        let chosen = argmax(&estimates);
        self.core.chosen_x = candidates.into_iter().nth(chosen).unwrap();
        (chosen, Some(estimates))
    }
    fn observe(&mut self, reward: f64) {
        self.core.observe(reward);
    }
}

/// Argmax estimate, except a uniform candidate with probability eps.
pub struct EpsilonGreedyPolicy {
    name: String,
    eps: f64,
    core: ModelCore,
}

impl EpsilonGreedyPolicy {
    pub fn new(eps: f64, bits: u32) -> Self {
        EpsilonGreedyPolicy {
            name: format!("epsilon-{eps}"),
            eps,
            core: ModelCore::new(bits),
        }
    }
}

impl Policy for EpsilonGreedyPolicy {
    fn name(&self) -> &str {
        &self.name
    }
    fn begin(&mut self, _seed: u64) {
        self.core.begin();
    }
    fn choose(&mut self, rd: &Round, seed: u64) -> (usize, Option<Vec<f64>>) {
        let (candidates, estimates) = self.core.estimates(rd);
        let key = stream(&self.name, seed, &format!("choose:{:?}", rd.t));
        let chosen = if uniform(key, 0, 0.0, 1.0) < self.eps {
            randint(key, 1, candidates.len())
        } else {
            argmax(&estimates)
        };
        self.core.chosen_x = candidates.into_iter().nth(chosen).unwrap();
        (chosen, Some(estimates))
    }
    fn observe(&mut self, reward: f64) {
        self.core.observe(reward);
    }
}

/// The engine, on the real public path: rewards go to the ledger's `learn`
/// by decision id, and the `expire` sweep runs with every round's time, as
/// a real event loop would.
pub struct GittinsPolicy {
    bits: u32,
    horizon: f64,
    state: BanditState,
    salt: String,
    decision_id: String,
    t: f64,
}

impl GittinsPolicy {
    pub fn new(bits: u32) -> Self {
        GittinsPolicy {
            bits,
            horizon: 10.0,
            state: new_bandit(1 << bits, 10.0, 0.0, DEFAULT_EPSILON, DEFAULT_FORGETTING).unwrap(),
            salt: String::new(),
            decision_id: String::new(),
            t: 0.0,
        }
    }
}

impl Policy for GittinsPolicy {
    fn name(&self) -> &str {
        "gittins"
    }
    fn begin(&mut self, seed: u64) {
        self.state = new_bandit(
            1 << self.bits,
            self.horizon,
            0.0,
            DEFAULT_EPSILON,
            DEFAULT_FORGETTING,
        )
        .unwrap();
        self.salt = format!("gittins:{seed}");
    }
    fn choose(&mut self, rd: &Round, _seed: u64) -> (usize, Option<Vec<f64>>) {
        expire(&mut self.state, rd.t);
        let candidates: Vec<Features> = (0..rd.arm_ids.len())
            .map(|i| encode(&rd.context, &rd.arm_ids[i], &rd.actions[i], self.bits).unwrap())
            .collect();
        let record = decide(&mut self.state, &candidates, rd.t, &self.salt, None, None).unwrap();
        self.decision_id = record.decision_id;
        self.t = rd.t;
        // Metric-only read: the same estimates decide just scored with,
        // recomputed because decide deliberately logs only the chosen
        // candidate.
        let mut f = factorize(&self.state.model);
        let estimates = candidates.iter().map(|x| predict_factored(&mut f, x).0).collect();
        (record.chosen, Some(estimates))
    }
    fn observe(&mut self, reward: f64) {
        learn(&mut self.state, &self.decision_id, reward, self.t);
    }
}
