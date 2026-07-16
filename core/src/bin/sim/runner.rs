//! The runner: one policy through one environment for one seed — the port
//! of `sim/runner.py`. Rounds are pure functions of (environment, seed, t),
//! so every policy run on the same seed faces identical contexts,
//! candidates, and noise draws — comparisons are paired.

use crate::environments::Environment;
use crate::policies::Policy;

/// Per-round series computed from the oracle means, so regret is exact.
/// The Python RunResult also keeps the per-round best means and realized
/// rewards; they only feed `recovery_time`, which the battery driver never
/// calls, so this port drops them.
pub struct RunResult {
    pub regret: Vec<f64>,
    pub normalizer: Vec<f64>,
    pub sq_error: Option<Vec<f64>>,
}

pub fn run(env: &dyn Environment, policy: &mut dyn Policy, seed: u64, rounds: usize) -> RunResult {
    assert!(rounds >= 1, "need at least one round");
    policy.begin(seed);
    let mut regret = Vec::with_capacity(rounds);
    let mut normalizer = Vec::with_capacity(rounds);
    let mut sq_error: Option<Vec<f64>> = None;
    for t in 0..rounds {
        let rd = env.round(seed, t);
        let (chosen, estimates) = policy.choose(&rd, seed);
        let r = env.reward(seed, t, &rd, chosen);
        policy.observe(r);

        let k = rd.means.len();
        let best = rd.means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        regret.push(best - rd.means[chosen]);
        normalizer.push(best - rd.means.iter().sum::<f64>() / k as f64);
        if let Some(estimates) = estimates {
            let series = sq_error.get_or_insert_with(Vec::new);
            let mut total = 0.0;
            for i in 0..k {
                total += (estimates[i] - rd.means[i]) * (estimates[i] - rd.means[i]);
            }
            series.push(total / k as f64);
        }
    }
    RunResult {
        regret,
        normalizer,
        sq_error,
    }
}
