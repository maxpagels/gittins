//! Epsilon-greedy exploration with exact-tie splitting — the port of
//! `exploration.py`. `epsilon` of the probability mass is spent uniformly
//! over all k candidates; the remaining `1 - epsilon` splits evenly across
//! the candidates tied (in exact float equality) for the maximal estimate.
//! Sampling is inverse-CDF over one counter-RNG uniform, in fixed index
//! order, so the distribution and the choice are bit-identical everywhere.

use crate::rng::random_unit;

/// Probability mass spent uniformly over the candidates; every candidate is
/// guaranteed at least DEFAULT_EPSILON / k. An expert override at
/// construction, not a knob to routinely tune.
pub const DEFAULT_EPSILON: f64 = 0.05;

/// The epsilon-greedy distribution over candidates: `epsilon` uniform,
/// `1 - epsilon` split across the maximal estimates (exact-tie split).
pub fn epsilon_greedy_probabilities(estimates: &[f64], epsilon: f64) -> Vec<f64> {
    let k = estimates.len();
    assert!(k >= 1, "need at least one candidate");
    assert!((0.0..=1.0).contains(&epsilon), "epsilon must be in [0, 1]");
    let mut best = estimates[0];
    for &e in &estimates[1..] {
        if e > best {
            best = e;
        }
    }
    let m = estimates.iter().filter(|&&e| e == best).count();
    let uniform = epsilon / k as f64;
    let share = (1.0 - epsilon) / m as f64;
    estimates
        .iter()
        .map(|&e| if e == best { uniform + share } else { uniform })
        .collect()
}

/// Draw one index from distribution `p` using the uniform value at position
/// `counter` of RNG stream `key` (inverse CDF, fixed index order).
pub fn sample_index(p: &[f64], key: u64, counter: u64) -> usize {
    let u = random_unit(key, counter);
    let mut cumulative = 0.0;
    for (i, &pi) in p.iter().enumerate() {
        cumulative += pi;
        if u < cumulative {
            return i;
        }
    }
    p.len() - 1 // u landed in the rounding gap above the final sum
}
