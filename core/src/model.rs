//! Per-coordinate ridge regression with per-update exponential forgetting —
//! the port of `model.py`, pre-scaled sums and all. See the reference's
//! module docstring for the full derivation; every arithmetic expression
//! here keeps its exact order. The one deliberate difference: this core
//! mutates the sums in place (the reference copies immutable tuples), which
//! changes no bit of any value.

use std::collections::HashMap;

use crate::encoding::Features;

pub const DEFAULT_FORGETTING: f64 = 0.999;

/// Renormalize the pre-scaled sums when the scale falls this low (2^-512):
/// 1/scale stays <= 2^512, keeping every pre-scaled entry far from double
/// overflow for roughly-unit-scale features.
const RENORM_THRESHOLD: f64 = f64::from_bits(511 << 52); // 2.0^-512 exactly

#[derive(Clone, Debug)]
pub struct LinearModel {
    pub dim: usize,
    pub ridge: f64,
    pub forgetting: f64, // per-update geometric discount on the true sums
    pub scale: f64,      // movable origin: true sums are scale * xx, scale * xy
    pub xx: Vec<f64>,    // dim pre-scaled sums of x_j^2 (the diagonal)
    pub xy: Vec<f64>,    // dim pre-scaled sums of reward * x_j
}

pub fn new_model(dim: usize, forgetting: f64, ridge: f64) -> LinearModel {
    assert!(dim >= 1, "dim must be at least 1");
    assert!(
        0.0 < forgetting && forgetting <= 1.0,
        "forgetting must be in (0, 1] (1.0 = never forget)"
    );
    assert!(ridge > 0.0, "ridge must be positive");
    LinearModel {
        dim,
        ridge,
        forgetting,
        scale: 1.0,
        xx: vec![0.0; dim],
        xy: vec![0.0; dim],
    }
}

/// Absorb one observation: fold the forgetting discount into the scale, add
/// the observation's pre-scaled contribution to its touched coordinates
/// only. O(nonzeros), in place.
pub fn update(model: &mut LinearModel, x: &Features, reward: f64) {
    let scale = model.forgetting * model.scale;
    let inv = 1.0 / scale;
    for &(j, v) in x {
        model.xx[j] += (v * v) * inv;
        model.xy[j] += (reward * v) * inv;
    }
    if scale <= RENORM_THRESHOLD {
        for u in model.xx.iter_mut() {
            *u = scale * *u;
        }
        for u in model.xy.iter_mut() {
            *u = scale * *u;
        }
        model.scale = 1.0;
    } else {
        model.scale = scale;
    }
}

/// The candidate-independent part of prediction, solved lazily: a
/// coordinate's (1/a_j, theta_j) is computed the first time any candidate
/// touches it and memoized for the rest of the decision. Valid until the
/// next update.
pub struct Factorization<'a> {
    model: &'a LinearModel,
    cache: HashMap<usize, (f64, f64)>,
}

impl<'a> Factorization<'a> {
    /// (1 / a_j, theta_j) for one coordinate, memoized.
    pub fn coordinate(&mut self, j: usize) -> (f64, f64) {
        if let Some(&got) = self.cache.get(&j) {
            return got;
        }
        let m = self.model;
        let inv_a = 1.0 / (m.scale * m.xx[j] + m.ridge);
        let got = (inv_a, (m.scale * m.xy[j]) * inv_a);
        self.cache.insert(j, got);
        got
    }
}

/// The lazy solve: O(1) now, one reciprocal per touched coordinate later.
pub fn factorize(model: &LinearModel) -> Factorization<'_> {
    Factorization {
        model,
        cache: HashMap::new(),
    }
}

/// Estimated reward only, for callers that need no uncertainty (the decide
/// layer): one multiply-add per nonzero and no sqrt.
pub fn estimate_factored(f: &mut Factorization, x: &Features) -> f64 {
    let mut estimate = 0.0;
    for &(j, v) in x {
        estimate += v * f.coordinate(j).1;
    }
    estimate
}

/// (estimated reward, uncertainty) for features x, given a factorization
/// built from the same model state. O(nonzeros).
pub fn predict_factored(f: &mut Factorization, x: &Features) -> (f64, f64) {
    let mut estimate = 0.0;
    let mut variance = 0.0;
    for &(j, v) in x {
        let (inv_a, theta) = f.coordinate(j);
        estimate += v * theta;
        variance += (v * v) * inv_a;
    }
    (estimate, variance.sqrt())
}

/// (estimated reward, uncertainty) for features x.
pub fn predict(model: &LinearModel, x: &Features) -> (f64, f64) {
    predict_factored(&mut factorize(model), x)
}
