//! Per-coordinate ridge regression with per-update exponential forgetting —
//! the port of `model.py`, pre-scaled sums and all. See the reference's
//! module docstring for the full derivation; every arithmetic expression
//! here keeps its exact order. The one deliberate difference: this core
//! mutates the sums in place (the reference copies immutable tuples), which
//! changes no bit of any value.

use crate::encoding::Features;
use crate::error::Error;

pub const DEFAULT_FORGETTING: f64 = 0.999;

/// Renormalize the pre-scaled sums when the scale falls this low (2^-512):
/// 1/scale stays <= 2^512, keeping every pre-scaled entry far from double
/// overflow for roughly-unit-scale features.
const RENORM_THRESHOLD: f64 = f64::from_bits(511 << 52); // 2.0^-512 exactly

#[derive(Clone, Debug, PartialEq)]
pub struct LinearModel {
    pub dim: usize,
    pub ridge: f64,
    pub forgetting: f64, // per-update geometric discount on the true sums
    pub scale: f64,      // movable origin: true sums are scale * xx, scale * xy
    pub xx: Vec<f64>,    // dim pre-scaled sums of x_j^2 (the diagonal)
    pub xy: Vec<f64>,    // dim pre-scaled sums of reward * x_j
}

pub fn new_model(dim: usize, forgetting: f64, ridge: f64) -> Result<LinearModel, Error> {
    if dim < 1 {
        return Err(Error::new("dim must be at least 1"));
    }
    if !(0.0 < forgetting && forgetting <= 1.0) {
        return Err(Error::new("forgetting must be in (0, 1] (1.0 = never forget)"));
    }
    if !(ridge > 0.0) {
        return Err(Error::new("ridge must be positive"));
    }
    Ok(LinearModel {
        dim,
        ridge,
        forgetting,
        scale: 1.0,
        xx: vec![0.0; dim],
        xy: vec![0.0; dim],
    })
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
            *u *= scale;
        }
        for u in model.xy.iter_mut() {
            *u *= scale;
        }
        model.scale = 1.0;
    } else {
        model.scale = scale;
    }
}

/// The candidate-independent part of prediction, solved lazily: a
/// coordinate's (1/a_j, theta_j) is computed when a candidate touches it.
/// A diagonal system needs one reciprocal per *touched* coordinate, never
/// O(dim); the type is kept for the once-per-decision shape it gives the
/// layer above. Valid until the next update.
///
/// The reference memoizes each coordinate for the rest of the decision and
/// this core does not (the second deliberate difference, after in-place
/// mutation) — every value is identical either way, since a memo only ever
/// returns the same reciprocal the recompute produces. Under the hashed
/// encoding the memo has almost nothing to hit: every slot comes from a
/// pair hash folding in the arm identity, and `encode` has already merged
/// a candidate's duplicate slots, so nearly every lookup is a first touch
/// and hashing the key costs several times the reciprocal it saves.
pub struct Factorization<'a> {
    model: &'a LinearModel,
}

impl<'a> Factorization<'a> {
    /// (1 / a_j, theta_j) for one coordinate. Takes `&mut self` so that
    /// reinstating a memo stays a change to this function alone.
    pub fn coordinate(&mut self, j: usize) -> (f64, f64) {
        let m = self.model;
        let inv_a = 1.0 / (m.scale * m.xx[j] + m.ridge);
        (inv_a, (m.scale * m.xy[j]) * inv_a)
    }
}

/// The lazy solve: O(1) now, one reciprocal per touched coordinate later.
pub fn factorize(model: &LinearModel) -> Factorization<'_> {
    Factorization { model }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The golden corpus never drives the scale down to RENORM_THRESHOLD
    /// (that takes hundreds of updates), so the renormalization path gets
    /// its own check: crossing the threshold must reset the scale without
    /// moving the true sums (up to the one rounding multiply per entry).
    #[test]
    fn renormalization_preserves_true_sums() {
        // forgetting = 0.5 halves the scale per update: exactly 2^-512
        // after 512 updates, so update 512 renormalizes.
        let x: Features = vec![(0, 1.0)];
        let mut m = new_model(1, 0.5, 1.0).unwrap();
        let mut true_xx = 0.0;
        for i in 0..600 {
            update(&mut m, &x, 1.0);
            true_xx = 0.5 * true_xx + 1.0;
            if i == 511 {
                assert!(m.scale == 1.0, "renorm did not trigger at 2^-512");
            }
        }
        let stored = m.scale * m.xx[0];
        assert!(
            (stored - true_xx).abs() <= 1e-12 * true_xx,
            "true sums moved across renormalization: {stored} vs {true_xx}"
        );
    }

    /// forgetting = 1.0 (never forget) is excluded from the golden corpus
    /// by design, so it gets pinned here as the reference's
    /// pytest suite pins it: the scale stays exactly 1.0, the sums are plain
    /// sums, and the weight is the shrunk running average n / (n + ridge).
    #[test]
    fn never_forget_keeps_plain_sums() {
        let x: Features = vec![(0, 1.0)];
        let mut m = new_model(2, 1.0, 1.0).unwrap();
        for _ in 0..9 {
            update(&mut m, &x, 1.0);
        }
        assert!(m.scale == 1.0, "never-forget scale moved");
        assert!(m.xx[0] == 9.0 && m.xy[0] == 9.0, "sums are not plain sums");
        let (estimate, _) = predict(&m, &x);
        assert!(estimate == 9.0 * (1.0 / 10.0), "weight is not 9/(9+1)");
    }
}
