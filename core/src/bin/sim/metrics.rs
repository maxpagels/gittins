//! Regret metrics over run results — the port of `sim/metrics.py`.
//! (`recovery_time` is not ported: the battery driver doesn't use it.)

use crate::runner::RunResult;

/// Exact float summation — a faithful port of CPython's `math.fsum`
/// (Shewchuk's algorithm plus CPython's round-half-even correction on the
/// final partial), for finite inputs. The Python metrics use `math.fsum`,
/// so matching its every bit requires matching its exact algorithm.
pub fn fsum(values: &[f64]) -> f64 {
    let mut partials: Vec<f64> = Vec::with_capacity(32);
    for &value in values {
        let mut x = value;
        let mut i = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partials.truncate(i);
        partials.push(x);
    }
    let mut n = partials.len();
    let mut hi = 0.0;
    if n > 0 {
        n -= 1;
        hi = partials[n];
        let mut lo = 0.0;
        while n > 0 {
            let x = hi;
            n -= 1;
            let y = partials[n];
            hi = x + y;
            let yr = hi - x;
            lo = y - yr;
            if lo != 0.0 {
                break;
            }
        }
        // Round half-to-even when the discarded tail is an exact half ulp.
        if n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0)) {
            let y = lo * 2.0;
            let x = hi + y;
            if y == x - hi {
                hi = x;
            }
        }
    }
    hi
}

/// Cumulative regret over rounds [first, last) as a fraction of uniform
/// random's expected cumulative regret on the same rounds (0 = oracle,
/// 1 = uniform).
pub fn normalized_regret(result: &RunResult, first: usize) -> f64 {
    let total = fsum(&result.regret[first..]);
    let unit = fsum(&result.normalizer[first..]);
    if unit == 0.0 {
        return 0.0;
    }
    total / unit
}

/// Normalized regret over just the final window of the run.
pub fn final_window_regret(result: &RunResult, window_fraction: f64) -> f64 {
    let n = result.regret.len();
    let window = ((n as f64 * window_fraction) as usize).max(1);
    normalized_regret(result, n - window)
}

/// Root-mean-square prediction error vs the oracle means over rounds
/// [first, ..); None for model-free policies.
pub fn rmse(result: &RunResult, first: usize) -> Option<f64> {
    let sq_error = result.sq_error.as_ref()?;
    let window = &sq_error[first..];
    Some((fsum(window) / window.len() as f64).sqrt())
}

/// The q-quantile with linear interpolation (numpy's default rule).
pub fn quantile(values: &[f64], q: f64) -> f64 {
    assert!(!values.is_empty(), "need at least one value");
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let position = q * (ordered.len() - 1) as f64;
    let lo = position as usize;
    let hi = (lo + 1).min(ordered.len() - 1);
    ordered[lo] + (position - lo as f64) * (ordered[hi] - ordered[lo])
}

/// (median, 25th percentile, 75th percentile) over seeds.
pub fn median_iqr(values: &[f64]) -> (f64, f64, f64) {
    (
        quantile(values, 0.5),
        quantile(values, 0.25),
        quantile(values, 0.75),
    )
}
