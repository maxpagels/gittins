//! Seeded randomness for the harness — the port of `sim/rand.py`. Every
//! draw is the engine's counter RNG on a stream keyed by (name, seed,
//! label), so runs replay exactly and comparators on the same seed face the
//! same world. Labels are formatted exactly as Python's f-strings format
//! them (floats via `{:?}`, which prints the same shortest repr), so the
//! streams are the same streams.

use gittins_core::rng::{derive_key, random_unit};

/// The 64-bit key of one named draw stream within one (name, seed) run.
pub fn stream(name: &str, seed: u64, label: &str) -> u64 {
    derive_key(&format!("{name}:{seed}:{label}"), "sim")
}

/// A uniform float in [lo, hi) at position `counter` of stream `key`.
pub fn uniform(key: u64, counter: u64, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * random_unit(key, counter)
}

/// A uniform integer in [0, n) at position `counter` of stream `key`.
pub fn randint(key: u64, counter: u64, n: usize) -> usize {
    let i = (random_unit(key, counter) * n as f64) as usize;
    if i >= n {
        n - 1
    } else {
        i
    }
}

/// A standard normal draw (Box-Muller); consumes stream positions
/// 2*counter and 2*counter + 1.
///
/// The one piece of the harness that is not bit-pinned across platforms:
/// ln/cos come from the system libm with no cross-platform rounding
/// guarantee. On any one machine, though, Rust and CPython call the same
/// libm, so the battery's realized rewards match the Python harness exactly
/// there — which is all the side-by-side CI comparison needs.
pub fn gaussian(key: u64, counter: u64) -> f64 {
    let mut u1 = random_unit(key, 2 * counter);
    let u2 = random_unit(key, 2 * counter + 1);
    if u1 <= 0.0 {
        // log(0) guard; random_unit can return exactly 0.0
        u1 = (2.0f64).powi(-53);
    }
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
