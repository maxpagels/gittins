//! The environment protocol and the battery worlds — the port of
//! `sim/environments.py`. Rounds are pure functions of (environment name,
//! seed, t); every arithmetic expression keeps the Python module's exact
//! order so realized rounds and rewards match it bit for bit (up to libm,
//! see rand.rs). Python memoizes per-seed hidden parameters; here they are
//! recomputed per round — same draws from the same streams, just cheaper to
//! write, and Rust pays the arithmetic without noticing.

use gittins_core::encoding::Value;

use crate::rand::{gaussian, randint, stream, uniform};

pub struct Round {
    pub t: f64,
    pub context: Vec<(String, Value)>,
    pub arm_ids: Vec<String>,
    pub actions: Vec<Vec<(String, Value)>>,
    pub means: Vec<f64>,
}

pub trait Environment {
    fn name(&self) -> &str;
    fn noise(&self) -> f64;
    fn round(&self, seed: u64, t: usize) -> Round;

    /// The reward rule, shared: chosen mean plus seeded gaussian noise.
    fn reward(&self, seed: u64, t: usize, rd: &Round, chosen: usize) -> f64 {
        let key = stream(self.name(), seed, &format!("reward:{t}"));
        rd.means[chosen] + self.noise() * gaussian(key, 0)
    }
}

fn arm_ids(k: usize) -> Vec<String> {
    (0..k).map(|a| format!("arm{a}")).collect()
}

fn numeric_context(prefix: &str, x: &[f64]) -> Vec<(String, Value)> {
    x.iter()
        .enumerate()
        .map(|(j, &v)| (format!("{prefix}{j}"), Value::Num(v)))
        .collect()
}

/// One hidden linear world — (per-arm base, per-arm slope vector) — as a
/// pure function of (name, seed, label).
fn linear_params(
    name: &str,
    seed: u64,
    label: &str,
    k: usize,
    n_features: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let key = stream(name, seed, label);
    let scale = 1.0 / n_features as f64;
    let mut bases = Vec::with_capacity(k);
    let mut weights = Vec::with_capacity(k);
    let mut c = 0;
    for _ in 0..k {
        bases.push(uniform(key, c, -0.5, 0.5));
        c += 1;
        let mut w = Vec::with_capacity(n_features);
        for _ in 0..n_features {
            w.push(uniform(key, c, -scale, scale));
            c += 1;
        }
        weights.push(w);
    }
    (bases, weights)
}

/// base_a + w[a] . x for every arm, accumulated in the reference's order.
fn linear_means(bases: &[f64], weights: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    bases
        .iter()
        .zip(weights)
        .map(|(&base, w)| {
            let mut dot = 0.0;
            for j in 0..x.len() {
                dot += w[j] * x[j];
            }
            base + dot
        })
        .collect()
}

fn context_draw(name: &str, seed: u64, t: usize, n: usize) -> (u64, Vec<f64>) {
    let key = stream(name, seed, &format!("context:{t}"));
    let x = (0..n).map(|j| uniform(key, j as u64, -1.0, 1.0)).collect();
    (key, x)
}

/// Stationary, well-specified: the model's home turf.
pub struct LinearEnvironment {
    name: String,
    k: usize,
    n_features: usize,
    noise: f64,
}

impl LinearEnvironment {
    pub fn new(k: usize) -> Self {
        let n_features = 3;
        LinearEnvironment {
            name: format!("linear-k{k}-f{n_features}"),
            k,
            n_features,
            noise: 0.1,
        }
    }
}

impl Environment for LinearEnvironment {
    fn name(&self) -> &str {
        &self.name
    }
    fn noise(&self) -> f64 {
        self.noise
    }
    fn round(&self, seed: u64, t: usize) -> Round {
        let (bases, weights) = linear_params(&self.name, seed, "params", self.k, self.n_features);
        let (_, x) = context_draw(&self.name, seed, t, self.n_features);
        Round {
            t: t as f64,
            context: numeric_context("f", &x),
            arm_ids: arm_ids(self.k),
            actions: vec![Vec::new(); self.k],
            means: linear_means(&bases, &weights, &x),
        }
    }
}

/// Stationary, misspecified: the graceful-degradation check.
pub struct XorEnvironment {
    name: String,
    k: usize,
    noise: f64,
}

impl XorEnvironment {
    const HIGH: f64 = 0.75;
    const LOW: f64 = 0.25;

    pub fn new(k: usize) -> Self {
        XorEnvironment {
            name: format!("xor-k{k}"),
            k,
            noise: 0.1,
        }
    }
}

impl Environment for XorEnvironment {
    fn name(&self) -> &str {
        &self.name
    }
    fn noise(&self) -> f64 {
        self.noise
    }
    fn round(&self, seed: u64, t: usize) -> Round {
        let key = stream(&self.name, seed, &format!("context:{t}"));
        let s1 = randint(key, 0, 2);
        let s2 = randint(key, 1, 2);
        let parity = s1 ^ s2;
        Round {
            t: t as f64,
            context: vec![
                ("s1".to_string(), Value::Str(s1.to_string())),
                ("s2".to_string(), Value::Str(s2.to_string())),
            ],
            arm_ids: arm_ids(self.k),
            actions: vec![Vec::new(); self.k],
            means: (0..self.k)
                .map(|a| if a % 2 == parity { Self::HIGH } else { Self::LOW })
                .collect(),
        }
    }
}

/// Stationary, context-free: the pure exploration stress test.
pub struct NeedleEnvironment {
    name: String,
    k: usize,
    gap: f64,
    noise: f64,
}

impl NeedleEnvironment {
    const BASE: f64 = 0.5;

    pub fn new(k: usize) -> Self {
        let gap = 0.2;
        NeedleEnvironment {
            name: format!("needle-k{k}-g{gap}"),
            k,
            gap,
            noise: 0.1,
        }
    }
}

impl Environment for NeedleEnvironment {
    fn name(&self) -> &str {
        &self.name
    }
    fn noise(&self) -> f64 {
        self.noise
    }
    fn round(&self, seed: u64, t: usize) -> Round {
        let needle = randint(stream(&self.name, seed, "needle"), 0, self.k);
        Round {
            t: t as f64,
            context: Vec::new(),
            arm_ids: arm_ids(self.k),
            actions: vec![Vec::new(); self.k],
            means: (0..self.k)
                .map(|a| if a == needle { Self::BASE + self.gap } else { Self::BASE })
                .collect(),
        }
    }
}

/// Stationary, well-specified through action features: the
/// generalization-across-arms check.
pub struct ActionFeatureEnvironment {
    name: String,
    k: usize,
    n_features: usize,
    noise: f64,
}

impl ActionFeatureEnvironment {
    pub fn new(k: usize) -> Self {
        let n_features = 3;
        ActionFeatureEnvironment {
            name: format!("actions-k{k}-f{n_features}"),
            k,
            n_features,
            noise: 0.1,
        }
    }

    /// (per-arm action vectors, hidden weight matrix) for one seed.
    fn params(&self, seed: u64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let key = stream(&self.name, seed, "params");
        let n = self.n_features;
        let scale = 1.0 / n as f64;
        let mut c: u64 = 0;
        let mut arms = Vec::with_capacity(self.k);
        for _ in 0..self.k {
            arms.push((0..n).map(|j| uniform(key, c + j as u64, -1.0, 1.0)).collect());
            c += n as u64;
        }
        let mut weights = Vec::with_capacity(n);
        for _ in 0..n {
            weights.push((0..n).map(|j| uniform(key, c + j as u64, -scale, scale)).collect());
            c += n as u64;
        }
        (arms, weights)
    }
}

impl Environment for ActionFeatureEnvironment {
    fn name(&self) -> &str {
        &self.name
    }
    fn noise(&self) -> f64 {
        self.noise
    }
    fn round(&self, seed: u64, t: usize) -> Round {
        let (arms, weights) = self.params(seed);
        let n = self.n_features;
        let (_, x) = context_draw(&self.name, seed, t, n);
        let means = (0..self.k)
            .map(|a| {
                let mut total = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        total += x[i] * weights[i][j] * arms[a][j];
                    }
                }
                total
            })
            .collect();
        Round {
            t: t as f64,
            context: numeric_context("f", &x),
            arm_ids: arm_ids(self.k),
            actions: (0..self.k).map(|a| numeric_context("z", &arms[a])).collect(),
            means,
        }
    }
}

/// Non-stationary, well-specified within each epoch: the recovery check.
pub struct AbruptShiftEnvironment {
    name: String,
    k: usize,
    period: usize,
    n_features: usize,
    noise: f64,
}

impl AbruptShiftEnvironment {
    pub fn new(k: usize, period: usize) -> Self {
        AbruptShiftEnvironment {
            name: format!("shift-k{k}-p{period}"),
            k,
            period,
            n_features: 3,
            noise: 0.1,
        }
    }
}

impl Environment for AbruptShiftEnvironment {
    fn name(&self) -> &str {
        &self.name
    }
    fn noise(&self) -> f64 {
        self.noise
    }
    fn round(&self, seed: u64, t: usize) -> Round {
        let epoch = t / self.period;
        let (bases, weights) = linear_params(
            &self.name,
            seed,
            &format!("params:{epoch}"),
            self.k,
            self.n_features,
        );
        let (_, x) = context_draw(&self.name, seed, t, self.n_features);
        Round {
            t: t as f64,
            context: numeric_context("f", &x),
            arm_ids: arm_ids(self.k),
            actions: vec![Vec::new(); self.k],
            means: linear_means(&bases, &weights, &x),
        }
    }
}

/// Non-stationary, smooth: parameters rotate continuously between two
/// independently drawn linear worlds.
pub struct DriftEnvironment {
    name: String,
    k: usize,
    period: usize,
    n_features: usize,
    noise: f64,
}

impl DriftEnvironment {
    pub fn new(k: usize, period: usize) -> Self {
        DriftEnvironment {
            name: format!("drift-k{k}-p{period}"),
            k,
            period,
            n_features: 3,
            noise: 0.1,
        }
    }
}

impl Environment for DriftEnvironment {
    fn name(&self) -> &str {
        &self.name
    }
    fn noise(&self) -> f64 {
        self.noise
    }
    fn round(&self, seed: u64, t: usize) -> Round {
        let (b0, w0) = linear_params(&self.name, seed, "params:0", self.k, self.n_features);
        let (b1, w1) = linear_params(&self.name, seed, "params:1", self.k, self.n_features);
        let theta = 2.0 * std::f64::consts::PI * t as f64 / self.period as f64;
        let (c, s) = (theta.cos(), theta.sin());
        let bases: Vec<f64> = (0..self.k).map(|a| c * b0[a] + s * b1[a]).collect();
        let weights: Vec<Vec<f64>> = (0..self.k)
            .map(|a| (0..self.n_features).map(|j| c * w0[a][j] + s * w1[a][j]).collect())
            .collect();
        let (_, x) = context_draw(&self.name, seed, t, self.n_features);
        Round {
            t: t as f64,
            context: numeric_context("f", &x),
            arm_ids: arm_ids(self.k),
            actions: vec![Vec::new(); self.k],
            means: linear_means(&bases, &weights, &x),
        }
    }
}

/// Arm churn: the best arm disappears and returns, and a strictly better
/// newcomer is born late — the candidate list changes shape under the
/// policy.
pub struct ChurnEnvironment {
    name: String,
    k: usize,
    absent: (usize, usize),
    newcomer_at: usize,
    newcomer_gap: f64,
    noise: f64,
}

impl ChurnEnvironment {
    pub fn new(k: usize) -> Self {
        ChurnEnvironment {
            name: format!("churn-k{k}"),
            k,
            absent: (400, 800),
            newcomer_at: 1100,
            newcomer_gap: 0.1,
            noise: 0.1,
        }
    }
}

impl Environment for ChurnEnvironment {
    fn name(&self) -> &str {
        &self.name
    }
    fn noise(&self) -> f64 {
        self.noise
    }
    fn round(&self, seed: u64, t: usize) -> Round {
        let key = stream(&self.name, seed, "means");
        let means: Vec<f64> = (0..self.k).map(|a| uniform(key, a as u64, 0.25, 0.75)).collect();
        let mut best = 0;
        for a in 1..self.k {
            if means[a] > means[best] {
                best = a;
            }
        }
        let mut ids = Vec::new();
        let mut mu = Vec::new();
        for a in 0..self.k {
            if a == best && self.absent.0 <= t && t < self.absent.1 {
                continue;
            }
            ids.push(format!("arm{a}"));
            mu.push(means[a]);
        }
        if t >= self.newcomer_at {
            ids.push("newcomer".to_string());
            mu.push(means[best] + self.newcomer_gap);
        }
        let n = ids.len();
        Round {
            t: t as f64,
            context: Vec::new(),
            arm_ids: ids,
            actions: vec![Vec::new(); n],
            means: mu,
        }
    }
}

/// Missing features: the world is linear in x, but the policy sees a
/// damaged view of it — dropped features plus always-on distractors.
pub struct DropoutEnvironment {
    name: String,
    k: usize,
    n_features: usize,
    p_drop: f64,
    n_distractors: usize,
    noise: f64,
}

impl DropoutEnvironment {
    pub fn new(k: usize, p_drop: f64) -> Self {
        let n_features = 3;
        DropoutEnvironment {
            name: format!("dropout-k{k}-f{n_features}-p{p_drop}"),
            k,
            n_features,
            p_drop,
            n_distractors: 2,
            noise: 0.1,
        }
    }
}

impl Environment for DropoutEnvironment {
    fn name(&self) -> &str {
        &self.name
    }
    fn noise(&self) -> f64 {
        self.noise
    }
    fn round(&self, seed: u64, t: usize) -> Round {
        let (bases, weights) = linear_params(&self.name, seed, "params", self.k, self.n_features);
        let n = self.n_features;
        let (key, x) = context_draw(&self.name, seed, t, n);
        // Draws beyond the features: dropout coins, then distractor values.
        let mut context = Vec::new();
        for j in 0..n {
            if uniform(key, (n + j) as u64, 0.0, 1.0) >= self.p_drop {
                context.push((format!("f{j}"), Value::Num(x[j])));
            }
        }
        for d in 0..self.n_distractors {
            context.push((
                format!("d{d}"),
                Value::Num(uniform(key, (2 * n + d) as u64, -1.0, 1.0)),
            ));
        }
        Round {
            t: t as f64,
            context,
            arm_ids: arm_ids(self.k),
            actions: vec![Vec::new(); self.k],
            means: linear_means(&bases, &weights, &x),
        }
    }
}
