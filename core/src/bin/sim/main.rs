//! `cargo run --release --bin sim`: the environment battery on the Rust
//! core — the port of `python -m sim`, printing the same markdown table
//! over the same seeded worlds. Run both and diff: every draw comes from
//! the same counter-RNG streams, so on any one platform the numbers should
//! match the Python battery's exactly (libm is the shared dependency —
//! see rand.rs), while the total runtime prices the compiled core.

mod environments;
mod metrics;
mod policies;
mod rand;
mod runner;

use std::time::Instant;

use environments::{
    AbruptShiftEnvironment, ActionFeatureEnvironment, ChurnEnvironment, DriftEnvironment,
    DropoutEnvironment, Environment, LinearEnvironment, NeedleEnvironment, XorEnvironment,
};
use metrics::{final_window_regret, median_iqr, normalized_regret, rmse};
use policies::{
    EpsilonGreedyPolicy, GittinsPolicy, GreedyPolicy, OraclePolicy, Policy, UniformPolicy,
};
use runner::run;

const SEEDS: std::ops::Range<u64> = 0..5;
const ROUNDS: usize = 1500;
const BITS: u32 = 8;

fn spread(values: &[f64]) -> String {
    let (median, q1, q3) = median_iqr(values);
    format!("{median:.3} [{q1:.3}, {q3:.3}]")
}

fn commas(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn main() {
    let start = Instant::now();
    let mut decisions: usize = 0;

    let environments: Vec<Box<dyn Environment>> = vec![
        Box::new(LinearEnvironment::new(5)),
        Box::new(LinearEnvironment::new(20)),
        Box::new(XorEnvironment::new(4)),
        Box::new(XorEnvironment::new(10)),
        Box::new(NeedleEnvironment::new(10)),
        Box::new(ActionFeatureEnvironment::new(16)),
        Box::new(AbruptShiftEnvironment::new(5, 375)),
        Box::new(DriftEnvironment::new(5, 500)),
        Box::new(ChurnEnvironment::new(8)),
        Box::new(DropoutEnvironment::new(5, 0.3)),
    ];
    let policies: Vec<Box<dyn Fn() -> Box<dyn Policy>>> = vec![
        Box::new(|| Box::new(OraclePolicy)),
        Box::new(|| Box::new(UniformPolicy)),
        Box::new(|| Box::new(GreedyPolicy::new(BITS))),
        Box::new(|| Box::new(EpsilonGreedyPolicy::new(0.05, BITS))),
        Box::new(|| Box::new(EpsilonGreedyPolicy::new(0.1, BITS))),
        Box::new(|| Box::new(GittinsPolicy::new(BITS))),
    ];

    println!("### Environment battery (Rust core)");
    println!();
    println!(
        "{ROUNDS} rounds, seeds {:?}, bits={BITS}. Normalized regret: \
         0 = oracle, 1 = uniform (median [IQR] over seeds). RMSE is late-run \
         (final half) prediction error vs the oracle means.",
        SEEDS.collect::<Vec<_>>()
    );
    println!();
    println!("| environment | policy | normalized regret | final 10% | RMSE (late) |");
    println!("|---|---|---|---|---|");
    for env in &environments {
        for make in &policies {
            let mut regrets = Vec::new();
            let mut finals = Vec::new();
            let mut errors = Vec::new();
            let mut name = String::new();
            for seed in SEEDS {
                let mut policy = make();
                let result = run(env.as_ref(), policy.as_mut(), seed, ROUNDS);
                decisions += result.regret.len();
                regrets.push(normalized_regret(&result, 0));
                finals.push(final_window_regret(&result, 0.1));
                if let Some(e) = rmse(&result, ROUNDS / 2) {
                    errors.push(e);
                }
                name = policy.name().to_string();
            }
            let err = if errors.is_empty() { "—".to_string() } else { spread(&errors) };
            println!(
                "| {} | {} | {} | {} | {} |",
                env.name(),
                name,
                spread(&regrets),
                spread(&finals),
                err
            );
        }
    }
    println!();
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "_{elapsed:.1}s on {}, Rust (release); {} decisions total, {} decisions/s._",
        std::env::consts::OS,
        commas(decisions),
        commas((decisions as f64 / elapsed) as usize)
    );
}
