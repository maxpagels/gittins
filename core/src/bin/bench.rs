//! The native-core leg of the decision-cycle benchmark: the same grid
//! workload as bindings/python/bench.py (which invokes this), driven
//! straight through the core's api module with no binding boundary at all
//! — the floor the bindings are measured against.
//!
//! Args: bits min_seconds max_rounds variants arms_csv features_csv
//! Output: one line per cell, "arms features seconds_per_decision".

use std::time::Instant;

use gittins_core::api;
use gittins_core::encoding::Value;
use gittins_core::exploration::DEFAULT_EPSILON;
use gittins_core::model::DEFAULT_FORGETTING;

type Features = Vec<(String, Value)>;

// The catalog and context value formulas mirror bench.py's exactly.

fn catalog_for(arms: usize) -> Vec<(String, Features)> {
    (0..arms)
        .map(|a| {
            (
                format!("arm{a}"),
                vec![
                    ("z0".to_string(), Value::Num((a % 7) as f64 * 0.25)),
                    ("z1".to_string(), Value::Num((a % 3) as f64 * 0.5)),
                ],
            )
        })
        .collect()
}

fn contexts_for(n_features: usize, variants: usize) -> Vec<Features> {
    (0..variants)
        .map(|i| {
            let mut context = vec![(
                "seg".to_string(),
                Value::Str(["a", "b", "c", "d"][i % 4].to_string()),
            )];
            for j in 0..n_features - 1 {
                context.push((format!("f{j}"), Value::Num(((i + j) % 10) as f64 * 0.1)));
            }
            context
        })
        .collect()
}

fn drive(
    bits: u32,
    contexts: &[Features],
    catalog: &[(String, Features)],
    min_seconds: f64,
    max_rounds: usize,
) -> f64 {
    let mut state = api::create(bits, 1e9, 0.0, DEFAULT_EPSILON, DEFAULT_FORGETTING).unwrap();
    for i in 0..5 {
        let record =
            api::decide(&mut state, &contexts[i % contexts.len()], catalog, i as f64, "warm", None, None)
                .unwrap();
        api::learn(&mut state, &record.decision_id, 1.0, i as f64, None).unwrap();
    }
    let mut rounds = 0;
    let start = Instant::now();
    while rounds < max_rounds {
        let i = rounds;
        let record =
            api::decide(&mut state, &contexts[i % contexts.len()], catalog, i as f64, "bench", None, None)
                .unwrap();
        api::learn(
            &mut state,
            &record.decision_id,
            if i % 3 != 0 { 1.0 } else { 0.0 },
            i as f64,
            None,
        )
        .unwrap();
        rounds += 1;
        if start.elapsed().as_secs_f64() >= min_seconds {
            break;
        }
    }
    start.elapsed().as_secs_f64() / rounds as f64
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: bench <bits> <min_seconds> <max_rounds> <variants> <arms_csv> <features_csv>";
    assert!(args.len() == 6, "{usage}");
    let bits: u32 = args[0].parse().expect(usage);
    let min_seconds: f64 = args[1].parse().expect(usage);
    let max_rounds: usize = args[2].parse().expect(usage);
    let variants: usize = args[3].parse().expect(usage);
    let arm_counts: Vec<usize> = args[4].split(',').map(|s| s.parse().expect(usage)).collect();
    let feature_counts: Vec<usize> =
        args[5].split(',').map(|s| s.parse().expect(usage)).collect();

    for &arms in &arm_counts {
        let catalog = catalog_for(arms);
        for &n_features in &feature_counts {
            let contexts = contexts_for(n_features, variants);
            let seconds = drive(bits, &contexts, &catalog, min_seconds, max_rounds);
            println!("{arms} {n_features} {seconds}");
        }
    }
}
