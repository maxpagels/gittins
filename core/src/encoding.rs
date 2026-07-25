//! Fully hashed feature encoding — the port of `encoding.py`.
//!
//! Features arrive as (name, value) pairs; every feature, categorical
//! value, arm identity, intercept, and interaction hashes into one 2^bits
//! space via the outer product ([bias]+context) x ([bias]+action+identity).
//! Tokens are processed in sorted name order and per-slot contributions
//! accumulate in outer-product iteration order, so every output value is
//! bit-identical to the reference's.

use std::collections::BTreeMap;

use crate::error::Error;
use crate::rng::{fnv1a_64, mix64};

/// Separator between the two tokens of a pair; no printable token contains it.
const PAIR_SEP: u8 = 0x1f;

/// A feature value as the public dict-shaped API supplies it. Booleans count
/// as numeric (true = 1.0); `None` means absent (the feature is skipped).
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Str(String),
    Num(f64),
    None,
}

/// A sparse candidate: (index, value) pairs in strictly increasing index
/// order, values nonzero — the format every core layer consumes.
pub type Features = Vec<(usize, f64)>;

/// (token, contribution) per supplied feature, in sorted token order.
pub fn feature_tokens(namespace: &str, values: &[(String, Value)]) -> Vec<(String, f64)> {
    let mut sorted: Vec<&(String, Value)> = values.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = Vec::with_capacity(sorted.len());
    for (name, value) in sorted {
        match value {
            Value::None => {}
            Value::Str(s) => out.push((format!("{namespace}|{name}={s}"), 1.0)),
            Value::Num(v) => out.push((format!("{namespace}|{name}"), *v)),
        }
    }
    out
}

/// The 64-bit hash of one (left token, right token) pair: FNV-1a over the
/// tokens joined by the separator byte, finished with the splitmix64 mixer.
pub fn pair_hash(left_token: &str, right_token: &str) -> u64 {
    let mut data = Vec::with_capacity(left_token.len() + 1 + right_token.len());
    data.extend_from_slice(left_token.as_bytes());
    data.push(PAIR_SEP);
    data.extend_from_slice(right_token.as_bytes());
    mix64(fnv1a_64(&data))
}

/// The left half of the outer product — the bias token plus the context's
/// tokens. The context is candidate-invariant, so callers encoding a whole
/// candidate set compute this once per decision and pass it to
/// `encode_with_context`, instead of rebuilding (sort + format) the same
/// list once per candidate.
pub fn context_tokens(context: &[(String, Value)]) -> Vec<(String, f64)> {
    let mut left = vec![(String::new(), 1.0)];
    left.extend(feature_tokens("c", context));
    left
}

/// The sparse feature vector for one (context, candidate) pair — the hashed
/// outer product, as (index, value) pairs in strictly increasing index
/// order, exact zeros absent. The model dimension is 2^bits.
pub fn encode(
    context: &[(String, Value)],
    arm_id: &str,
    action: &[(String, Value)],
    bits: u32,
) -> Result<Features, Error> {
    encode_with_context(&context_tokens(context), arm_id, action, bits)
}

/// `encode` with the context's token list already built (`context_tokens`):
/// the same outer product over the same tokens in the same order, so the
/// output is bit-identical to `encode`'s — only the per-candidate rebuild
/// of the left half is saved.
pub fn encode_with_context(
    left: &[(String, f64)],
    arm_id: &str,
    action: &[(String, Value)],
    bits: u32,
) -> Result<Features, Error> {
    if !(1..=24).contains(&bits) {
        return Err(Error::new("bits must be between 1 and 24"));
    }
    let mask = (1u64 << bits) - 1;
    let mut right = vec![(String::new(), 1.0)];
    right.extend(feature_tokens("a", action));
    right.push((format!("i|{arm_id}"), 1.0));
    let mut slots: BTreeMap<usize, f64> = BTreeMap::new();
    for (left_token, left_value) in left {
        for (right_token, right_value) in &right {
            let h = pair_hash(left_token, right_token);
            let sign = if h >> 63 == 0 { 1.0 } else { -1.0 };
            *slots.entry((h & mask) as usize).or_insert(0.0) += sign * left_value * right_value;
        }
    }
    Ok(slots.into_iter().filter(|&(_, v)| v != 0.0).collect())
}
