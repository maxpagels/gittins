//! Fully hashed feature encoding — the port of `encoding.py`.
//!
//! Features arrive as (name, value) pairs; every feature, categorical
//! value, arm identity, intercept, and interaction hashes into one 2^bits
//! space via the outer product ([bias]+context) x ([bias]+action+identity).
//! Tokens are processed in sorted name order and per-slot contributions
//! accumulate in outer-product iteration order, so every output value is
//! bit-identical to the reference's.

use crate::error::Error;
use crate::rng::{fnv1a_extend, mix64, FNV_START};

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

/// The FNV-1a accumulator after folding `left_token` and the separator:
/// the shared prefix state of every pair hash with this left token, folded
/// once per left token instead of once per pair.
fn left_prefix(left_token: &str) -> u64 {
    fnv1a_extend(fnv1a_extend(FNV_START, left_token.as_bytes()), &[PAIR_SEP])
}

/// The 64-bit hash of one (left token, right token) pair: FNV-1a over the
/// tokens joined by the separator byte, finished with the splitmix64 mixer.
/// Streamed through the accumulator — the same byte fold as hashing the
/// joined string, with no intermediate buffer.
pub fn pair_hash(left_token: &str, right_token: &str) -> u64 {
    mix64(fnv1a_extend(left_prefix(left_token), right_token.as_bytes()))
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
    // Each right token as the byte parts feature_tokens would have
    // concatenated — bias "", then "a|name" / "a|name=value" in sorted
    // name order, then "i|arm" — folded through the accumulator per pair
    // and never materialized as a string. Same bytes, same order, same
    // hashes; what disappears is one String allocation per action token
    // per candidate per decision.
    const EMPTY: &[u8] = b"";
    let mut sorted: Vec<&(String, Value)> = action.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut right: Vec<([&[u8]; 4], f64)> = Vec::with_capacity(sorted.len() + 2);
    right.push(([EMPTY, EMPTY, EMPTY, EMPTY], 1.0)); // the bias token ""
    for (name, value) in sorted {
        match value {
            Value::None => {}
            Value::Str(s) => {
                right.push(([b"a|", name.as_bytes(), b"=", s.as_bytes()], 1.0))
            }
            Value::Num(v) => right.push(([b"a|", name.as_bytes(), EMPTY, EMPTY], *v)),
        }
    }
    right.push(([b"i|", arm_id.as_bytes(), EMPTY, EMPTY], 1.0));
    // The outer product, flat: (slot, contribution) in iteration order,
    // then a stable sort by slot. Stability keeps equal slots in encounter
    // order, so summing each run left to right performs the exact addition
    // sequence the per-slot accumulator did — same bits, no tree of nodes.
    let mut pairs: Vec<(usize, f64)> = Vec::with_capacity(left.len() * right.len());
    for (left_token, left_value) in left {
        let prefix = left_prefix(left_token);
        for (parts, right_value) in &right {
            let mut folded = prefix;
            for part in parts {
                folded = fnv1a_extend(folded, part);
            }
            let h = mix64(folded);
            let sign = if h >> 63 == 0 { 1.0 } else { -1.0 };
            pairs.push(((h & mask) as usize, sign * left_value * right_value));
        }
    }
    pairs.sort_by_key(|&(slot, _)| slot); // stable: encounter order per slot
    let mut out: Features = Vec::with_capacity(pairs.len());
    for (slot, v) in pairs {
        match out.last_mut() {
            Some(last) if last.0 == slot => last.1 += v,
            _ => out.push((slot, v)),
        }
    }
    out.retain(|&(_, v)| v != 0.0);
    Ok(out)
}
