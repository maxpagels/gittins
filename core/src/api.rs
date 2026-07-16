//! The public API — the port of `api.py`, the dict-shaped facade every
//! binding mirrors (spec: `api.md`). This module is the complete binding
//! surface: PR 20/21 expose exactly these names and nothing else. Every
//! function is the layered API unchanged, with the hashed encoding folded
//! inside `decide` so sparse pairs never cross the public boundary; the
//! facade adds no randomness, no reordering, and no arithmetic, so a
//! decision through it is bit-identical to `encode` + `decide` by hand.

use crate::decide::{decide as decide_encoded, new_bandit, BanditState, DecisionRecord};
use crate::encoding::{encode, Features, Value};
use crate::error::Error;

// Resolution and serialization pass through unchanged; re-exported so the
// binding surface is this one module.
pub use crate::ledger::{censor, expire, learn, Kind, Resolution};
pub use crate::state::{deserialize, serialize};

/// A fresh bandit whose model spans the 2^bits hashed feature space —
/// `bits` is the one encoding declaration; every other parameter is
/// `new_bandit`'s (bindings supply the reference's defaults).
pub fn create(
    bits: u32,
    horizon: f64,
    default_reward: f64,
    epsilon: f64,
    forgetting: f64,
) -> Result<BanditState, Error> {
    if !(1..=24).contains(&bits) {
        return Err(Error::new("bits must be between 1 and 24"));
    }
    new_bandit(1usize << bits, horizon, default_reward, epsilon, forgetting)
}

/// The `bits` declaration recovered from the model dimension. States built
/// by `create` always satisfy dim == 2^bits; anything else was built
/// against the layered API and has no public encoding space.
pub fn model_bits(state: &BanditState) -> Result<u32, Error> {
    let dim = state.model.dim;
    let bits = dim.trailing_zeros();
    if dim.count_ones() != 1 || !(1..=24).contains(&bits) {
        return Err(Error::new(
            "model dimension is not 2**bits for bits in [1, 24]; \
             the state was not built by create()",
        ));
    }
    Ok(bits)
}

/// Score, explore, choose, and record one decision over dict-shaped
/// inputs: `context` is one feature list, each candidate is an
/// (arm_id, action features) pair, encoded here in candidate order.
/// Everything returned and every state change is `decide`'s, unchanged.
pub fn decide(
    state: &mut BanditState,
    context: &[(String, Value)],
    candidates: &[(String, Vec<(String, Value)>)],
    t: f64,
    salt: &str,
) -> Result<DecisionRecord, Error> {
    let bits = model_bits(state)?;
    let encoded: Vec<Features> = candidates
        .iter()
        .map(|(arm_id, action)| encode(context, arm_id, action, bits))
        .collect::<Result<_, _>>()?;
    decide_encoded(state, &encoded, t, salt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exploration::DEFAULT_EPSILON;
    use crate::model::DEFAULT_FORGETTING;

    fn catalog() -> Vec<(String, Vec<(String, Value)>)> {
        vec![
            ("basic".into(), vec![("price".into(), Value::Num(3.0))]),
            ("plus".into(), vec![("price".into(), Value::Num(9.0))]),
            ("free".into(), vec![]),
        ]
    }

    /// The facade must add nothing: same record, same state, bit for bit,
    /// as encoding by hand and deciding over the sparse pairs.
    #[test]
    fn decide_is_exactly_the_layered_path() {
        let context = vec![("seg".to_string(), Value::Str("a".to_string()))];
        let mut facade = create(4, 10.0, 0.0, DEFAULT_EPSILON, DEFAULT_FORGETTING).unwrap();
        let mut layered = facade.clone();
        for t in [0.0, 1.0, 2.0] {
            let record = decide(&mut facade, &context, &catalog(), t, "s").unwrap();
            let encoded: Vec<Features> = catalog()
                .iter()
                .map(|(arm, action)| encode(&context, arm, action, 4).unwrap())
                .collect();
            let expected = decide_encoded(&mut layered, &encoded, t, "s").unwrap();
            assert!(record == expected);
        }
        assert!(facade == layered);
    }

    #[test]
    fn rejects_bits_and_dimensions_outside_the_declaration() {
        for bits in [0, 25] {
            assert!(create(bits, 10.0, 0.0, DEFAULT_EPSILON, DEFAULT_FORGETTING).is_err());
        }
        // dim 5: not a power of two — built against the layered API.
        let mut state = new_bandit(5, 10.0, 0.0, DEFAULT_EPSILON, DEFAULT_FORGETTING).unwrap();
        assert!(decide(&mut state, &[], &catalog(), 0.0, "s")
            .unwrap_err()
            .message()
            .contains("2**bits"));
    }
}
