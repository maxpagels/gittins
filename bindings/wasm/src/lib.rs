//! The JavaScript binding: the public API (spec: `spec/api.md`) exposed to
//! the browser and Node via wasm-bindgen — the same eight names as
//! `gittins_reference.api` and the Python wheel. The acceptance gate is the
//! golden `api` and `serialization` sections replayed through this module
//! under Node (`tests/golden.rs`); because WASM mandates IEEE-754
//! semantics and the engine uses no libm transcendentals, passing that gate
//! is also the project's first cross-platform bit-identity check.
//!
//! Shapes, translated to JavaScript:
//! - the state is an opaque handle, **mutated in place** — calls return
//!   their result only, not a `(result, state)` pair, since rebinding
//!   tuples is a Python idiom, not a JS one. Everything else matches the
//!   reference name for name.
//! - contexts and action features are plain objects (string values are
//!   categorical, numbers numeric, booleans 1/0, null/undefined absent);
//!   candidates are `[armId, actionFeatures]` pairs.
//! - decision records and resolutions come back as plain objects with the
//!   reference's field names; `candidate_hash` is a BigInt (it uses all 64
//!   bits — a JS number would silently round it).
//! - one boundary crossing per decision: `decide` takes the whole
//!   candidate list, and encode → score → sample run inside the module.

use js_sys::{Array, BigInt, Object, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;

use gittins_core::api;
use gittins_core::decide::{BanditState as CoreState, DecisionRecord};
use gittins_core::encoding::Value;
use gittins_core::error::Error;
use gittins_core::exploration::DEFAULT_EPSILON;
use gittins_core::ledger::Resolution;
use gittins_core::model::DEFAULT_FORGETTING;

// The reference raises ValueError; here every rejection throws a JS Error
// with the same message.
fn js_error(e: Error) -> JsError {
    JsError::new(e.message())
}

/// A feature object as the core's (name, Value) pairs — the reference's
/// contract: strings categorical, numbers (booleans included) numeric,
/// null/undefined absent, anything else an error naming the feature.
fn features(obj: &JsValue) -> Result<Vec<(String, Value)>, JsError> {
    if !obj.is_object() {
        return Err(JsError::new("features must be a plain object"));
    }
    let entries = Object::entries(obj.unchecked_ref());
    let mut out = Vec::with_capacity(entries.length() as usize);
    for entry in entries.iter() {
        let pair: Array = entry.unchecked_into();
        let name = pair
            .get(0)
            .as_string()
            .ok_or_else(|| JsError::new("feature names must be strings"))?;
        let v = pair.get(1);
        let value = if v.is_null() || v.is_undefined() {
            Value::None
        } else if let Some(s) = v.as_string() {
            Value::Str(s)
        } else if let Some(b) = v.as_bool() {
            Value::Num(if b { 1.0 } else { 0.0 })
        } else if let Some(x) = v.as_f64() {
            Value::Num(x)
        } else {
            return Err(JsError::new(&format!(
                "feature '{name}' has unsupported type"
            )));
        };
        out.push((name, value));
    }
    Ok(out)
}

/// The `[armId, actionFeatures]` candidate list.
fn candidate_list(candidates: &JsValue) -> Result<Vec<(String, Vec<(String, Value)>)>, JsError> {
    let array: &Array = candidates
        .dyn_ref()
        .ok_or_else(|| JsError::new("candidates must be an array of [armId, features] pairs"))?;
    let mut out = Vec::with_capacity(array.length() as usize);
    for item in array.iter() {
        let pair: &Array = item
            .dyn_ref()
            .ok_or_else(|| JsError::new("each candidate must be an [armId, features] pair"))?;
        let arm_id = pair
            .get(0)
            .as_string()
            .ok_or_else(|| JsError::new("arm ids must be strings"))?;
        out.push((arm_id, features(&pair.get(1))?));
    }
    Ok(out)
}

fn set(obj: &Object, key: &str, value: &JsValue) {
    Reflect::set(obj, &JsValue::from_str(key), value).expect("setting a plain object field");
}

fn record_to_js(record: DecisionRecord) -> JsValue {
    let obj = Object::new();
    set(&obj, "decision_id", &JsValue::from_str(&record.decision_id));
    set(&obj, "t", &JsValue::from_f64(record.t));
    set(&obj, "candidate_hash", &BigInt::from(record.candidate_hash).into());
    set(&obj, "chosen", &JsValue::from_f64(record.chosen as f64));
    let features = Array::new();
    for (j, v) in &record.features {
        features.push(&Array::of2(&JsValue::from_f64(*j as f64), &JsValue::from_f64(*v)));
    }
    set(&obj, "features", &features);
    set(&obj, "propensity", &JsValue::from_f64(record.propensity));
    set(&obj, "model_version", &JsValue::from_f64(record.model_version as f64));
    set(&obj, "salt", &JsValue::from_str(&record.salt));
    obj.into()
}

fn resolution_to_js(resolution: Resolution) -> JsValue {
    let obj = Object::new();
    set(&obj, "decision_id", &JsValue::from_str(&resolution.decision_id));
    set(&obj, "kind", &JsValue::from_str(resolution.kind.as_str()));
    let reward = match resolution.reward {
        Some(v) => JsValue::from_f64(v),
        None => JsValue::NULL,
    };
    set(&obj, "reward", &reward);
    obj.into()
}

/// The bandit's state: an opaque handle, mutated in place by decide/learn/
/// censor/expire; persist it with serialize/deserialize.
#[wasm_bindgen]
pub struct BanditState {
    inner: CoreState,
}

#[wasm_bindgen]
pub fn create(
    bits: u32,
    horizon: f64,
    default_reward: Option<f64>,
    epsilon: Option<f64>,
    forgetting: Option<f64>,
) -> Result<BanditState, JsError> {
    Ok(BanditState {
        inner: api::create(
            bits,
            horizon,
            default_reward.unwrap_or(0.0),
            epsilon.unwrap_or(DEFAULT_EPSILON),
            forgetting.unwrap_or(DEFAULT_FORGETTING),
        )
        .map_err(js_error)?,
    })
}

#[wasm_bindgen]
pub fn model_bits(state: &BanditState) -> Result<u32, JsError> {
    api::model_bits(&state.inner).map_err(js_error)
}

#[wasm_bindgen]
pub fn decide(
    state: &mut BanditState,
    context: &JsValue,
    candidates: &JsValue,
    t: f64,
    salt: &str,
) -> Result<JsValue, JsError> {
    let context = features(context)?;
    let candidates = candidate_list(candidates)?;
    let record =
        api::decide(&mut state.inner, &context, &candidates, t, salt).map_err(js_error)?;
    Ok(record_to_js(record))
}

/// The resolution object, or null if the id is unknown or already resolved.
#[wasm_bindgen]
pub fn learn(state: &mut BanditState, decision_id: &str, reward: f64) -> JsValue {
    match api::learn(&mut state.inner, decision_id, reward) {
        Some(resolution) => resolution_to_js(resolution),
        None => JsValue::NULL,
    }
}

/// The resolution object, or null if the id is unknown or already resolved.
#[wasm_bindgen]
pub fn censor(state: &mut BanditState, decision_id: &str) -> JsValue {
    match api::censor(&mut state.inner, decision_id) {
        Some(resolution) => resolution_to_js(resolution),
        None => JsValue::NULL,
    }
}

/// Every decision past its horizon at time `t`, resolved as expired, in
/// ledger order.
#[wasm_bindgen]
pub fn expire(state: &mut BanditState, t: f64) -> Array {
    let out = Array::new();
    for resolution in api::expire(&mut state.inner, t) {
        out.push(&resolution_to_js(resolution));
    }
    out
}

#[wasm_bindgen]
pub fn serialize(state: &BanditState) -> Uint8Array {
    Uint8Array::from(api::serialize(&state.inner).as_slice())
}

#[wasm_bindgen]
pub fn deserialize(data: &[u8]) -> Result<BanditState, JsError> {
    Ok(BanditState {
        inner: api::deserialize(data).map_err(js_error)?,
    })
}
