//! The JavaScript binding: the public API exposed to
//! the browser and Node via wasm-bindgen — the same nine names as
//! `gittins_reference.api` and the Python wheel. The acceptance gate is the
//! golden `api` and `serialization` sections replayed through this module
//! under Node (`tests/golden.rs`); because WASM mandates IEEE-754
//! semantics and the engine uses no libm transcendentals, passing that gate
//! is also the project's first cross-platform bit-identity check.
//!
//! Shapes:
//! - the state is an opaque handle, **updated in place** — calls return
//!   their result only. This is the uniform convention across every
//!   implementation, reference and Python wheel included.
//! - contexts and action features are plain objects (string values are
//!   categorical, numbers numeric, booleans 1/0, null/undefined absent);
//!   candidates are `[armId, actionFeatures]` pairs.
//! - decision records and resolutions come back as plain objects with the
//!   reference's field names; `candidate_hash` is a BigInt (it uses all 64
//!   bits — a JS number would silently round it).
//! - one boundary crossing per decision: `decide` takes the whole
//!   candidate list, and encode → score → sample run inside the module.
//! - the BYO callbacks are plain JS functions: `score` and
//!   `explore` on `decide`, `train` on `learn`/`expire`. Each crosses the
//!   boundary once per call — with the caller's own context/candidates,
//!   all estimates, or the one resolved record — never per candidate. A
//!   callback's own thrown value is rethrown as itself; a malformed
//!   callback *result* throws an Error with the reference's message.

use std::cell::RefCell;

use js_sys::{Array, BigInt, Function, Object, Reflect};
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

/// A stash for the one value a JS callback may throw while Rust is on the
/// stack: the core sees an `Error` (so it unwinds cleanly with no state
/// half-changed), and the original thrown value is rethrown, as itself, at
/// the boundary.
struct Caught(RefCell<Option<JsValue>>);

impl Caught {
    fn new() -> Caught {
        Caught(RefCell::new(None))
    }

    fn stash(&self, e: JsValue, what: &str) -> Error {
        *self.0.borrow_mut() = Some(e);
        Error::new(format!("{what} callback raised"))
    }

    fn rethrow<T>(&self, result: Result<T, Error>) -> Result<T, JsValue> {
        if let Some(e) = self.0.borrow_mut().take() {
            return Err(e);
        }
        result.map_err(|e| js_error(e).into())
    }
}

/// A callback result as a list of doubles, or the reference's rejection
/// message; the core validates lengths, finiteness, and sums.
fn number_list(value: &JsValue, message: &str) -> Result<Vec<f64>, Error> {
    let array: &Array = value.dyn_ref().ok_or_else(|| Error::new(message))?;
    let mut out = Vec::with_capacity(array.length() as usize);
    for v in array.iter() {
        out.push(v.as_f64().ok_or_else(|| Error::new(message))?);
    }
    Ok(out)
}

/// The BYO train callback as the core's shape: called with
/// the resolved decision's record (a plain object) and the reward, once
/// per resolution.
fn train_callback<'a>(
    f: &'a Function,
    caught: &'a Caught,
) -> impl FnMut(&DecisionRecord, f64) -> Result<(), Error> + 'a {
    move |record: &DecisionRecord, reward: f64| {
        f.call2(
            &JsValue::NULL,
            &record_to_js(record.clone()),
            &JsValue::from_f64(reward),
        )
        .map(|_| ())
        .map_err(|e| caught.stash(e, "train"))
    }
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

fn get(obj: &JsValue, key: &str) -> JsValue {
    Reflect::get(obj, &JsValue::from_str(key)).unwrap_or(JsValue::UNDEFINED)
}

fn record_to_js(record: DecisionRecord) -> JsValue {
    let obj = Object::new();
    // The input fields exist on every record; only `decide` fills them
    // — a train callback's record carries null there, the
    // ledger keeping only the compact record.
    set(&obj, "bits", &JsValue::NULL);
    set(&obj, "context", &JsValue::NULL);
    set(&obj, "candidates", &JsValue::NULL);
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
    forgetfulness: Option<f64>,
) -> Result<BanditState, JsError> {
    Ok(BanditState {
        inner: api::create(
            bits,
            horizon,
            default_reward.unwrap_or(0.0),
            epsilon.unwrap_or(DEFAULT_EPSILON),
            forgetfulness.unwrap_or(DEFAULT_FORGETTING),
        )
        .map_err(js_error)?,
    })
}

#[wasm_bindgen]
pub fn model_bits(state: &BanditState) -> Result<u32, JsError> {
    api::model_bits(&state.inner).map_err(js_error)
}

/// `score` and `explore` are the BYO callbacks:
/// `score(context, candidates)` — called with the very values passed here —
/// and `explore(estimates, epsilon)`; each returns an array of numbers.
#[wasm_bindgen]
pub fn decide(
    state: &mut BanditState,
    context: &JsValue,
    candidates: &JsValue,
    t: f64,
    salt: &str,
    score: Option<Function>,
    explore: Option<Function>,
) -> Result<JsValue, JsValue> {
    let parsed_context = features(context).map_err(JsValue::from)?;
    let parsed = candidate_list(candidates).map_err(JsValue::from)?;
    let caught = Caught::new();
    let caught_ref = &caught;
    let mut score_cb = score.as_ref().map(|f| {
        move |_: &[gittins_core::encoding::Features]| -> Result<Vec<f64>, Error> {
            let result = f
                .call2(&JsValue::NULL, context, candidates)
                .map_err(|e| caught_ref.stash(e, "score"))?;
            number_list(&result, "score must return one finite estimate per candidate")
        }
    });
    let mut explore_cb = explore.as_ref().map(|f| {
        move |estimates: &[f64], epsilon: f64| -> Result<Vec<f64>, Error> {
            let list: Array = estimates.iter().map(|&v| JsValue::from_f64(v)).collect();
            let result = f
                .call2(&JsValue::NULL, &list, &JsValue::from_f64(epsilon))
                .map_err(|e| caught_ref.stash(e, "explore"))?;
            number_list(&result, "explore must return one probability per candidate")
        }
    });
    let bits = api::model_bits(&state.inner).map_err(|e| JsValue::from(js_error(e)))?;
    let result = api::decide(
        &mut state.inner,
        &parsed_context,
        &parsed,
        t,
        salt,
        score_cb.as_mut().map(|c| c as _),
        explore_cb.as_mut().map(|c| c as _),
    );
    let record = record_to_js(caught.rethrow(result)?);
    // The returned record carries the very values the caller passed —
    // attached at the only moment they exist.
    let obj: &Object = record.unchecked_ref();
    set(obj, "bits", &JsValue::from_f64(bits as f64));
    set(obj, "context", context);
    set(obj, "candidates", candidates);
    Ok(record)
}

/// One canonical experience-log line for a decision record
/// or resolution — append it to your log verbatim; it is exactly what
/// the `gittins` CLI's verify/eval/replay consume. Decision records must
/// be the ones `decide` returned (they carry the inputs; a `train`
/// callback's record does not — its inputs are already in your log).
#[wasm_bindgen]
pub fn log_line(item: &JsValue) -> Result<String, JsError> {
    fn quoted(v: &JsValue) -> Result<String, JsError> {
        js_sys::JSON::stringify(v)
            .map(String::from)
            .map_err(|_| JsError::new("log_line could not serialize a field"))
    }
    let absent = |v: &JsValue| v.is_null() || v.is_undefined();
    if !item.is_object() {
        return Err(JsError::new("log_line takes a DecisionRecord or Resolution"));
    }
    if !absent(&get(item, "propensity")) {
        let (bits, context, candidates) =
            (get(item, "bits"), get(item, "context"), get(item, "candidates"));
        if absent(&bits) || absent(&context) || absent(&candidates) {
            return Err(JsError::new(
                "log_line needs the record decide returned (it carries the inputs)",
            ));
        }
        // candidate_hash is a BigInt using all 64 bits: it must land in
        // the JSON as an exact integer token, which JSON.stringify cannot
        // produce — hence the assembled line.
        let hash: BigInt = get(item, "candidate_hash").unchecked_into();
        let hash = String::from(
            hash.to_string(10)
                .map_err(|_| JsError::new("log_line could not serialize a field"))?,
        );
        return Ok(format!(
            "{{\"event\":\"decision\",\"bits\":{},\"context\":{},\"candidates\":{},\
             \"record\":{{\"decision_id\":{},\"t\":{},\"candidate_hash\":{hash},\
             \"chosen\":{},\"features\":{},\"propensity\":{},\"model_version\":{},\"salt\":{}}}}}",
            quoted(&bits)?,
            quoted(&context)?,
            quoted(&candidates)?,
            quoted(&get(item, "decision_id"))?,
            quoted(&get(item, "t"))?,
            quoted(&get(item, "chosen"))?,
            quoted(&get(item, "features"))?,
            quoted(&get(item, "propensity"))?,
            quoted(&get(item, "model_version"))?,
            quoted(&get(item, "salt"))?,
        ));
    }
    if !absent(&get(item, "kind")) {
        return Ok(format!(
            "{{\"event\":\"resolution\",\"decision_id\":{},\"kind\":{},\"reward\":{}}}",
            quoted(&get(item, "decision_id"))?,
            quoted(&get(item, "kind"))?,
            quoted(&get(item, "reward"))?,
        ));
    }
    Err(JsError::new("log_line takes a DecisionRecord or Resolution"))
}

/// The resolution object, or null if the id is unknown or already resolved.
/// The engine classifies against the horizon at time `t`: a reward
/// arriving at or past the record's `t + horizon` resolves as expired
/// with the default reward. `train` is the BYO training tap: it replaces
/// the built-in update and fires after the resolution commits, exactly
/// once, with the record object and the classified reward.
#[wasm_bindgen]
pub fn learn(
    state: &mut BanditState,
    decision_id: &str,
    reward: f64,
    t: f64,
    train: Option<Function>,
) -> Result<JsValue, JsValue> {
    let caught = Caught::new();
    let mut train_cb = train.as_ref().map(|f| train_callback(f, &caught));
    let result = api::learn(
        &mut state.inner,
        decision_id,
        reward,
        t,
        train_cb.as_mut().map(|c| c as _),
    );
    Ok(match caught.rethrow(result)? {
        Some(resolution) => resolution_to_js(resolution),
        None => JsValue::NULL,
    })
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
/// ledger order. `train` as on `learn`, fired per due record.
#[wasm_bindgen]
pub fn expire(
    state: &mut BanditState,
    t: f64,
    train: Option<Function>,
) -> Result<Array, JsValue> {
    let caught = Caught::new();
    let mut train_cb = train.as_ref().map(|f| train_callback(f, &caught));
    let result = api::expire(&mut state.inner, t, train_cb.as_mut().map(|c| c as _));
    let out = Array::new();
    for resolution in caught.rethrow(result)? {
        out.push(&resolution_to_js(resolution));
    }
    Ok(out)
}

#[wasm_bindgen]
/// The current state as one plain hex string — drops straight into
/// localStorage, JSON, or a database column, no encoding on the caller's
/// side.
pub fn serialize(state: &BanditState) -> String {
    api::serialize(&state.inner)
}

#[wasm_bindgen]
pub fn deserialize(data: &str) -> Result<BanditState, JsError> {
    Ok(BanditState {
        inner: api::deserialize(data).map_err(js_error)?,
    })
}
