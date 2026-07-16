//! The Python binding: `gittins_reference.api` mirrored name for name over
//! the Rust core's `api` module (spec: `spec/api.md`), so the reference's
//! usage docs and this module are interchangeable. The acceptance gate is
//! the golden `api` section replayed through this module alone
//! (`tests/test_binding.py`).
//!
//! The one performance rule (PROGRESS, PR 20): one boundary crossing per
//! decision. `decide` takes the context dict and the whole candidate list
//! and encode → score → sample all happen inside Rust; nothing calls back
//! into Python per candidate or per feature.
//!
//! State handling, the uniform convention across every implementation
//! (spec/api.md): the state is an opaque handle, updated in place — calls
//! return only their results (`record = gittins.decide(state, ...)`), and
//! every alias of the handle observes the current state. Snapshot or
//! persist it with `serialize`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyString, PyTuple};

use gittins_core::api;
use gittins_core::decide::{BanditState as CoreState, DecisionRecord as CoreRecord};
use gittins_core::encoding::Value;
use gittins_core::error::Error;
use gittins_core::exploration::DEFAULT_EPSILON;
use gittins_core::ledger::Resolution as CoreResolution;
use gittins_core::model::DEFAULT_FORGETTING;

// The reference raises ValueError everywhere; the core's Error mirrors the
// messages, so the binding maps 1:1.
fn value_error(e: Error) -> PyErr {
    PyValueError::new_err(e.message().to_string())
}

/// A feature dict as the core's (name, Value) pairs — the reference's
/// `feature_tokens` contract: strings are categorical, ints/floats/bools
/// numeric, None absent, anything else a ValueError naming the feature.
fn features(dict: &Bound<'_, PyDict>) -> PyResult<Vec<(String, Value)>> {
    let mut out = Vec::with_capacity(dict.len());
    for (name, value) in dict.iter() {
        let name: String = name
            .extract()
            .map_err(|_| PyValueError::new_err("feature names must be strings"))?;
        let value = if value.is_none() {
            Value::None
        } else if value.downcast::<PyString>().is_ok() {
            Value::Str(value.extract()?)
        } else if value.downcast::<PyBool>().is_ok() {
            Value::Num(if value.is_truthy()? { 1.0 } else { 0.0 })
        } else if value.downcast::<PyInt>().is_ok() || value.downcast::<PyFloat>().is_ok() {
            Value::Num(value.extract()?)
        } else {
            return Err(PyValueError::new_err(format!(
                "feature '{name}' has unsupported type {}",
                value.get_type().name()?
            )));
        };
        out.push((name, value));
    }
    Ok(out)
}

/// The bandit's state: an opaque handle. Treat it linearly — always rebind
/// the returned state — and persist it with serialize/deserialize.
#[pyclass(name = "BanditState", module = "gittins")]
struct BanditState {
    inner: CoreState,
}

/// One decision, self-contained: what was chosen, with what probability,
/// under which id — everything outcome reporting and offline evaluation
/// need.
#[pyclass(frozen, name = "DecisionRecord", module = "gittins")]
struct DecisionRecord {
    #[pyo3(get)]
    decision_id: String,
    #[pyo3(get)]
    t: f64,
    #[pyo3(get)]
    candidate_hash: u64,
    #[pyo3(get)]
    chosen: usize,
    #[pyo3(get)]
    features: Vec<(usize, f64)>,
    #[pyo3(get)]
    propensity: f64,
    #[pyo3(get)]
    model_version: u64,
    #[pyo3(get)]
    salt: String,
}

impl From<CoreRecord> for DecisionRecord {
    fn from(r: CoreRecord) -> DecisionRecord {
        DecisionRecord {
            decision_id: r.decision_id,
            t: r.t,
            candidate_hash: r.candidate_hash,
            chosen: r.chosen,
            features: r.features,
            propensity: r.propensity,
            model_version: r.model_version,
            salt: r.salt,
        }
    }
}

/// One resolution event: how a decision left the ledger. `reward` is what
/// the model trained with; None for censored.
#[pyclass(frozen, name = "Resolution", module = "gittins")]
struct Resolution {
    #[pyo3(get)]
    decision_id: String,
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    reward: Option<f64>,
}

impl From<CoreResolution> for Resolution {
    fn from(r: CoreResolution) -> Resolution {
        Resolution {
            decision_id: r.decision_id,
            kind: r.kind.as_str().to_string(),
            reward: r.reward,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (bits, horizon, default_reward = 0.0, epsilon = DEFAULT_EPSILON, forgetting = DEFAULT_FORGETTING))]
fn create(
    bits: u32,
    horizon: f64,
    default_reward: f64,
    epsilon: f64,
    forgetting: f64,
) -> PyResult<BanditState> {
    Ok(BanditState {
        inner: api::create(bits, horizon, default_reward, epsilon, forgetting)
            .map_err(value_error)?,
    })
}

#[pyfunction]
fn model_bits(state: &Bound<'_, BanditState>) -> PyResult<u32> {
    api::model_bits(&state.borrow().inner).map_err(value_error)
}

#[pyfunction]
fn decide(
    state: &Bound<'_, BanditState>,
    context: &Bound<'_, PyDict>,
    candidates: Vec<(String, Bound<'_, PyDict>)>,
    t: f64,
    salt: &str,
) -> PyResult<DecisionRecord> {
    let context = features(context)?;
    let candidates: Vec<(String, Vec<(String, Value)>)> = candidates
        .into_iter()
        .map(|(arm_id, action)| Ok((arm_id, features(&action)?)))
        .collect::<PyResult<_>>()?;
    let record = api::decide(&mut state.borrow_mut().inner, &context, &candidates, t, salt)
        .map_err(value_error)?;
    Ok(record.into())
}

/// The resolution, or None if the id is unknown or already resolved.
#[pyfunction]
fn learn(state: &Bound<'_, BanditState>, decision_id: &str, reward: f64) -> Option<Resolution> {
    api::learn(&mut state.borrow_mut().inner, decision_id, reward).map(Resolution::from)
}

/// The resolution, or None if the id is unknown or already resolved.
#[pyfunction]
fn censor(state: &Bound<'_, BanditState>, decision_id: &str) -> Option<Resolution> {
    api::censor(&mut state.borrow_mut().inner, decision_id).map(Resolution::from)
}

/// Every decision past its horizon at time `t`, resolved as expired, in
/// ledger order.
#[pyfunction]
fn expire<'py>(
    py: Python<'py>,
    state: &Bound<'py, BanditState>,
    t: f64,
) -> PyResult<Bound<'py, PyTuple>> {
    let resolutions = api::expire(&mut state.borrow_mut().inner, t);
    PyTuple::new(py, resolutions.into_iter().map(Resolution::from))
}

#[pyfunction]
/// The current state as one plain hex string — storable anywhere text
/// goes, with no byte handling on the caller's side.
fn serialize(state: &Bound<'_, BanditState>) -> String {
    api::serialize(&state.borrow().inner)
}

#[pyfunction]
fn deserialize(data: &str) -> PyResult<BanditState> {
    Ok(BanditState {
        inner: api::deserialize(data).map_err(value_error)?,
    })
}

#[pymodule]
fn gittins(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BanditState>()?;
    m.add_class::<DecisionRecord>()?;
    m.add_class::<Resolution>()?;
    m.add_function(wrap_pyfunction!(create, m)?)?;
    m.add_function(wrap_pyfunction!(decide, m)?)?;
    m.add_function(wrap_pyfunction!(learn, m)?)?;
    m.add_function(wrap_pyfunction!(censor, m)?)?;
    m.add_function(wrap_pyfunction!(expire, m)?)?;
    m.add_function(wrap_pyfunction!(serialize, m)?)?;
    m.add_function(wrap_pyfunction!(deserialize, m)?)?;
    m.add_function(wrap_pyfunction!(model_bits, m)?)?;
    Ok(())
}
