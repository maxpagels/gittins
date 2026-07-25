//! The Python binding: `gittins_reference.api` mirrored name for name over
//! the Rust core's `api` module, so the reference's
//! usage docs and this module are interchangeable. The acceptance gate is
//! the golden `api` section replayed through this module alone
//! (`tests/test_binding.py`).
//!
//! The one performance rule (PROGRESS, PR 20): one boundary crossing per
//! decision. `decide` takes the context dict and the whole candidate list
//! and encode → score → sample all happen inside Rust; nothing calls back
//! into Python per candidate or per feature. The BYO callbacks
//! keep that rule: `score`/`explore`/`train` each cross the
//! boundary once per call — with all candidates, all estimates, or the one
//! resolved record — never per candidate. A callback's own Python
//! exception is re-raised as itself; a malformed callback *result* raises
//! ValueError with the reference's message.
//!
//! State handling, the uniform convention across every implementation:
//! the state is an opaque handle, updated in place — calls
//! return only their results (`record = gittins.decide(state, ...)`), and
//! every alias of the handle observes the current state. Snapshot or
//! persist it with `serialize`.

use std::cell::RefCell;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyString, PyTuple};

use gittins_core::api;
use gittins_core::encoding::Features;
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
/// need. Records returned by `decide` additionally carry the inputs the
/// decision was made over (`bits`, `context`, `candidates` — the
/// caller's own objects), so they log as-is via `log_line`; records that
/// crossed the state boundary (a `train` callback's) carry None there —
/// the ledger keeps only the compact record.
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
    #[pyo3(get)]
    bits: Option<u32>,
    #[pyo3(get)]
    context: Option<Py<PyAny>>,
    #[pyo3(get)]
    candidates: Option<Py<PyAny>>,
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
            bits: None,
            context: None,
            candidates: None,
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
#[pyo3(signature = (bits, horizon, default_reward = 0.0, epsilon = DEFAULT_EPSILON, forgetfulness = DEFAULT_FORGETTING))]
fn create(
    bits: u32,
    horizon: f64,
    default_reward: f64,
    epsilon: f64,
    forgetfulness: f64,
) -> PyResult<BanditState> {
    Ok(BanditState {
        inner: api::create(bits, horizon, default_reward, epsilon, forgetfulness)
            .map_err(value_error)?,
    })
}

#[pyfunction]
fn model_bits(state: &Bound<'_, BanditState>) -> PyResult<u32> {
    api::model_bits(&state.borrow().inner).map_err(value_error)
}

/// A stash for the one Python exception a callback may raise while Rust is
/// on the stack: the core sees an `Error` (so it unwinds cleanly with no
/// state half-changed), and the original exception — traceback intact — is
/// re-raised at the boundary.
struct Caught(RefCell<Option<PyErr>>);

impl Caught {
    fn new() -> Caught {
        Caught(RefCell::new(None))
    }

    fn stash(&self, e: PyErr, what: &str) -> Error {
        *self.0.borrow_mut() = Some(e);
        Error::new(&format!("{what} callback raised"))
    }

    fn rethrow<T>(&self, result: Result<T, Error>) -> PyResult<T> {
        if let Some(e) = self.0.borrow_mut().take() {
            return Err(e);
        }
        result.map_err(value_error)
    }
}

/// The BYO train callback as the core's shape: called with
/// the resolved decision's record and the reward, once per resolution.
fn train_callback<'a>(
    py: Python<'a>,
    f: &'a Bound<'a, PyAny>,
    caught: &'a Caught,
) -> impl FnMut(&CoreRecord, f64) -> Result<(), Error> + 'a {
    move |record: &CoreRecord, reward: f64| {
        Py::new(py, DecisionRecord::from(record.clone()))
            .and_then(|rec| f.call1((rec, reward)).map(|_| ()))
            .map_err(|e| caught.stash(e, "train"))
    }
}

#[pyfunction]
#[pyo3(signature = (state, context, candidates, t, salt, score=None, explore=None))]
fn decide(
    state: &Bound<'_, BanditState>,
    context: &Bound<'_, PyDict>,
    candidates: &Bound<'_, PyAny>,
    t: f64,
    salt: &str,
    score: Option<&Bound<'_, PyAny>>,
    explore: Option<&Bound<'_, PyAny>>,
) -> PyResult<DecisionRecord> {
    let raw: Vec<(String, Bound<'_, PyDict>)> = candidates.extract()?;
    let parsed_context = features(context)?;
    let parsed: Vec<(String, Vec<(String, Value)>)> = raw
        .into_iter()
        .map(|(arm_id, action)| Ok((arm_id, features(&action)?)))
        .collect::<PyResult<_>>()?;
    let caught = Caught::new();
    // The callbacks receive the very objects the caller passed;
    // the encoded candidates the core offers are not re-surfaced to Python.
    let mut score_cb = score.map(|f| {
        |_: &[Features]| -> Result<Vec<f64>, Error> {
            let result = f.call1((context, candidates)).map_err(|e| caught.stash(e, "score"))?;
            result.extract::<Vec<f64>>().map_err(|_| {
                Error::new("score must return one finite estimate per candidate")
            })
        }
    });
    let mut explore_cb = explore.map(|f| {
        |estimates: &[f64], epsilon: f64| -> Result<Vec<f64>, Error> {
            let result = f
                .call1((estimates.to_vec(), epsilon))
                .map_err(|e| caught.stash(e, "explore"))?;
            result.extract::<Vec<f64>>().map_err(|_| {
                Error::new("explore must return one probability per candidate")
            })
        }
    });
    let bits = api::model_bits(&state.borrow().inner).map_err(value_error)?;
    let result = api::decide(
        &mut state.borrow_mut().inner,
        &parsed_context,
        &parsed,
        t,
        salt,
        score_cb.as_mut().map(|c| c as _),
        explore_cb.as_mut().map(|c| c as _),
    );
    let mut record: DecisionRecord = caught.rethrow(result)?.into();
    // The returned record carries the very objects the caller passed —
    // attached at the only moment they exist.
    record.bits = Some(bits);
    record.context = Some(context.clone().into_any().unbind());
    record.candidates = Some(candidates.clone().unbind());
    Ok(record)
}

/// One canonical experience-log line for a decision record
/// or resolution — append it to your log verbatim; it is exactly what
/// the `gittins` CLI's verify/eval/replay consume. Decision records must
/// be the ones `decide` returned (they carry the inputs; a `train`
/// callback's record does not — its inputs are already in your log).
#[pyfunction]
fn log_line(py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<String> {
    let json = py.import("json")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("separators", (",", ":"))?;
    let line = PyDict::new(py);
    if let Ok(record) = item.downcast::<DecisionRecord>() {
        let r = record.get();
        let (Some(bits), Some(context), Some(candidates)) = (r.bits, &r.context, &r.candidates)
        else {
            return Err(PyValueError::new_err(
                "log_line needs the record decide returned (it carries the inputs)",
            ));
        };
        line.set_item("event", "decision")?;
        line.set_item("bits", bits)?;
        line.set_item("context", context)?;
        line.set_item("candidates", candidates)?;
        let compact = PyDict::new(py);
        compact.set_item("decision_id", &r.decision_id)?;
        compact.set_item("t", r.t)?;
        compact.set_item("candidate_hash", r.candidate_hash)?;
        compact.set_item("chosen", r.chosen)?;
        compact.set_item("features", &r.features)?;
        compact.set_item("propensity", r.propensity)?;
        compact.set_item("model_version", r.model_version)?;
        compact.set_item("salt", &r.salt)?;
        line.set_item("record", compact)?;
    } else if let Ok(resolution) = item.downcast::<Resolution>() {
        let r = resolution.get();
        line.set_item("event", "resolution")?;
        line.set_item("decision_id", &r.decision_id)?;
        line.set_item("kind", &r.kind)?;
        line.set_item("reward", r.reward)?;
    } else {
        return Err(PyValueError::new_err("log_line takes a DecisionRecord or Resolution"));
    }
    json.getattr("dumps")?.call((line,), Some(&kwargs))?.extract()
}

/// The resolution, or None if the id is unknown or already resolved.
/// The engine classifies against the horizon at time `t`: a reward
/// arriving at or past the record's `t + horizon` resolves as expired
/// with the default reward. `train` is the BYO training tap: it replaces
/// the built-in update and fires after the resolution commits, exactly
/// once, with the classified reward.
#[pyfunction]
#[pyo3(signature = (state, decision_id, reward, t, train=None))]
fn learn(
    py: Python<'_>,
    state: &Bound<'_, BanditState>,
    decision_id: &str,
    reward: f64,
    t: f64,
    train: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Resolution>> {
    let caught = Caught::new();
    let mut train_cb = train.map(|f| train_callback(py, f, &caught));
    let result = api::learn(
        &mut state.borrow_mut().inner,
        decision_id,
        reward,
        t,
        train_cb.as_mut().map(|c| c as _),
    );
    caught.rethrow(result).map(|o| o.map(Resolution::from))
}

/// The resolution, or None if the id is unknown or already resolved.
#[pyfunction]
fn censor(state: &Bound<'_, BanditState>, decision_id: &str) -> Option<Resolution> {
    api::censor(&mut state.borrow_mut().inner, decision_id).map(Resolution::from)
}

/// Every decision past its horizon at time `t`, resolved as expired, in
/// ledger order. `train` as on `learn`, fired per due record.
#[pyfunction]
#[pyo3(signature = (state, t, train=None))]
fn expire<'py>(
    py: Python<'py>,
    state: &Bound<'py, BanditState>,
    t: f64,
    train: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyTuple>> {
    let caught = Caught::new();
    let mut train_cb = train.map(|f| train_callback(py, f, &caught));
    let result = api::expire(
        &mut state.borrow_mut().inner,
        t,
        train_cb.as_mut().map(|c| c as _),
    );
    let resolutions = caught.rethrow(result)?;
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
    m.add_function(wrap_pyfunction!(log_line, m)?)?;
    Ok(())
}
