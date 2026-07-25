//! The experience log and offline policy evaluation — the port of
//! `gittins_reference/ope.py`.
//!
//! Same shapes, same walk, same fixed-order arithmetic: `read_log`
//! streams one event per line (plain files read line by line, gzipped
//! files through the streaming decompressor — a log's size is bounded
//! by disk, not memory), `verify` collects every semantic violation as
//! the reference's exact strings, `evaluate` is the progressive
//! IPS/SNIPS walk, and `replay` rebuilds a deployable state — each one
//! pass over any event iterator, holding only its own bookkeeping.
//! Feature dicts keep their file order end to end (serde_json's
//! preserve_order): iteration order is semantic under hashed encoding,
//! where per-slot accumulation is order-sensitive at the last bit.

use std::collections::{HashMap, HashSet};
use std::io::BufRead;

use serde_json::Value as Json;

use gittins_core::decide::{candidate_set_hash, new_bandit, BanditState, DecisionRecord};
use gittins_core::encoding::{encode, Features, Value};
use gittins_core::error::Error;
use gittins_core::exploration::epsilon_greedy_probabilities;
use gittins_core::model::{estimate_factored, factorize, new_model, update};

const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];
const REWARDED: &str = "rewarded";
const EXPIRED: &str = "expired";

/// Feature dicts stay raw JSON until an encode site needs them: the
/// reference parses shapes strictly but leaves feature *values* to the
/// encoding layer's contract, so an unsupported value is an encode
/// failure, not a parse failure.
pub type RawFeatures = Vec<(String, Json)>;

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionEvent {
    pub line: usize,
    pub bits: i64,
    pub context: RawFeatures,
    pub candidates: Vec<(String, RawFeatures)>,
    pub record: DecisionRecord,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolutionEvent {
    pub line: usize,
    pub decision_id: String,
    pub kind: String,
    pub reward: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Decision(DecisionEvent),
    Resolution(ResolutionEvent),
}

/// One evaluation's estimates and the diagnostics they must never be
/// read without. Estimates are None when nothing resolved.
#[derive(Clone, Debug, PartialEq)]
pub struct OpeReport {
    pub decisions: u64,
    pub resolved: u64,
    pub unresolved: u64,
    pub logged_mean: Option<f64>,
    pub ips: Option<f64>,
    pub snips: Option<f64>,
    pub ess: Option<f64>,
    pub max_weight: Option<f64>,
}

/// One log file's events as a lazy stream: lines are read (and, for
/// gzip, decompressed) one at a time, never the file whole. Gzip is
/// detected by the magic bytes, never the file name. Each item is one
/// event or that line's error; consume once.
pub fn read_log(path: &str) -> Result<Box<dyn Iterator<Item = Result<Event, String>>>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let mut reader = std::io::BufReader::new(file);
    let head = reader.fill_buf().map_err(|e| format!("cannot read {path}: {e}"))?;
    if head.starts_with(&GZIP_MAGIC) {
        let decoder = std::io::BufReader::new(flate2::read::GzDecoder::new(reader));
        Ok(Box::new(events_from(decoder)))
    } else {
        Ok(Box::new(events_from(reader)))
    }
}

/// The streaming core of `read_log`: one parsed event per nonempty
/// line, yielded as lines arrive from any buffered reader.
fn events_from<R: BufRead>(reader: R) -> impl Iterator<Item = Result<Event, String>> {
    reader.lines().enumerate().filter_map(|(i, line)| {
        let n = i + 1;
        match line {
            Err(e) => Some(Err(format!("line {n}: read failed: {e}"))),
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(parse_event(n, trimmed))
                }
            }
        }
    })
}

fn number(v: &Json) -> Option<f64> {
    v.as_f64() // numbers only: bools and strings are None here
}

fn raw_features(v: &Json) -> Option<RawFeatures> {
    Some(v.as_object()?.iter().map(|(name, value)| (name.clone(), value.clone())).collect())
}

fn record(obj: &Json) -> Option<DecisionRecord> {
    let features = obj.get("features")?.as_array()?;
    let mut pairs: Features = Vec::with_capacity(features.len());
    for pair in features {
        let pair = pair.as_array()?;
        if pair.len() != 2 {
            return None;
        }
        pairs.push((pair[0].as_u64()? as usize, number(&pair[1])?));
    }
    Some(DecisionRecord {
        decision_id: obj.get("decision_id")?.as_str()?.to_string(),
        t: number(obj.get("t")?)?,
        candidate_hash: obj.get("candidate_hash")?.as_u64()?,
        chosen: obj.get("chosen")?.as_u64()? as usize,
        features: pairs,
        propensity: number(obj.get("propensity")?)?,
        model_version: obj.get("model_version")?.as_u64()?,
        salt: obj.get("salt")?.as_str()?.to_string(),
    })
}

fn decision(n: usize, obj: &Json) -> Option<Event> {
    let candidates = obj.get("candidates")?.as_array()?;
    let mut parsed = Vec::with_capacity(candidates.len());
    for pair in candidates {
        let pair = pair.as_array()?;
        if pair.len() != 2 {
            return None;
        }
        parsed.push((pair[0].as_str()?.to_string(), raw_features(&pair[1])?));
    }
    Some(Event::Decision(DecisionEvent {
        line: n,
        bits: obj.get("bits")?.as_i64()?,
        context: raw_features(obj.get("context")?)?,
        candidates: parsed,
        record: record(obj.get("record")?)?,
    }))
}

fn resolution(n: usize, obj: &Json) -> Option<Event> {
    let reward = obj.get("reward")?;
    let reward = if reward.is_null() { None } else { Some(number(reward)?) };
    Some(Event::Resolution(ResolutionEvent {
        line: n,
        decision_id: obj.get("decision_id")?.as_str()?.to_string(),
        kind: obj.get("kind")?.as_str()?.to_string(),
        reward,
    }))
}

/// One line as one event; structural problems are hard errors naming
/// the line, semantic problems are `verify`'s job.
fn parse_event(n: usize, text: &str) -> Result<Event, String> {
    let value: Json =
        serde_json::from_str(text).map_err(|_| format!("line {n}: not valid JSON"))?;
    let kind = value.get("event").and_then(Json::as_str);
    if !value.is_object() || kind.is_none() {
        return Err(format!("line {n}: each event must be a JSON object with an 'event' kind"));
    }
    match kind.unwrap() {
        "decision" => decision(n, &value).ok_or(format!("line {n}: malformed decision event")),
        "resolution" => {
            resolution(n, &value).ok_or(format!("line {n}: malformed resolution event"))
        }
        _ => Err(format!("line {n}: unknown event kind")),
    }
}

/// In-memory convenience over the same streamed parse, collected —
/// programmatic and test use; the CLI itself streams via `read_log`.
pub fn parse_log(text: &str) -> Result<Vec<Event>, String> {
    events_from(text.as_bytes()).collect()
}

/// A raw feature dict as the encoding layer's typed pairs, in file
/// order; unsupported values fail like any other encode failure.
fn typed(raw: &RawFeatures) -> Result<Vec<(String, Value)>, Error> {
    raw.iter()
        .map(|(name, v)| {
            let value = match v {
                Json::Null => Value::None,
                Json::Bool(b) => Value::Num(if *b { 1.0 } else { 0.0 }),
                Json::Number(_) => Value::Num(
                    number(v).ok_or_else(|| Error::new(format!("feature '{name}' has unsupported type")))?,
                ),
                Json::String(s) => Value::Str(s.clone()),
                _ => return Err(Error::new(format!("feature '{name}' has unsupported type"))),
            };
            Ok((name.clone(), value))
        })
        .collect()
}

fn encode_candidates(e: &DecisionEvent, bits: u32) -> Result<Vec<Features>, Error> {
    let context = typed(&e.context)?;
    e.candidates
        .iter()
        .map(|(arm_id, action)| encode(&context, arm_id, &typed(action)?, bits))
        .collect()
}

/// Every semantic violation, as the reference's exact `line N: ...`
/// strings in event order; empty means the log is trustworthy input for
/// `evaluate` and `replay`. One pass over any event stream; a stream
/// (parse) error propagates as `Err`.
pub fn verify<I>(events: I) -> Result<Vec<String>, String>
where
    I: IntoIterator<Item = Result<Event, String>>,
{
    let mut problems = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut resolved: HashSet<String> = HashSet::new();
    let mut last_t: HashMap<String, f64> = HashMap::new();
    for event in events {
        match &event? {
            Event::Decision(e) => {
                let r = &e.record;
                let at = format!("line {}: decision {}", e.line, r.decision_id);
                if !seen.insert(r.decision_id.clone()) {
                    problems.push(format!(
                        "line {}: duplicate decision id '{}'",
                        e.line, r.decision_id
                    ));
                }
                if !(1..=24).contains(&e.bits) {
                    problems.push(format!("{at}: bits must be between 1 and 24"));
                    continue;
                }
                if r.chosen >= e.candidates.len() {
                    problems.push(format!("{at}: chosen index out of range"));
                    continue;
                }
                if !(r.propensity > 0.0 && r.propensity <= 1.0) {
                    problems.push(format!("{at}: propensity must be in (0, 1]"));
                }
                let encoded = match encode_candidates(e, e.bits as u32) {
                    Ok(encoded) => encoded,
                    Err(_) => {
                        problems.push(format!("{at}: candidates failed to encode"));
                        continue;
                    }
                };
                if candidate_set_hash(&encoded, 1usize << e.bits) != r.candidate_hash {
                    problems.push(format!("{at}: candidates do not match the logged candidate_hash"));
                }
                if encoded[r.chosen] != r.features {
                    problems.push(format!("{at}: chosen features do not match the logged candidates"));
                }
                if last_t.get(&r.salt).is_some_and(|&t| r.t < t) {
                    problems.push(format!("{at}: t decreases within salt '{}'", r.salt));
                }
                let t = last_t.get(&r.salt).map_or(r.t, |&t| t.max(r.t));
                last_t.insert(r.salt.clone(), t);
            }
            Event::Resolution(e) => {
                let at = format!("line {}: resolution for", e.line);
                if !seen.contains(&e.decision_id) {
                    problems.push(format!("{at} unknown decision '{}'", e.decision_id));
                    continue;
                }
                if !resolved.insert(e.decision_id.clone()) {
                    problems.push(format!(
                        "line {}: second resolution for decision '{}'",
                        e.line, e.decision_id
                    ));
                    continue;
                }
                if !(e.kind == REWARDED || e.kind == EXPIRED) {
                    problems.push(format!("{at} {}: unknown kind '{}'", e.decision_id, e.kind));
                } else if !e.reward.is_some_and(f64::is_finite) {
                    problems.push(format!("{at} {}: reward must be a finite number", e.decision_id));
                }
            }
        }
    }
    Ok(problems)
}

/// Progressive IPS/SNIPS over the log for one candidate configuration
/// of the built-in engine. One pass over any event
/// stream. Strict where `verify` is lenient — run `verify` first.
pub fn evaluate<I>(
    events: I,
    bits: u32,
    epsilon: f64,
    forgetfulness: f64,
) -> Result<OpeReport, String>
where
    I: IntoIterator<Item = Result<Event, String>>,
{
    if !(1..=24).contains(&bits) {
        return Err("bits must be between 1 and 24".to_string());
    }
    let mut model =
        new_model(1usize << bits, forgetfulness, 1.0).map_err(|e| e.message().to_string())?;
    let mut pending: HashMap<String, (Features, f64)> = HashMap::new();
    let (mut decisions, mut resolved) = (0u64, 0u64);
    let (mut sum_r, mut sum_w, mut sum_wr, mut sum_w2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut max_weight = 0.0f64;
    for event in events {
        match &event? {
            Event::Decision(e) => {
                decisions += 1;
                let r = &e.record;
                if e.candidates.is_empty() {
                    return Err(format!("line {}: need at least one candidate", e.line));
                }
                if r.chosen >= e.candidates.len() {
                    return Err(format!("line {}: chosen index out of range", e.line));
                }
                if !(r.propensity > 0.0 && r.propensity <= 1.0) {
                    return Err(format!("line {}: propensity must be in (0, 1]", e.line));
                }
                let encoded = encode_candidates(e, bits)
                    .map_err(|err| format!("line {}: {}", e.line, err.message()))?;
                let mut factored = factorize(&model);
                let estimates: Vec<f64> =
                    encoded.iter().map(|x| estimate_factored(&mut factored, x)).collect();
                let p = epsilon_greedy_probabilities(&estimates, epsilon)
                    .map_err(|err| err.message().to_string())?;
                let w = p[r.chosen] / r.propensity;
                pending.insert(r.decision_id.clone(), (encoded[r.chosen].clone(), w));
            }
            Event::Resolution(e) => {
                let Some((x, w)) = pending.remove(&e.decision_id) else {
                    continue;
                };
                if !(e.kind == REWARDED || e.kind == EXPIRED) {
                    return Err(format!("line {}: unknown resolution kind", e.line));
                }
                let Some(reward) = e.reward.filter(|r| r.is_finite()) else {
                    return Err(format!("line {}: reward must be a finite number", e.line));
                };
                resolved += 1;
                sum_r += reward;
                sum_w += w;
                sum_wr += w * reward;
                sum_w2 += w * w;
                if w > max_weight {
                    max_weight = w;
                }
                update(&mut model, &x, reward);
            }
        }
    }
    if resolved == 0 {
        return Ok(OpeReport {
            decisions,
            resolved: 0,
            unresolved: pending.len() as u64,
            logged_mean: None,
            ips: None,
            snips: None,
            ess: None,
            max_weight: None,
        });
    }
    Ok(OpeReport {
        decisions,
        resolved,
        unresolved: pending.len() as u64,
        logged_mean: Some(sum_r / resolved as f64),
        ips: Some(sum_wr / resolved as f64),
        snips: if sum_w > 0.0 { Some(sum_wr / sum_w) } else { None },
        ess: if sum_w2 > 0.0 { Some((sum_w * sum_w) / sum_w2) } else { None },
        max_weight: Some(max_weight),
    })
}

/// The fleet-pooling rebuild: a fresh, deployable state
/// whose model absorbed every rewarded/expired resolution's re-encoded
/// chosen candidate and logged reward, in log order. One pass over any
/// event stream.
pub fn replay<I>(
    events: I,
    bits: u32,
    horizon: f64,
    default_reward: f64,
    epsilon: f64,
    forgetfulness: f64,
) -> Result<BanditState, String>
where
    I: IntoIterator<Item = Result<Event, String>>,
{
    if !(1..=24).contains(&bits) {
        return Err("bits must be between 1 and 24".to_string());
    }
    let mut state = new_bandit(1usize << bits, horizon, default_reward, epsilon, forgetfulness)
        .map_err(|e| e.message().to_string())?;
    let mut pending: HashMap<String, Features> = HashMap::new();
    let mut absorbed = 0u64;
    for event in events {
        match &event? {
            Event::Decision(e) => {
                let r = &e.record;
                if r.chosen >= e.candidates.len() {
                    return Err(format!("line {}: chosen index out of range", e.line));
                }
                let context = typed(&e.context)
                    .map_err(|err| format!("line {}: {}", e.line, err.message()))?;
                let (arm_id, action) = &e.candidates[r.chosen];
                let action = typed(action)
                    .map_err(|err| format!("line {}: {}", e.line, err.message()))?;
                let x = encode(&context, arm_id, &action, bits)
                    .map_err(|err| format!("line {}: {}", e.line, err.message()))?;
                pending.insert(r.decision_id.clone(), x);
            }
            Event::Resolution(e) => {
                let Some(x) = pending.remove(&e.decision_id) else {
                    continue;
                };
                if !(e.kind == REWARDED || e.kind == EXPIRED) {
                    return Err(format!("line {}: unknown resolution kind", e.line));
                }
                let Some(reward) = e.reward.filter(|r| r.is_finite()) else {
                    return Err(format!("line {}: reward must be a finite number", e.line));
                };
                update(&mut state.model, &x, reward);
                absorbed += 1;
            }
        }
    }
    state.model_version = absorbed;
    Ok(state)
}
