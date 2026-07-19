//! Verify the Rust core against `spec/golden.json`, bit for bit — every
//! section the reference generates (`gittins_reference/golden.py`), plus a
//! full replay of the end-to-end episode, the Phase 0 exit criterion: an
//! independent decide/learn implementation is done when it reproduces the
//! episode section exactly. Floats are compared via `to_bits`; hashes and
//! RNG outputs as u64.

mod json;

use json::Json;

use gittins_core::api;
use gittins_core::decide::{candidate_set_hash, decide, new_bandit, BanditState, DecisionRecord};
use gittins_core::encoding::{encode, feature_tokens, pair_hash, Features, Value};
use gittins_core::exploration::{epsilon_greedy_probabilities, sample_index, DEFAULT_EPSILON};
use gittins_core::ledger::{censor, expire, learn, Resolution};
use gittins_core::model::{new_model, predict, update, DEFAULT_FORGETTING};
use gittins_core::rng::{derive_key, random_u64, random_unit};
use gittins_core::state::{deserialize, serialize, FORMAT_VERSION};

fn golden() -> Json {
    Json::parse(include_str!("../../../spec/golden.json"))
}

fn section(name: &str) -> Json {
    golden().get("sections").get(name).clone()
}

/// Bitwise float equality against a corpus number.
fn assert_bits(actual: f64, expected: &Json, what: &str) {
    let expected = expected.f64_();
    assert!(
        actual.to_bits() == expected.to_bits(),
        "{what}: got {actual:?}, corpus has {expected:?}"
    );
}

/// A corpus [[index, value], ...] list against computed sparse pairs.
fn assert_pairs(actual: &Features, expected: &Json, what: &str) {
    let expected = expected.arr();
    assert!(
        actual.len() == expected.len(),
        "{what}: {} pairs, corpus has {}",
        actual.len(),
        expected.len()
    );
    for (i, (&(j, v), pair)) in actual.iter().zip(expected).enumerate() {
        assert!(j == pair.arr()[0].usize_(), "{what}[{i}]: index mismatch");
        assert_bits(v, &pair.arr()[1], &format!("{what}[{i}] value"));
    }
}

/// A corpus feature dict ({name: string | number | bool | null}) as the
/// core's (name, Value) pairs. Booleans count as numeric, null as absent —
/// the reference's `feature_tokens` contract.
fn values(obj: &Json) -> Vec<(String, Value)> {
    obj.entries()
        .iter()
        .map(|(name, v)| {
            let value = match v {
                Json::Null => Value::None,
                Json::Bool(b) => Value::Num(if *b { 1.0 } else { 0.0 }),
                Json::Num(_) => Value::Num(v.f64_()),
                Json::Str(s) => Value::Str(s.clone()),
                _ => panic!("unsupported feature value"),
            };
            (name.clone(), value)
        })
        .collect()
}

#[test]
fn rng_section() {
    let s = section("rng");
    for case in s.get("derive_key").arr() {
        let key = derive_key(case.get("decision_id").str_(), case.get("salt").str_());
        assert!(key == case.get("key").u64_(), "derive_key mismatch");
    }
    let stream = s.get("stream");
    let key = derive_key(stream.get("decision_id").str_(), stream.get("salt").str_());
    assert!(key == stream.get("key").u64_());
    for (c, expected) in stream.get("u64").arr().iter().enumerate() {
        assert!(random_u64(key, c as u64) == expected.u64_(), "u64[{c}]");
    }
    for (c, expected) in stream.get("unit").arr().iter().enumerate() {
        assert_bits(random_unit(key, c as u64), expected, &format!("unit[{c}]"));
    }
    assert!(random_u64(key, 1 << 32) == stream.get("u64_at_2_32").u64_());
}

#[test]
fn model_section() {
    let s = section("model");
    let mut m = new_model(
        s.get("dim").usize_(),
        s.get("forgetfulness").f64_(),
        s.get("ridge").f64_(),
    )
    .unwrap();
    let states = s.get("states").arr();
    for (i, u) in s.get("updates").arr().iter().enumerate() {
        let x: Features = u
            .get("x")
            .arr()
            .iter()
            .map(|p| (p.arr()[0].usize_(), p.arr()[1].f64_()))
            .collect();
        update(&mut m, &x, u.get("reward").f64_());
        let expected = &states[i];
        assert_bits(m.scale, expected.get("scale"), &format!("state[{i}].scale"));
        for (j, e) in expected.get("xx").arr().iter().enumerate() {
            assert_bits(m.xx[j], e, &format!("state[{i}].xx[{j}]"));
        }
        for (j, e) in expected.get("xy").arr().iter().enumerate() {
            assert_bits(m.xy[j], e, &format!("state[{i}].xy[{j}]"));
        }
    }
    let probe = s.get("predict");
    let x: Features = probe
        .get("x")
        .arr()
        .iter()
        .map(|p| (p.arr()[0].usize_(), p.arr()[1].f64_()))
        .collect();
    let (estimate, uncertainty) = predict(&m, &x);
    assert_bits(estimate, probe.get("estimate"), "predict estimate");
    assert_bits(uncertainty, probe.get("uncertainty"), "predict uncertainty");
}

#[test]
fn exploration_section() {
    let s = section("exploration");
    let estimates: Vec<f64> = s.get("estimates").arr().iter().map(Json::f64_).collect();
    let p = epsilon_greedy_probabilities(&estimates, s.get("epsilon").f64_()).unwrap();
    for (i, e) in s.get("probabilities").arr().iter().enumerate() {
        assert_bits(p[i], e, &format!("probabilities[{i}]"));
    }
    let key = s.get("key").u64_();
    for (c, expected) in s.get("samples").arr().iter().enumerate() {
        assert!(
            sample_index(&p, key, c as u64) == expected.usize_(),
            "samples[{c}]"
        );
    }
}

#[test]
fn encoding_section() {
    let s = section("encoding");
    for case in s.get("tokens").arr() {
        let tokens = feature_tokens(case.get("namespace").str_(), &values(case.get("values")));
        let expected = case.get("tokens").arr();
        assert!(tokens.len() == expected.len(), "token count mismatch");
        for ((token, contribution), e) in tokens.iter().zip(expected) {
            assert!(token == e.arr()[0].str_(), "token mismatch");
            assert_bits(*contribution, &e.arr()[1], &format!("contribution of {token}"));
        }
    }
    for case in s.get("pair_hashes").arr() {
        assert!(
            pair_hash(case.get("left").str_(), case.get("right").str_())
                == case.get("hash").u64_(),
            "pair_hash mismatch"
        );
    }
    for case in s.get("encode").arr() {
        let pairs = encode(
            &values(case.get("context")),
            case.get("arm_id").str_(),
            &values(case.get("action")),
            case.get("bits").u64_() as u32,
        )
        .unwrap();
        assert_pairs(&pairs, case.get("pairs"), "encode pairs");
    }
}

fn assert_record(record: &DecisionRecord, expected: &Json, what: &str) {
    assert!(
        record.decision_id == expected.get("decision_id").str_(),
        "{what}: decision_id"
    );
    assert_bits(record.t, expected.get("t"), &format!("{what}: t"));
    assert!(
        record.candidate_hash == expected.get("candidate_hash").u64_(),
        "{what}: candidate_hash"
    );
    assert!(record.chosen == expected.get("chosen").usize_(), "{what}: chosen");
    assert_pairs(&record.features, expected.get("features"), &format!("{what}: features"));
    assert_bits(record.propensity, expected.get("propensity"), &format!("{what}: propensity"));
    assert!(
        record.model_version == expected.get("model_version").u64_(),
        "{what}: model_version"
    );
    assert!(record.salt == expected.get("salt").str_(), "{what}: salt");
}

fn assert_resolutions(actual: &[Resolution], expected: &Json) {
    let expected = expected.arr();
    assert!(actual.len() == expected.len(), "resolution count mismatch");
    for (i, (r, e)) in actual.iter().zip(expected).enumerate() {
        assert!(r.decision_id == e.get("decision_id").str_(), "resolution[{i}]: id");
        assert!(r.kind.as_str() == e.get("kind").str_(), "resolution[{i}]: kind");
        match (&r.reward, e.get("reward")) {
            (None, Json::Null) => {}
            (Some(v), num) => assert_bits(*v, num, &format!("resolution[{i}]: reward")),
            _ => panic!("resolution[{i}]: reward mismatch"),
        }
    }
}

/// Replay the reference's end-to-end episode generator — same schedule,
/// same reward rule, same out-of-order resolutions, same rejected no-op
/// attempts — asserting every record and resolution against the corpus
/// along the way. Returns (mid state, final state): the mid state is the
/// snapshot just before the censor (two open ledger records), matching the
/// reference's serialization snapshot.
fn replay_episode(s: &Json) -> (BanditState, BanditState) {
    let bits: u32 = s.get("bits").u64_() as u32;
    let forgetting = s.get("forgetfulness").f64_();
    let horizon = s.get("horizon").f64_();
    let default_reward = s.get("default_reward").f64_();
    let t0 = 1_752_000_000.0;
    let arms = ["x", "y"];
    let events = s.get("events").arr();
    let mut resolutions: Vec<Resolution> = Vec::new();

    let mut state =
        new_bandit(1 << bits, horizon, default_reward, DEFAULT_EPSILON, forgetting).unwrap();

    // One decision on the real path; checks the event's candidate_hash and
    // full record, and returns the record plus the rule's reward.
    let play = |state: &mut _, t: f64, seg: &str, event: &Json| -> (DecisionRecord, f64) {
        let context = vec![("seg".to_string(), Value::Str(seg.to_string()))];
        let candidates: Vec<Features> = arms
            .iter()
            .map(|arm| encode(&context, arm, &[], bits).unwrap())
            .collect();
        assert!(
            candidate_set_hash(&candidates, 1 << bits) == event.get("candidate_hash").u64_(),
            "event candidate_hash at t={t}"
        );
        let record = decide(state, &candidates, t, "fleet-a").unwrap();
        assert_record(&record, event.get("record"), &record.decision_id.clone());
        let reward = if (arms[record.chosen] == "x") == (seg == "a") { 1.0 } else { 0.0 };
        (record, reward)
    };

    let mut deferred: Vec<(String, f64)> = Vec::new();
    for i in 0..10 {
        let seg = if i % 2 == 0 { "a" } else { "b" };
        let (record, reward) = play(&mut state, t0 + i as f64 * 600.0, seg, &events[i]);
        if i == 7 {
            // Never resolved: must expire.
        } else if i % 2 == 1 {
            deferred.push((record.decision_id, reward)); // resolved late, in reverse
        } else {
            resolutions.push(learn(&mut state, &record.decision_id, reward).unwrap());
        }
    }
    for (decision_id, reward) in deferred.into_iter().rev() {
        resolutions.push(learn(&mut state, &decision_id, reward).unwrap());
    }
    let (record, _) = play(&mut state, t0 + 6000.0, "a", &events[10]);
    let mid_state = state.clone(); // decisions 7 and 10 open
    resolutions.push(censor(&mut state, &record.decision_id).unwrap());
    let sweep_t = t0 + 7.0 * 600.0 + horizon; // decision 7 is exactly due
    assert_bits(sweep_t, s.get("expire_sweep_at"), "expire_sweep_at");
    resolutions.extend(expire(&mut state, sweep_t));
    assert_resolutions(&resolutions, s.get("resolutions"));

    // The corpus's rejected attempts: every one must be a structural no-op,
    // enforced by the final-state comparisons coming *after* them.
    for attempt in s.get("rejected").arr() {
        let id = attempt.get("decision_id").str_();
        let resolved = match attempt.get("action").str_() {
            "learn" => learn(&mut state, id, attempt.get("reward").f64_()).is_some(),
            "censor" => censor(&mut state, id).is_some(),
            other => panic!("unknown rejected action {other:?}"),
        };
        assert!(!resolved, "rejected attempt on {id} unexpectedly resolved");
    }
    (mid_state, state)
}

/// The end-to-end episode: replay it and require the final model bits and
/// predictions to match the corpus.
#[test]
fn episode_section() {
    let s = section("episode");
    let bits: u32 = s.get("bits").u64_() as u32;
    let arms = ["x", "y"];
    let (_, state) = replay_episode(&s);

    let fin = s.get("final");
    assert!(state.model_version == fin.get("model_version").u64_(), "final model_version");
    assert!(state.next_seq == fin.get("next_seq").u64_(), "final next_seq");
    assert!(
        state.ledger.len() == fin.get("open_ids").arr().len(),
        "final open decisions"
    );
    let m = fin.get("model");
    assert_bits(state.model.scale, m.get("scale"), "final scale");
    for (j, e) in m.get("xx").arr().iter().enumerate() {
        assert_bits(state.model.xx[j], e, &format!("final xx[{j}]"));
    }
    for (j, e) in m.get("xy").arr().iter().enumerate() {
        assert_bits(state.model.xy[j], e, &format!("final xy[{j}]"));
    }

    let mut expected = s.get("predictions").arr().iter();
    for seg in ["a", "b"] {
        for arm in arms {
            let e = expected.next().unwrap();
            assert!(e.get("seg").str_() == seg && e.get("arm").str_() == arm);
            let context = vec![("seg".to_string(), Value::Str(seg.to_string()))];
            let (estimate, uncertainty) =
                predict(&state.model, &encode(&context, arm, &[], bits).unwrap());
            assert_bits(estimate, e.get("estimate"), &format!("prediction {seg}/{arm} estimate"));
            assert_bits(
                uncertainty,
                e.get("uncertainty"),
                &format!("prediction {seg}/{arm} uncertainty"),
            );
        }
    }
}

/// The canonical state serialization: the corpus's byte strings must be
/// produced exactly, accepted back, and round-trip to equal states.
#[test]
fn serialization_section() {
    let s = section("serialization");
    assert!(s.get("format_version").u64_() == FORMAT_VERSION);

    let fresh_case = s.get("fresh");
    let fresh = new_bandit(
        fresh_case.get("dim").usize_(),
        fresh_case.get("horizon").f64_(),
        0.0,
        DEFAULT_EPSILON,
        DEFAULT_FORGETTING,
    )
    .unwrap();
    let (mid, fin) = replay_episode(&section("episode"));
    let open_ids: Vec<&str> = s
        .get("episode_mid")
        .get("open_ids")
        .arr()
        .iter()
        .map(Json::str_)
        .collect();
    assert!(
        mid.ledger.iter().map(|r| r.decision_id.as_str()).collect::<Vec<_>>() == open_ids,
        "mid snapshot open ids"
    );

    for (state, case, what) in [
        (&fresh, fresh_case, "fresh"),
        (&mid, s.get("episode_mid"), "episode_mid"),
        (&fin, s.get("episode_final"), "episode_final"),
    ] {
        let expected = unhex(case.get("bytes_hex").str_());
        let produced = serialize(state);
        assert!(produced == expected, "{what}: serialized bytes differ from corpus");
        let loaded = deserialize(&expected).unwrap();
        assert!(&loaded == state, "{what}: deserialized state differs");
        assert!(serialize(&loaded) == expected, "{what}: round-trip bytes differ");
    }
}

/// The public dict-shaped API: replay the corpus's api scenario calling
/// only the facade (`gittins_core::api`) — exactly what a binding's
/// acceptance test will do through its own public surface.
#[test]
fn api_section() {
    let s = section("api");
    let bits = s.get("bits").u64_() as u32;
    let horizon = s.get("horizon").f64_();
    let forgetting = s.get("forgetfulness").f64_();
    let salt = s.get("salt").str_();
    let catalog: Vec<(String, Vec<(String, Value)>)> = s
        .get("catalog")
        .arr()
        .iter()
        .map(|pair| (pair.arr()[0].str_().to_string(), values(&pair.arr()[1])))
        .collect();

    let mut state = api::create(bits, horizon, 0.0, DEFAULT_EPSILON, forgetting).unwrap();
    let mut records = Vec::new();
    for event in s.get("events").arr() {
        let context = values(event.get("context"));
        let record = api::decide(
            &mut state,
            &context,
            &catalog,
            event.get("t").f64_(),
            salt,
        )
        .unwrap();
        assert_record(&record, event.get("record"), &record.decision_id.clone());
        records.push(record);
    }

    let mut resolutions = Vec::new();
    resolutions.push(api::learn(&mut state, &records[1].decision_id, 1.0).unwrap());
    resolutions.push(api::learn(&mut state, &records[0].decision_id, 0.0).unwrap());
    resolutions.push(api::censor(&mut state, &records[2].decision_id).unwrap());
    resolutions.extend(api::expire(&mut state, s.get("expire_sweep_at").f64_()));
    assert_resolutions(&resolutions, s.get("resolutions"));

    let fin = s.get("final");
    assert!(state.model_version == fin.get("model_version").u64_(), "final model_version");
    assert!(state.next_seq == fin.get("next_seq").u64_(), "final next_seq");
    let expected = fin.get("state_hex").str_();
    assert!(
        api::serialize(&state) == expected,
        "final state hex differs from corpus"
    );
    assert!(api::deserialize(expected).unwrap() == state);
}

fn unhex(text: &str) -> Vec<u8> {
    assert!(text.len() % 2 == 0, "odd hex length");
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("bad hex"))
        .collect()
}
