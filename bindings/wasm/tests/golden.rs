//! The binding's acceptance gate, per spec/api.md: replay the golden `api`
//! section through the wasm module's public surface alone, and pin the
//! `serialization` section's bytes. Runs under Node via
//! `wasm-pack test --node`. WASM mandates IEEE-754 semantics and the
//! engine uses no libm transcendentals, so this is also the project's
//! first cross-platform bit-identity check: every float here must match
//! the corpus generated on the development machine, bit for bit.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::wasm_bindgen_test;

#[path = "../../../core/tests/golden/json.rs"]
mod json;
use json::Json;

use js_sys::{Array, BigInt, Object, Reflect};
use wasm_bindgen::prelude::*;

use gittins_wasm::{create, decide, deserialize, expire, learn, censor, model_bits, serialize};

fn section(name: &str) -> Json {
    Json::parse(include_str!("../../../spec/golden.json"))
        .get("sections")
        .get(name)
        .clone()
}

/// A corpus feature dict as a real JS object — the input shape the binding
/// is specified over.
fn js_features(values: &Json) -> JsValue {
    let obj = Object::new();
    for (name, v) in values.entries() {
        let value = match v {
            Json::Null => JsValue::NULL,
            Json::Bool(b) => JsValue::from_bool(*b),
            Json::Num(_) => JsValue::from_f64(v.f64_()),
            Json::Str(s) => JsValue::from_str(s),
            _ => panic!("unsupported feature value"),
        };
        Reflect::set(&obj, &JsValue::from_str(name), &value).unwrap();
    }
    obj.into()
}

fn js_catalog(catalog: &Json) -> JsValue {
    let array = Array::new();
    for pair in catalog.arr() {
        array.push(&Array::of2(
            &JsValue::from_str(pair.arr()[0].str_()),
            &js_features(&pair.arr()[1]),
        ));
    }
    array.into()
}

fn get(obj: &JsValue, key: &str) -> JsValue {
    Reflect::get(obj, &JsValue::from_str(key)).unwrap()
}

fn assert_bits(actual: f64, expected: &Json, what: &str) {
    let expected = expected.f64_();
    assert!(
        actual.to_bits() == expected.to_bits(),
        "{what}: got {actual:?}, corpus has {expected:?}"
    );
}

fn assert_record(record: &JsValue, expected: &Json) {
    let id = get(record, "decision_id").as_string().unwrap();
    assert!(id == expected.get("decision_id").str_(), "decision_id: {id}");
    assert_bits(get(record, "t").as_f64().unwrap(), expected.get("t"), &format!("{id}: t"));
    let hash: BigInt = get(record, "candidate_hash").unchecked_into();
    assert!(
        String::from(hash.to_string(10).unwrap()) == expected.get("candidate_hash").u64_().to_string(),
        "{id}: candidate_hash"
    );
    assert!(
        get(record, "chosen").as_f64().unwrap() as usize == expected.get("chosen").usize_(),
        "{id}: chosen"
    );
    let features: Array = get(record, "features").unchecked_into();
    let expected_features = expected.get("features").arr();
    assert!(features.length() as usize == expected_features.len(), "{id}: feature count");
    for (i, pair) in expected_features.iter().enumerate() {
        let actual: Array = features.get(i as u32).unchecked_into();
        assert!(
            actual.get(0).as_f64().unwrap() as usize == pair.arr()[0].usize_(),
            "{id}: feature index {i}"
        );
        assert_bits(actual.get(1).as_f64().unwrap(), &pair.arr()[1], &format!("{id}: feature value {i}"));
    }
    assert_bits(
        get(record, "propensity").as_f64().unwrap(),
        expected.get("propensity"),
        &format!("{id}: propensity"),
    );
    assert!(
        get(record, "model_version").as_f64().unwrap() as u64 == expected.get("model_version").u64_(),
        "{id}: model_version"
    );
    assert!(get(record, "salt").as_string().unwrap() == expected.get("salt").str_(), "{id}: salt");
}

fn assert_resolution(resolution: &JsValue, expected: &Json) {
    assert!(
        get(resolution, "decision_id").as_string().unwrap() == expected.get("decision_id").str_()
    );
    assert!(get(resolution, "kind").as_string().unwrap() == expected.get("kind").str_());
    match expected.get("reward") {
        Json::Null => assert!(get(resolution, "reward").is_null()),
        num => assert_bits(get(resolution, "reward").as_f64().unwrap(), num, "resolution reward"),
    }
}

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("bad hex"))
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[wasm_bindgen_test]
fn api_section_through_the_binding() {
    let s = section("api");
    let catalog = js_catalog(s.get("catalog"));
    let salt = s.get("salt").str_();
    let mut state = create(
        s.get("bits").u64_() as u32,
        s.get("horizon").f64_(),
        None,
        None,
        Some(s.get("forgetting").f64_()),
    )
    .unwrap();
    assert!(model_bits(&state).unwrap() as u64 == s.get("bits").u64_());

    let mut ids = Vec::new();
    for event in s.get("events").arr() {
        let record = decide(
            &mut state,
            &js_features(event.get("context")),
            &catalog,
            event.get("t").f64_(),
            salt,
        )
        .unwrap();
        assert_record(&record, event.get("record"));
        ids.push(get(&record, "decision_id").as_string().unwrap());
    }

    let mut resolutions = Vec::new();
    resolutions.push(learn(&mut state, &ids[1], 1.0));
    resolutions.push(learn(&mut state, &ids[0], 0.0));
    resolutions.push(censor(&mut state, &ids[2]));
    for r in expire(&mut state, s.get("expire_sweep_at").f64_()).iter() {
        resolutions.push(r);
    }
    let expected = s.get("resolutions").arr();
    assert!(resolutions.len() == expected.len(), "resolution count");
    for (r, e) in resolutions.iter().zip(expected) {
        assert_resolution(r, e);
    }

    // Second resolutions are no-ops, returning null.
    assert!(learn(&mut state, &ids[0], 1.0).is_null());
    assert!(censor(&mut state, &ids[2]).is_null());

    let fin = s.get("final");
    let bytes = serialize(&state).to_vec();
    assert!(hex(&bytes) == fin.get("state_hex").str_(), "final state bytes differ from corpus");
    let restored = deserialize(&bytes).unwrap();
    assert!(serialize(&restored).to_vec() == bytes, "round-trip bytes differ");
}

#[wasm_bindgen_test]
fn serialization_section_bytes() {
    let s = section("serialization");
    // The fresh state is reproducible through the public surface alone.
    let fresh = s.get("fresh");
    let state = create(fresh.get("dim").u64_().trailing_zeros(), fresh.get("horizon").f64_(), None, None, None).unwrap();
    assert!(hex(&serialize(&state).to_vec()) == fresh.get("bytes_hex").str_(), "fresh bytes");
    // Every pinned byte string is accepted and re-emitted exactly.
    for case in ["fresh", "episode_mid", "episode_final"] {
        let expected = unhex(s.get(case).get("bytes_hex").str_());
        let state = deserialize(&expected).unwrap();
        assert!(serialize(&state).to_vec() == expected, "{case}: bytes differ after round-trip");
    }
}

#[wasm_bindgen_test]
fn rejections_throw() {
    assert!(create(0, 10.0, None, None, None).is_err());
    let mut state = create(4, 10.0, None, None, None).unwrap();
    let context = Object::new();
    let bad = Object::new();
    Reflect::set(&bad, &JsValue::from_str("nested"), &Object::new().into()).unwrap();
    let candidates = Array::of1(&Array::of2(&JsValue::from_str("a"), &bad.into()));
    assert!(decide(&mut state, &context.into(), &candidates.into(), 0.0, "s").is_err());
    assert!(deserialize(&[0, 1, 2]).is_err());
}
