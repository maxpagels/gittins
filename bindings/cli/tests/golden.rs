//! The CLI binding's acceptance gate: the golden `ope` section
//! replayed through the library half — the log lines
//! re-serialized from the corpus and parsed as real JSONL, the clean
//! verify, every evaluation report bit for bit (the logging
//! configuration's identity included), the replay hex, and the tampered
//! log's exact findings. Plus the gzip path, which the corpus cannot
//! carry.

use serde_json::Value as Json;

use gittins_cli::ope::{evaluate, parse_log, read_log, replay, verify, Event};

fn section() -> Json {
    let corpus: Json = serde_json::from_str(include_str!("../../../spec/golden.json")).unwrap();
    corpus["sections"]["ope"].clone()
}

/// The corpus log as JSONL text. serde_json preserves the corpus's key
/// order, which the corpus guarantees is the order the reference
/// encoded with (golden.py's canonical-order note).
fn log_lines(log: &Json) -> String {
    log.as_array()
        .unwrap()
        .iter()
        .map(|e| serde_json::to_string(e).unwrap())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Collected events back as the stream shape verify/evaluate/replay
/// consume — the tests reuse one parsed log across configurations.
fn stream(events: &[Event]) -> impl Iterator<Item = Result<Event, String>> + '_ {
    events.iter().cloned().map(Ok)
}

fn assert_bits(actual: f64, expected: &Json, what: &str) {
    let expected = expected.as_f64().unwrap();
    assert!(
        actual.to_bits() == expected.to_bits(),
        "{what}: got {actual:?}, corpus has {expected:?}"
    );
}

fn assert_opt(actual: Option<f64>, expected: &Json, what: &str) {
    match actual {
        None => assert!(expected.is_null(), "{what}: got None, corpus has {expected:?}"),
        Some(v) => assert_bits(v, expected, what),
    }
}

#[test]
fn ope_section() {
    let s = section();
    let events = parse_log(&log_lines(&s["log"])).unwrap();
    assert!(verify(stream(&events)).unwrap().is_empty(), "the golden log must verify clean");

    for case in s["reports"].as_array().unwrap() {
        let bits = case["bits"].as_u64().unwrap() as u32;
        let epsilon = case["epsilon"].as_f64().unwrap();
        let forgetfulness = case["forgetfulness"].as_f64().unwrap();
        let what = format!("report bits={bits} epsilon={epsilon}");
        let report = evaluate(stream(&events), bits, epsilon, forgetfulness).unwrap();
        let expected = &case["report"];
        assert!(report.decisions == expected["decisions"].as_u64().unwrap(), "{what}: decisions");
        assert!(report.resolved == expected["resolved"].as_u64().unwrap(), "{what}: resolved");
        assert!(report.censored == expected["censored"].as_u64().unwrap(), "{what}: censored");
        assert!(report.unresolved == expected["unresolved"].as_u64().unwrap(), "{what}: unresolved");
        assert_opt(report.logged_mean, &expected["logged_mean"], &format!("{what}: logged_mean"));
        assert_opt(report.ips, &expected["ips"], &format!("{what}: ips"));
        assert_opt(report.snips, &expected["snips"], &format!("{what}: snips"));
        assert_opt(report.ess, &expected["ess"], &format!("{what}: ess"));
        assert_opt(report.max_weight, &expected["max_weight"], &format!("{what}: max_weight"));
    }

    // The logging configuration's report is the exact identity — the
    // sanity property every implementation must reproduce.
    let first = &s["reports"][0];
    let identity = evaluate(
        stream(&events),
        first["bits"].as_u64().unwrap() as u32,
        first["epsilon"].as_f64().unwrap(),
        first["forgetfulness"].as_f64().unwrap(),
    )
    .unwrap();
    assert!(identity.ips == identity.snips && identity.snips == identity.logged_mean);
    assert!(identity.max_weight == Some(1.0));
    assert!(identity.ess == Some(identity.resolved as f64));

    let r = &s["replay"];
    let state = replay(
        stream(&events),
        r["bits"].as_u64().unwrap() as u32,
        r["horizon"].as_f64().unwrap(),
        r["default_reward"].as_f64().unwrap(),
        r["epsilon"].as_f64().unwrap(),
        r["forgetfulness"].as_f64().unwrap(),
    )
    .unwrap();
    assert!(
        gittins_core::api::serialize(&state) == r["state_hex"].as_str().unwrap(),
        "replay state hex differs from corpus"
    );

    let tampered = parse_log(&log_lines(&s["tampered"]["log"])).unwrap();
    let problems = verify(stream(&tampered)).unwrap();
    let expected: Vec<&str> = s["tampered"]["problems"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_str().unwrap())
        .collect();
    assert!(problems == expected, "tampered findings differ: {problems:?}");
}

#[test]
fn gzip_and_plain_files_parse_identically() {
    use std::io::Write;

    let s = section();
    let text = log_lines(&s["log"]) + "\n";
    let dir = std::env::temp_dir();
    let plain = dir.join("gittins-cli-test-log.jsonl");
    std::fs::write(&plain, &text).unwrap();
    let mut encoder =
        flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(text.as_bytes()).unwrap();
    let gz = dir.join("gittins-cli-test-log.bin"); // detection is by magic, not name
    std::fs::write(&gz, encoder.finish().unwrap()).unwrap();

    let from_plain: Vec<Event> =
        read_log(plain.to_str().unwrap()).unwrap().collect::<Result<_, _>>().unwrap();
    let from_gz: Vec<Event> =
        read_log(gz.to_str().unwrap()).unwrap().collect::<Result<_, _>>().unwrap();
    assert!(from_plain == from_gz && !from_plain.is_empty());
    assert!(from_plain.iter().filter(|e| matches!(e, Event::Decision(_))).count() == 8);

    std::fs::remove_file(plain).ok();
    std::fs::remove_file(gz).ok();
}

/// The stream really is incremental: the first event arrives before a
/// later broken line's error is ever produced.
#[test]
fn reading_is_incremental() {
    let s = section();
    let first_line = serde_json::to_string(&s["log"][0]).unwrap();
    let path = std::env::temp_dir().join("gittins-cli-incremental.jsonl");
    std::fs::write(&path, format!("{first_line}\n{{broken\n")).unwrap();

    let mut events = read_log(path.to_str().unwrap()).unwrap();
    assert!(matches!(events.next(), Some(Ok(Event::Decision(_)))));
    assert!(events.next() == Some(Err("line 2: not valid JSON".to_string())));
    assert!(events.next().is_none());

    std::fs::remove_file(path).ok();
}

#[test]
fn parse_rejects_malformed_lines() {
    assert!(parse_log("{nope").unwrap_err() == "line 1: not valid JSON");
    assert!(
        parse_log("[1, 2]").unwrap_err()
            == "line 1: each event must be a JSON object with an 'event' kind"
    );
    assert!(parse_log("{\"event\": \"telemetry\"}").unwrap_err() == "line 1: unknown event kind");
    assert!(
        parse_log("{\"event\": \"decision\", \"bits\": 4}").unwrap_err()
            == "line 1: malformed decision event"
    );
    assert!(
        parse_log("{\"event\": \"resolution\", \"decision_id\": \"a:0\", \"kind\": \"rewarded\", \"reward\": \"big\"}")
            .unwrap_err()
            == "line 1: malformed resolution event"
    );
}
