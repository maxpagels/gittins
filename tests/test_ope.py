import gzip
import json

import pytest

from gittins_reference import api, ope
from gittins_reference import state as state_module

T0 = 1_752_000_000.0
HORIZON = 3600.0

CATALOG_A = [("basic", {"price": 3.0}), ("plus", {"price": 9.0, "trial": True}), ("free", {})]
CATALOG_B = [("basic", {"price": 3.0}), ("mega", {"price": 15.0})]


def run_agent(bits=4, epsilon=0.1, forgetfulness=0.9, salt="agent"):
    """Drive the public api exactly as an app would, logging by appending
    what each call returns — `api.log_line`, verbatim — in arrival order.
    Decisions interleave with resolutions so the model state at decision
    time actually evolves — the self-evaluation identity test below is
    vacuous otherwise. Returns (log dicts, state handle)."""
    state = api.create(bits, horizon=HORIZON, epsilon=epsilon, forgetfulness=forgetfulness)
    log = []
    records = []

    def decide(i):
        catalog = CATALOG_B if i % 3 == 2 else CATALOG_A
        context = {"seg": "a" if i % 2 == 0 else "b", "hour": i}
        record = api.decide(state, context, catalog, T0 + i * 60.0, salt)
        log.append(json.loads(api.log_line(record)))
        records.append(record)

    def resolve(resolution):
        log.append(json.loads(api.log_line(resolution)))

    for i in range(3):
        decide(i)
    resolve(api.learn(state, records[1].decision_id, 1.0))
    resolve(api.learn(state, records[0].decision_id, 0.0))
    decide(3)
    decide(4)
    resolve(api.censor(state, records[2].decision_id))
    resolve(api.learn(state, records[4].decision_id, 0.5))
    for i in (5, 6, 7):
        decide(i)
    for r in api.expire(state, T0 + 3 * 60.0 + HORIZON):  # decision 3 exactly due
        resolve(r)
    return log, state


def parse(log):
    return tuple(ope.parse_log(json.dumps(e) for e in log))


class TestLog:
    def test_a_real_run_verifies_clean(self):
        log, _ = run_agent()
        events = parse(log)
        assert ope.verify(events) == ()
        # 8 decisions; 3 rewarded + 1 expired + 1 censored; 3 left open.
        assert sum(isinstance(e, ope.DecisionEvent) for e in events) == 8
        assert sum(isinstance(e, ope.ResolutionEvent) for e in events) == 5

    def test_gzip_and_plain_files_parse_identically(self, tmp_path):
        log, _ = run_agent()
        text = "".join(json.dumps(e) + "\n" for e in log)
        plain = tmp_path / "log.jsonl"
        plain.write_text(text, encoding="utf-8")
        gz = tmp_path / "log.jsonl.gz"  # detection is by magic bytes, not name
        gz.write_bytes(gzip.compress(text.encode("utf-8")))
        assert tuple(ope.read_log(plain)) == tuple(ope.read_log(gz)) == parse(log)

    def test_blank_lines_and_unknown_fields_are_ignored(self):
        log, _ = run_agent()
        lines = ["", json.dumps({**log[0], "extra": "ignored"}), "  "]
        lines += [json.dumps(e) for e in log[1:]]
        assert ope.verify(ope.parse_log(lines)) == ()

    def test_parse_rejects_malformed_lines(self):
        with pytest.raises(ValueError, match="line 1: not valid JSON"):
            tuple(ope.parse_log(["{nope"]))
        with pytest.raises(ValueError, match="line 1: each event must be a JSON object"):
            tuple(ope.parse_log(['["not", "an", "object"]']))
        with pytest.raises(ValueError, match="line 1: unknown event kind"):
            tuple(ope.parse_log(['{"event": "telemetry"}']))
        with pytest.raises(ValueError, match="line 2: malformed decision event"):
            tuple(ope.parse_log(['{"event": "resolution", "decision_id": "a:0", "kind": "rewarded", "reward": 1.0}',
                                 '{"event": "decision", "bits": 4}']))
        with pytest.raises(ValueError, match="line 1: malformed resolution event"):
            tuple(ope.parse_log(['{"event": "resolution", "decision_id": "a:0", "kind": "rewarded", "reward": "big"}']))

    def test_parsing_is_incremental(self):
        # A generator end to end: events arrive as their lines are read,
        # and a later malformed line only surfaces when reached.
        log, _ = run_agent()
        lines = [json.dumps(log[0]), "{broken"]
        stream = ope.parse_log(iter(lines))
        first = next(stream)
        assert isinstance(first, ope.DecisionEvent) and first.line == 1
        with pytest.raises(ValueError, match="line 2: not valid JSON"):
            next(stream)


class TestVerify:
    def test_catches_tampered_candidates(self):
        log, _ = run_agent()
        tampered = json.loads(json.dumps(log))
        event = tampered[0]
        # Tamper a candidate that was NOT chosen: the hash breaks, the
        # chosen features stay consistent — one precise finding.
        not_chosen = (event["record"]["chosen"] + 1) % len(event["candidates"])
        event["candidates"][not_chosen][1]["price"] = 4.0
        problems = ope.verify(parse(tampered))
        assert problems == (
            f"line 1: decision {event['record']['decision_id']}: "
            "candidates do not match the logged candidate_hash",
        )

    def test_catches_tampered_chosen_candidate(self):
        log, _ = run_agent()
        tampered = json.loads(json.dumps(log))
        event = tampered[0]
        chosen = event["record"]["chosen"]
        event["candidates"][chosen][1]["surprise"] = 1.0
        problems = ope.verify(parse(tampered))
        assert len(problems) == 2  # hash AND chosen features
        assert "candidate_hash" in problems[0]
        assert "chosen features" in problems[1]

    def test_catches_resolution_misuse(self):
        log, _ = run_agent()
        first_resolution = next(e for e in log if e["event"] == "resolution")
        log = log + [first_resolution]  # duplicate
        log = log + [{"event": "resolution", "decision_id": "ghost:9", "kind": "rewarded", "reward": 1.0}]
        problems = ope.verify(parse(log))
        assert any("second resolution for decision" in p for p in problems)
        assert any("unknown decision 'ghost:9'" in p for p in problems)

    def test_catches_bad_records(self):
        log, _ = run_agent()
        decisions = [i for i, e in enumerate(log) if e["event"] == "decision"]
        bad = json.loads(json.dumps(log))
        bad[decisions[0]]["record"]["propensity"] = 0.0
        bad[decisions[1]]["bits"] = 0
        bad[decisions[-1]]["record"]["t"] = T0 - 1.0  # t is not hashed
        problems = ope.verify(parse(bad))
        assert any("propensity must be in (0, 1]" in p for p in problems)
        assert any("bits must be between 1 and 24" in p for p in problems)
        assert any("t decreases within salt 'agent'" in p for p in problems)


class TestEvaluate:
    def test_self_evaluation_is_the_identity(self):
        # Evaluating the logging configuration replays the exact model
        # states the logger had, so every importance weight is exactly
        # 1.0: ips == snips == logged_mean and ess == n, bit for bit.
        log, _ = run_agent(bits=4, epsilon=0.1, forgetfulness=0.9)
        report = ope.evaluate(parse(log), bits=4, epsilon=0.1, forgetfulness=0.9)
        assert report.decisions == 8
        assert report.resolved == 4  # 3 rewarded + 1 expired
        assert report.censored == 1
        assert report.unresolved == 3
        assert report.max_weight == 1.0
        assert report.ips == report.snips == report.logged_mean
        assert report.ess == float(report.resolved)

    def test_other_configurations_reweight(self):
        log, _ = run_agent(bits=4, epsilon=0.1, forgetfulness=0.9)
        events = parse(log)
        report = ope.evaluate(events, bits=5, epsilon=0.05, forgetfulness=0.999)
        assert report.resolved == 4
        # Logged propensities are >= epsilon/k = 0.1/3, target q <= 1.
        assert 0.0 < report.max_weight <= 30.0
        assert report.ess is not None and 0.0 < report.ess <= report.resolved
        assert report.ips is not None and report.snips is not None
        # And a different config really is a different answer.
        same = ope.evaluate(events, bits=4, epsilon=0.1, forgetfulness=0.9)
        assert report != same

    def test_nothing_resolved_means_no_estimates(self):
        log, _ = run_agent()
        decisions_only = [e for e in log if e["event"] == "decision"]
        report = ope.evaluate(parse(decisions_only), bits=4)
        assert report.decisions == 8 and report.unresolved == 8
        assert report.ips is None and report.snips is None and report.logged_mean is None

    def test_rejects_what_verify_would_catch(self):
        log, _ = run_agent()
        bad = json.loads(json.dumps(log))
        first = next(e for e in bad if e["event"] == "decision")
        first["record"]["propensity"] = 0.0
        with pytest.raises(ValueError, match="propensity must be in"):
            ope.evaluate(parse(bad), bits=4)


class TestReplay:
    def test_reproduces_the_logging_agents_model_bit_for_bit(self):
        log, state = run_agent(bits=4, epsilon=0.1, forgetfulness=0.9)
        rebuilt = ope.replay(parse(log), bits=4, horizon=HORIZON, epsilon=0.1, forgetfulness=0.9)
        assert rebuilt.model == state._state.model
        assert rebuilt.model_version == 4
        assert rebuilt.next_seq == 0 and rebuilt.ledger == ()
        # The rebuilt state is deployable: it serializes like any other.
        assert state_module.deserialize(state_module.serialize(rebuilt)) == rebuilt

    def test_censored_and_unresolved_never_train(self):
        log, _ = run_agent()
        rebuilt = ope.replay(parse(log), bits=4, horizon=HORIZON, epsilon=0.1, forgetfulness=0.9)
        assert rebuilt.model_version == 4  # of 8 decisions, only 4 resolved with rewards
