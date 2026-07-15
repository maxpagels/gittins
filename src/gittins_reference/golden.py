"""Golden test vector generation (design doc section 8).

This module generates `spec/golden.json`: a corpus of (input -> expected
output) vectors produced by this reference implementation, covering every
layer plus one full end-to-end episode. The corpus is the contract that
every other implementation — the Rust core, each language binding — must
match **bit for bit** in CI. It is also Phase 0's exit artifact: an
independent implementation of decide/learn is done when it reproduces the
episode section exactly.

Regenerate with:

    uv run python -m gittins_reference.golden

`tests/test_golden.py` asserts the checked-in file matches regeneration
exactly, so any semantic drift in the reference shows up as a failing test
and a reviewable diff of the vectors.

Serialization notes for independent implementations:
- Floats are JSON numbers written with Python's shortest round-trip repr;
  parse to IEEE-754 doubles and compare bitwise.
- Hashes and RNG outputs are unsigned 64-bit integers written as JSON
  numbers (JSON has no integer width; parse as u64).
- The corpus contains no NaN or infinity; scenarios needing
  half_life = inf live in the pytest property suites instead.
"""

import json
import sys

from gittins_reference.decay import (
    RENORM_LIMIT,
    add,
    merge,
    new_accumulator,
    new_vector,
    read,
    vector_add,
    vector_merge,
    vector_read,
)
from gittins_reference.decide import candidate_set_hash, decide, new_bandit
from gittins_reference.detmath import exp2
from gittins_reference.encoding import encode, feature_tokens, pair_hash
from gittins_reference.exploration import apply_floor, inverse_gap_probabilities, sample_index
from gittins_reference.ledger import censor, expire, learn
from gittins_reference.merge import merge_states
from gittins_reference.model import new_model, predict, update
from gittins_reference.rng import derive_key, random_u64, random_unit

HOUR = 3600.0
DAY = 24 * HOUR
T0 = 1_752_000_000.0


def record_json(record):
    return {
        "decision_id": record.decision_id,
        "t": record.t,
        "candidate_hash": record.candidate_hash,
        "chosen": record.chosen,
        "features": list(record.features),
        "propensity": record.propensity,
        "model_version": record.model_version,
        "salt": record.salt,
    }


def resolution_json(resolution):
    return {"decision_id": resolution.decision_id, "kind": resolution.kind, "reward": resolution.reward}


def model_json(model, t):
    return {
        "xx": {"origin": model.xx.origin, "values": list(model.xx.values)},
        "xy": {"origin": model.xy.origin, "values": list(model.xy.values)},
        "read_at": t,
        "xx_read": list(vector_read(model.xx, t)),
        "xy_read": list(vector_read(model.xy, t)),
    }


def rng_vectors():
    key_cases = [("", ""), ("d", ""), ("", "d"), ("ab", "c"), ("a", "bc"),
                 ("decision-0001", "pepper"), ("δεδομένα", "sälté")]
    key = derive_key("golden", "rng")
    return {
        "derive_key": [
            {"decision_id": d, "salt": s, "key": derive_key(d, s)} for d, s in key_cases
        ],
        "stream": {
            "decision_id": "golden",
            "salt": "rng",
            "key": key,
            "u64": [random_u64(key, c) for c in range(8)],
            "unit": [random_unit(key, c) for c in range(8)],
            "u64_at_2_32": random_u64(key, 2**32),
        },
    }


def exp2_vectors():
    args = [0.0, 1.0, -1.0, 0.5, -0.5, 2.75, -2.75, 10.125, -10.125,
            53.0, -53.0, 100.0, -100.0, 127.9375, -127.9375, 1e-9, -1e-9]
    return [{"arg": a, "value": exp2(a)} for a in args]


def decay_vectors():
    acc = new_accumulator(HOUR, T0)
    steps = []
    for value, t in [(1.0, T0), (2.0, T0 + 1800.0), (-0.5, T0 + 7200.0),
                     (0.25, T0 + (RENORM_LIMIT + 1.0) * HOUR)]:  # crosses renormalization
        acc = add(acc, value, t)
        steps.append({"add": [value, t], "origin": acc.origin, "total": acc.total})
    other = add(add(new_accumulator(HOUR, T0), 0.5, T0 + 900.0), -1.0, T0 + 3600.0)
    merged = merge(acc, other)
    vec = new_vector(HOUR, T0, 3)
    vec = vector_add(vec, [1.0, 0.5, -0.25], T0 + 600.0)
    vec = vector_add(vec, [0.0, 2.0, 1.0], T0 + 4200.0)
    vec2 = vector_add(new_vector(HOUR, T0, 3), [1.0, 1.0, 1.0], T0 + 2400.0)
    vmerged = vector_merge(vec, vec2)
    t_read = T0 + (RENORM_LIMIT + 2.0) * HOUR
    return {
        "accumulator": {
            "half_life": HOUR, "created_at": T0, "steps": steps,
            "read_at": t_read, "read": read(acc, t_read),
        },
        "merge": {"origin": merged.origin, "total": merged.total,
                  "read_at": T0 + 7200.0, "read": read(merge(other, acc), T0 + 7200.0)},
        "vector": {"origin": vmerged.origin, "values": list(vmerged.values),
                   "read_at": T0 + 4800.0, "read": list(vector_read(vmerged, T0 + 4800.0))},
    }


def model_vectors():
    m = new_model(2, HOUR, T0)
    history = [([1.0, 0.0], 1.0, T0), ([0.0, 1.0], -0.5, T0 + 1800.0), ([1.0, 1.0], 0.25, T0 + 3600.0)]
    for x, r, t in history:
        m = update(m, x, r, t)
    est, unc = predict(m, [1.0, -1.0], T0 + 5400.0)
    est_gap, unc_gap = predict(m, [1.0, -1.0], T0 + 5400.0 + 20 * HOUR)
    return {
        "dim": 2, "half_life": HOUR, "ridge": 1.0, "created_at": T0,
        "updates": [{"x": x, "reward": r, "t": t} for x, r, t in history],
        "state": model_json(m, T0 + 5400.0),
        "predict": {"x": [1.0, -1.0], "t": T0 + 5400.0, "estimate": est, "uncertainty": unc},
        "predict_after_gap": {"x": [1.0, -1.0], "t": T0 + 5400.0 + 20 * HOUR,
                              "estimate": est_gap, "uncertainty": unc_gap},
    }


def exploration_vectors():
    estimates = [0.5, -0.25, 0.0, 0.5]
    gamma = 10.0
    raw = inverse_gap_probabilities(estimates, gamma)
    floored = apply_floor(raw)
    key = derive_key("decision-0001", "pepper")
    return {
        "estimates": estimates, "gamma": gamma,
        "inverse_gap": raw, "floored": floored,
        "key": key, "samples": [sample_index(floored, key, c) for c in range(6)],
    }


def encoding_vectors():
    token_cases = [
        {"namespace": "c", "values": {"device": "mobile", "hour": 14}},
        {"namespace": "a", "values": {"price": 0.5, "on_sale": True, "skipped": None}},
    ]
    encode_cases = [
        {"context": {"hour": 0.5, "device": "mobile"}, "arm_id": "banner-sale",
         "action": {"color": "red", "price": 1}, "bits": 4},
        {"context": {"seg": "a"}, "arm_id": "x", "action": {}, "bits": 6},
        {"context": {}, "arm_id": "y", "action": {}, "bits": 6},
    ]
    return {
        "tokens": [
            {**case, "values": {k: v for k, v in case["values"].items()},
             "tokens": [[t, v] for t, v in feature_tokens(case["namespace"], case["values"])]}
            for case in token_cases
        ],
        "pair_hashes": [
            {"left": l, "right": r, "hash": pair_hash(l, r)}
            for l, r in [("", ""), ("", "i|x"), ("c|seg=a", "i|x"), ("c|hour", "a|price")]
        ],
        "encode": [
            {**case, "vector": encode(case["context"], case["arm_id"], case["action"], case["bits"])}
            for case in encode_cases
        ],
    }


def episode_vector():
    """The end-to-end acceptance scenario: two agents decide and learn over
    hash-encoded candidates with out-of-order rewards, a censor, and an
    expiry sweep, then merge. Matching this section exactly means an
    independent decide/learn implementation is done (Phase 0 exit)."""
    bits = 4
    half_life = DAY
    horizon = 2 * HOUR
    arms = ["x", "y"]
    events = []
    resolutions = []

    def play(state, salt, t, seg):
        context = {"seg": seg}
        cands = [encode(context, arm, {}, bits) for arm in arms]
        record, state = decide(state, cands, t, salt)
        events.append({"agent": salt, "t": t, "context": context, "arms": arms,
                       "candidate_hash": candidate_set_hash(cands),
                       "record": record_json(record)})
        reward = 1.0 if (arms[record.chosen] == "x") == (seg == "a") else 0.0
        return state, record, reward

    def resolve(action, state, *args):
        resolution, state = action(state, *args)
        resolutions.append(resolution_json(resolution))
        return state

    a = new_bandit(1 << bits, half_life, T0, horizon=horizon)
    b = new_bandit(1 << bits, half_life, T0, horizon=horizon)
    deferred = []
    for i in range(10):
        a, record, reward = play(a, "fleet-a", T0 + i * 600.0, "a" if i % 2 == 0 else "b")
        if i == 7:
            pass  # never resolved: must expire
        elif i % 2 == 1:
            deferred.append((record.decision_id, reward))  # resolved late, in reverse
        else:
            a = resolve(learn, a, record.decision_id, reward)
    for decision_id, reward in reversed(deferred):
        a = resolve(learn, a, decision_id, reward)
    a, record, _ = play(a, "fleet-a", T0 + 6000.0, "a")
    a = resolve(censor, a, record.decision_id)
    sweep_t = T0 + 7 * 600.0 + horizon  # decision 7 is exactly due
    expired, a = expire(a, sweep_t)
    resolutions.extend(resolution_json(r) for r in expired)
    for i in range(5):
        b, record, reward = play(b, "fleet-b", T0 + 300.0 + i * 600.0, "a" if i % 2 == 0 else "b")
        b = resolve(learn, b, record.decision_id, reward)
    merged = merge_states(a, b)
    t_query = T0 + 9000.0
    predictions = []
    for seg in ["a", "b"]:
        for arm in arms:
            est, unc = predict(merged.model, encode({"seg": seg}, arm, {}, bits), t_query)
            predictions.append({"seg": seg, "arm": arm, "t": t_query,
                                "estimate": est, "uncertainty": unc})
    return {
        "bits": bits, "half_life": half_life, "horizon": horizon,
        "default_reward": 0.0, "created_at": T0,
        "reward_rule": "1.0 if (arm == 'x') == (seg == 'a') else 0.0",
        "expire_sweep_at": sweep_t,
        "events": events,
        "resolutions": resolutions,
        "final": {
            "a": {"model_version": a.model_version, "next_seq": a.next_seq,
                  "open_ids": [r.decision_id for r in a.ledger]},
            "b": {"model_version": b.model_version, "next_seq": b.next_seq,
                  "open_ids": [r.decision_id for r in b.ledger]},
            "merged": {"model_version": merged.model_version,
                       "model": model_json(merged.model, t_query)},
        },
        "predictions": predictions,
    }


def generate() -> dict:
    return {
        "format_version": 1,
        "sections": {
            "rng": rng_vectors(),
            "exp2": exp2_vectors(),
            "decay": decay_vectors(),
            "model": model_vectors(),
            "exploration": exploration_vectors(),
            "encoding": encoding_vectors(),
            "episode": episode_vector(),
        },
    }


def render() -> str:
    return json.dumps(generate(), sort_keys=True, indent=1) + "\n"


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "spec/golden.json"
    # newline="\n": the corpus must be byte-identical on every platform.
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(render())
    print(f"wrote {path}")
