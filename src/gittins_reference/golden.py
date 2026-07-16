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
  forgetting = 1.0 (never forget) live in the pytest property suites.
- Model training is order-dependent (per-update forgetting), so every
  vector specifies its exact update/resolution sequence.
"""

import json
import sys

from gittins_reference.decide import candidate_set_hash, decide, new_bandit
from gittins_reference.encoding import encode, feature_tokens, pair_hash
from gittins_reference.exploration import epsilon_greedy_probabilities, sample_index
from gittins_reference.ledger import censor, expire, learn
from gittins_reference.model import new_model, predict, update
from gittins_reference.rng import derive_key, random_u64, random_unit
from gittins_reference.state import FORMAT_VERSION, deserialize, serialize

HOUR = 3600.0
T0 = 1_752_000_000.0


def pairs_json(pairs):
    """Sparse features as [[index, value], ...] in index order."""
    return [[j, v] for j, v in pairs]


def record_json(record):
    return {
        "decision_id": record.decision_id,
        "t": record.t,
        "candidate_hash": record.candidate_hash,
        "chosen": record.chosen,
        "features": pairs_json(record.features),
        "propensity": record.propensity,
        "model_version": record.model_version,
        "salt": record.salt,
    }


def resolution_json(resolution):
    return {"decision_id": resolution.decision_id, "kind": resolution.kind, "reward": resolution.reward}


def model_json(model):
    # The stored state is pre-scaled: true sums are scale * xx, scale * xy.
    return {"scale": model.scale, "xx": list(model.xx), "xy": list(model.xy)}


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


def model_vectors():
    forgetting = 0.9
    m = new_model(2, forgetting)
    history = [
        (((0, 1.0),), 1.0),
        (((1, 1.0),), -0.5),
        (((0, 1.0), (1, 1.0)), 0.25),
    ]
    states = []
    for x, r in history:
        m = update(m, x, r)
        states.append(model_json(m))
    probe = ((0, 1.0), (1, -1.0))
    est, unc = predict(m, probe)
    return {
        "dim": 2, "forgetting": forgetting, "ridge": 1.0,
        "updates": [{"x": pairs_json(x), "reward": r} for x, r in history],
        "states": states,
        "predict": {"x": pairs_json(probe), "estimate": est, "uncertainty": unc},
    }


def exploration_vectors():
    estimates = [0.5, -0.25, 0.0, 0.5]  # a tie for the maximum, deliberately
    epsilon = 0.05
    p = epsilon_greedy_probabilities(estimates, epsilon)
    key = derive_key("decision-0001", "pepper")
    return {
        "estimates": estimates, "epsilon": epsilon,
        "probabilities": p,
        "key": key, "samples": [sample_index(p, key, c) for c in range(6)],
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
            {**case, "pairs": pairs_json(
                encode(case["context"], case["arm_id"], case["action"], case["bits"])
            )}
            for case in encode_cases
        ],
    }


def run_episode():
    """The end-to-end acceptance scenario: one agent decides and learns over
    hash-encoded candidates with out-of-order rewards, a censor, and an
    expiry sweep. Training is order-dependent, so the exact resolution
    sequence below *is* the contract. Matching this section exactly means
    an independent decide/learn implementation is done (Phase 0 exit).

    Returns (episode section, mid state, final state): the mid state is the
    snapshot just before the censor — the moment the ledger holds two open
    records — and both states feed the serialization section."""
    bits = 4
    forgetting = 0.9
    horizon = 2 * HOUR
    arms = ["x", "y"]
    events = []
    resolutions = []

    def play(state, t, seg):
        context = {"seg": seg}
        cands = [encode(context, arm, {}, bits) for arm in arms]
        record, state = decide(state, cands, t, "fleet-a")
        events.append({"t": t, "context": context, "arms": arms,
                       "candidate_hash": candidate_set_hash(cands, 1 << bits),
                       "record": record_json(record)})
        reward = 1.0 if (arms[record.chosen] == "x") == (seg == "a") else 0.0
        return state, record, reward

    def resolve(action, state, *args):
        resolution, state = action(state, *args)
        resolutions.append(resolution_json(resolution))
        return state

    a = new_bandit(1 << bits, horizon=horizon, forgetting=forgetting)
    deferred = []
    for i in range(10):
        a, record, reward = play(a, T0 + i * 600.0, "a" if i % 2 == 0 else "b")
        if i == 7:
            pass  # never resolved: must expire
        elif i % 2 == 1:
            deferred.append((record.decision_id, reward))  # resolved late, in reverse
        else:
            a = resolve(learn, a, record.decision_id, reward)
    for decision_id, reward in reversed(deferred):
        a = resolve(learn, a, decision_id, reward)
    a, record, _ = play(a, T0 + 6000.0, "a")
    mid_state = a  # decisions 7 and 10 open: the serialization mid snapshot
    a = resolve(censor, a, record.decision_id)
    sweep_t = T0 + 7 * 600.0 + horizon  # decision 7 is exactly due
    expired, a = expire(a, sweep_t)
    resolutions.extend(resolution_json(r) for r in expired)
    # Rejected resolutions: every way an attempt can find no open record —
    # conflicting duplicate reward, post-expiry reward, reward after censor,
    # censor after reward, unknown id. Each must be a structural no-op. The
    # final state below is captured *after* these attempts, so an
    # independent implementation must reject them all to match it.
    rejected = [
        {"action": "learn", "decision_id": "fleet-a:0", "reward": 0.0},
        {"action": "learn", "decision_id": "fleet-a:7", "reward": 1.0},
        {"action": "learn", "decision_id": "fleet-a:10", "reward": 1.0},
        {"action": "censor", "decision_id": "fleet-a:0"},
        {"action": "learn", "decision_id": "fleet-a:99", "reward": 1.0},
    ]
    for attempt in rejected:
        if attempt["action"] == "learn":
            resolution, a = learn(a, attempt["decision_id"], attempt["reward"])
        else:
            resolution, a = censor(a, attempt["decision_id"])
        assert resolution is None, f"rejected attempt resolved: {attempt}"
    predictions = []
    for seg in ["a", "b"]:
        for arm in arms:
            est, unc = predict(a.model, encode({"seg": seg}, arm, {}, bits))
            predictions.append({"seg": seg, "arm": arm,
                                "estimate": est, "uncertainty": unc})
    section = {
        "bits": bits, "forgetting": forgetting, "horizon": horizon,
        "default_reward": 0.0,
        "reward_rule": "1.0 if (arm == 'x') == (seg == 'a') else 0.0",
        "expire_sweep_at": sweep_t,
        "events": events,
        "resolutions": resolutions,
        "rejected": rejected,
        "final": {
            "model_version": a.model_version,
            "next_seq": a.next_seq,
            "open_ids": [r.decision_id for r in a.ledger],
            "model": model_json(a.model),
        },
        "predictions": predictions,
    }
    return section, mid_state, a


def serialization_vectors(mid_state, final_state):
    """The canonical byte serialization (state.py) as hex: a fresh state, the
    episode's mid snapshot (two open ledger records, covering the
    DecisionRecord encoding), and the episode's final state. Independent
    implementations must produce these exact bytes and accept them back."""
    fresh = new_bandit(4, horizon=60.0)
    for state in (fresh, mid_state, final_state):
        assert deserialize(serialize(state)) == state, "serialization must round-trip"
    return {
        "format_version": FORMAT_VERSION,
        "fresh": {"dim": 4, "horizon": 60.0, "bytes_hex": serialize(fresh).hex()},
        "episode_mid": {
            "open_ids": [r.decision_id for r in mid_state.ledger],
            "bytes_hex": serialize(mid_state).hex(),
        },
        "episode_final": {"bytes_hex": serialize(final_state).hex()},
    }


def generate() -> dict:
    episode, mid_state, final_state = run_episode()
    return {
        "format_version": 2,
        "sections": {
            "rng": rng_vectors(),
            "model": model_vectors(),
            "exploration": exploration_vectors(),
            "encoding": encoding_vectors(),
            "episode": episode,
            "serialization": serialization_vectors(mid_state, final_state),
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
