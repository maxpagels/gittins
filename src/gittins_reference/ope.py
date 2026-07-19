"""The experience log and offline policy evaluation.

The decision log *is* the offline-evaluation dataset: the app appends
every decision (with the dict-shaped inputs it was made over) and every
resolution, in arrival order, as one JSON object per line. This module
turns that log back into answers:

    read_log(path)                  -> events, streamed (gzip by magic bytes)
    parse_log(lines)                -> events, streamed (line-numbered errors)
    verify(events)                  -> every semantic violation, as strings
    evaluate(events, bits, ...)     -> progressive IPS/SNIPS report
    replay(events, bits, horizon, ...) -> a deployable BanditState

Reading is incremental end to end: `read_log`/`parse_log` are
generators that yield one event per log line (plain files read line by
line, gzipped files streamed through the decompressor), and
`verify`/`evaluate`/`replay` are single-pass walks over any event
iterable — so a log's size is bounded by disk, not memory. What they
hold is only their own bookkeeping: seen ids for exactly-once checks,
open decisions awaiting their resolution. Generators are consumed once;
wrap in `tuple` when events must be reused.

`verify` is the guard: it recomputes what the log claims —
`candidate_hash` over the re-encoded candidates, the chosen candidate's
features, propensity bounds, exactly-once resolutions, per-salt time
order — and collects every violation instead of stopping at the first.
An invalid evaluation should be impossible to run by accident, not
merely discouraged.

`evaluate` walks the log once, progressively: at each decision the
candidate configuration (bits, epsilon, forgetfulness — the built-in
engine's knobs) is scored with the model it would actually have had at
that moment, giving the importance weight `w = q / propensity`; at each
rewarded/expired resolution the logged reward feeds both the estimators
(IPS `sum(w*r)/n`, SNIPS `sum(w*r)/sum(w)`) and the candidate model's
training, at the resolution's logged position. Censored resolutions are
excluded from both, on record. Diagnostics (effective sample size, max
weight) travel with every estimate because an IPS number without them
is a trap. There is no clipping and no clipping knob: epsilon-greedy
logging bounds every weight structurally at k / epsilon.

`replay` is the fleet-pooling rebuild: the same walk, no estimator,
emitting a complete deployable state whose model absorbed every logged
outcome in order. Replay at the logging agent's own configuration
reproduces its model bit for bit — and therefore evaluating the logging
configuration yields w = 1 everywhere: ips == snips == logged_mean and
ess == resolved, exactly. The test suite pins both.

Everything here is fixed-order IEEE-754 over the engine's own layers
(encoding, model, exploration), so the same log and configuration give
bit-identical reports and replay bytes on every platform — pinned by
the golden `ope` section, which the CLI binding (`bindings/cli`) must
reproduce exactly.
"""

import gzip
import json
import math
from dataclasses import dataclass, replace

from gittins_reference.decide import DecisionRecord, candidate_set_hash, new_bandit
from gittins_reference.encoding import encode
from gittins_reference.exploration import DEFAULT_EPSILON, epsilon_greedy_probabilities
from gittins_reference.ledger import CENSORED, EXPIRED, REWARDED
from gittins_reference.model import (
    DEFAULT_FORGETTING,
    estimate_factored,
    factorize,
    new_model,
    update,
)

GZIP_MAGIC = b"\x1f\x8b"


@dataclass(frozen=True)
class DecisionEvent:
    line: int
    bits: int
    context: dict
    candidates: "tuple[tuple[str, dict], ...]"
    record: DecisionRecord


@dataclass(frozen=True)
class ResolutionEvent:
    line: int
    decision_id: str
    kind: str
    reward: "float | None"


@dataclass(frozen=True)
class OpeReport:
    """One evaluation's estimates and the diagnostics they must never be
    read without. Estimates are None when nothing resolved (and snips/ess
    when their denominators are zero)."""

    decisions: int
    resolved: int  # rewarded + expired: the estimator's n
    censored: int
    unresolved: int  # open at end of log
    logged_mean: "float | None"  # the logging policy's realized mean reward
    ips: "float | None"
    snips: "float | None"
    ess: "float | None"  # effective sample size, (sum w)^2 / sum w^2
    max_weight: "float | None"


def read_log(path):
    """One log file's events, yielded lazily — the file is read line by
    line and never held in memory whole, gzip-compressed (streamed
    through the decompressor) or plain, detected by the gzip magic
    bytes, never the file name. A generator: consume it once; wrap in
    `tuple` to materialize."""
    with open(path, "rb") as f:
        head = f.read(2)
    opener = gzip.open if head == GZIP_MAGIC else open
    with opener(path, "rt", encoding="utf-8") as f:
        yield from parse_log(f)


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _record(obj) -> DecisionRecord:
    features = obj["features"]
    if not isinstance(features, list) or not all(
        isinstance(p, list) and len(p) == 2 and isinstance(p[0], int) and _number(p[1])
        for p in features
    ):
        raise ValueError
    if not (
        isinstance(obj["decision_id"], str)
        and _number(obj["t"])
        and isinstance(obj["candidate_hash"], int)
        and isinstance(obj["chosen"], int)
        and _number(obj["propensity"])
        and isinstance(obj["model_version"], int)
        and isinstance(obj["salt"], str)
    ):
        raise ValueError
    return DecisionRecord(
        decision_id=obj["decision_id"],
        t=float(obj["t"]),
        candidate_hash=obj["candidate_hash"],
        chosen=obj["chosen"],
        features=tuple((j, float(v)) for j, v in features),
        propensity=float(obj["propensity"]),
        model_version=obj["model_version"],
        salt=obj["salt"],
    )


def _decision(n, obj) -> DecisionEvent:
    try:
        bits = obj["bits"]
        context = obj["context"]
        candidates = obj["candidates"]
        if not (isinstance(bits, int) and isinstance(context, dict) and isinstance(candidates, list)):
            raise ValueError
        if not all(
            isinstance(pair, list) and len(pair) == 2
            and isinstance(pair[0], str) and isinstance(pair[1], dict)
            for pair in candidates
        ):
            raise ValueError
        record = _record(obj["record"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"line {n}: malformed decision event") from None
    return DecisionEvent(
        line=n,
        bits=bits,
        context=context,
        candidates=tuple((arm_id, action) for arm_id, action in candidates),
        record=record,
    )


def _resolution(n, obj) -> ResolutionEvent:
    try:
        decision_id = obj["decision_id"]
        kind = obj["kind"]
        reward = obj["reward"]
        if not (isinstance(decision_id, str) and isinstance(kind, str)):
            raise ValueError
        if reward is not None and not _number(reward):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"line {n}: malformed resolution event") from None
    return ResolutionEvent(
        line=n,
        decision_id=decision_id,
        kind=kind,
        reward=None if reward is None else float(reward),
    )


def parse_log(lines):
    """Each nonempty line as one event, yielded as it is read — a
    generator over any line iterable (a file handle included), so a log
    is never held in memory whole. Structural problems — bad JSON,
    unknown kinds, missing or mistyped fields — are hard errors naming
    the line, raised when that line is reached; semantic problems are
    `verify`'s job. Unknown fields are ignored (forward compatibility)."""
    for n, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except ValueError:
            raise ValueError(f"line {n}: not valid JSON") from None
        if not isinstance(obj, dict) or not isinstance(obj.get("event"), str):
            raise ValueError(f"line {n}: each event must be a JSON object with an 'event' kind")
        if obj["event"] == "decision":
            yield _decision(n, obj)
        elif obj["event"] == "resolution":
            yield _resolution(n, obj)
        else:
            raise ValueError(f"line {n}: unknown event kind")


def verify(events) -> "tuple[str, ...]":
    """Every semantic violation in the log, as `line N: ...` strings in
    event order; empty means the log is trustworthy input for `evaluate`
    and `replay`. One pass over any event iterable (a `read_log`
    generator included). Decisions left unresolved at end of log are
    normal (open at capture time), not a violation."""
    problems = []
    seen = set()  # decision ids
    resolved = set()
    last_t = {}  # salt -> latest decision t
    for e in events:
        if isinstance(e, DecisionEvent):
            r = e.record
            where = f"line {e.line}: decision {r.decision_id}"
            if r.decision_id in seen:
                problems.append(f"line {e.line}: duplicate decision id '{r.decision_id}'")
            seen.add(r.decision_id)
            if not (1 <= e.bits <= 24):
                problems.append(f"{where}: bits must be between 1 and 24")
                continue
            if not (0 <= r.chosen < len(e.candidates)):
                problems.append(f"{where}: chosen index out of range")
                continue
            if not (0.0 < r.propensity <= 1.0):
                problems.append(f"{where}: propensity must be in (0, 1]")
            try:
                encoded = [encode(e.context, arm, action, e.bits) for arm, action in e.candidates]
            except ValueError:
                problems.append(f"{where}: candidates failed to encode")
                continue
            if candidate_set_hash(encoded, 1 << e.bits) != r.candidate_hash:
                problems.append(f"{where}: candidates do not match the logged candidate_hash")
            if tuple(encoded[r.chosen]) != r.features:
                problems.append(f"{where}: chosen features do not match the logged candidates")
            if r.salt in last_t and r.t < last_t[r.salt]:
                problems.append(f"{where}: t decreases within salt '{r.salt}'")
            last_t[r.salt] = max(last_t.get(r.salt, r.t), r.t)
        else:
            where = f"line {e.line}: resolution for"
            if e.decision_id not in seen:
                problems.append(f"{where} unknown decision '{e.decision_id}'")
                continue
            if e.decision_id in resolved:
                problems.append(f"line {e.line}: second resolution for decision '{e.decision_id}'")
                continue
            resolved.add(e.decision_id)
            if e.kind not in (REWARDED, EXPIRED, CENSORED):
                problems.append(f"{where} {e.decision_id}: unknown kind '{e.kind}'")
            elif e.kind == CENSORED:
                if e.reward is not None:
                    problems.append(f"{where} {e.decision_id}: censored resolutions carry no reward")
            elif e.reward is None or not math.isfinite(e.reward):
                problems.append(f"{where} {e.decision_id}: reward must be a finite number")
    return tuple(problems)


def evaluate(
    events,
    bits: int,
    epsilon: float = DEFAULT_EPSILON,
    forgetfulness: float = DEFAULT_FORGETTING,
) -> OpeReport:
    """Progressive IPS/SNIPS over the log for one candidate configuration
    of the built-in engine (module docstring). One pass over
    any event iterable. Strict where `verify` is lenient: malformed
    decisions raise rather than skew the estimate — run `verify` first."""
    if not (1 <= bits <= 24):
        raise ValueError("bits must be between 1 and 24")
    model = new_model(1 << bits, forgetfulness)
    pending = {}  # decision_id -> (chosen encoding at candidate bits, weight)
    decisions = resolved = censored = 0
    sum_r = sum_w = sum_wr = sum_w2 = 0.0
    max_weight = 0.0
    for e in events:
        if isinstance(e, DecisionEvent):
            decisions += 1
            r = e.record
            if len(e.candidates) < 1:
                raise ValueError(f"line {e.line}: need at least one candidate")
            if not (0 <= r.chosen < len(e.candidates)):
                raise ValueError(f"line {e.line}: chosen index out of range")
            if not (0.0 < r.propensity <= 1.0):
                raise ValueError(f"line {e.line}: propensity must be in (0, 1]")
            encoded = [encode(e.context, arm, action, bits) for arm, action in e.candidates]
            factored = factorize(model)
            estimates = [estimate_factored(factored, x) for x in encoded]
            p = epsilon_greedy_probabilities(estimates, epsilon)
            w = p[r.chosen] / r.propensity
            pending[r.decision_id] = (encoded[r.chosen], w)
        else:
            got = pending.pop(e.decision_id, None)
            if got is None:
                continue
            if e.kind == CENSORED:
                censored += 1
                continue
            if e.kind not in (REWARDED, EXPIRED):
                raise ValueError(f"line {e.line}: unknown resolution kind")
            if e.reward is None or not math.isfinite(e.reward):
                raise ValueError(f"line {e.line}: reward must be a finite number")
            x, w = got
            reward = e.reward
            resolved += 1
            sum_r += reward
            sum_w += w
            sum_wr += w * reward
            sum_w2 += w * w
            if w > max_weight:
                max_weight = w
            model = update(model, x, reward)
    if resolved == 0:
        return OpeReport(decisions, 0, censored, len(pending), None, None, None, None, None)
    return OpeReport(
        decisions=decisions,
        resolved=resolved,
        censored=censored,
        unresolved=len(pending),
        logged_mean=sum_r / resolved,
        ips=sum_wr / resolved,
        snips=(sum_wr / sum_w) if sum_w > 0.0 else None,
        ess=((sum_w * sum_w) / sum_w2) if sum_w2 > 0.0 else None,
        max_weight=max_weight,
    )


def replay(
    events,
    bits: int,
    horizon: float,
    default_reward: float = 0.0,
    epsilon: float = DEFAULT_EPSILON,
    forgetfulness: float = DEFAULT_FORGETTING,
):
    """The fleet-pooling rebuild: a fresh, deployable BanditState
    whose model absorbed every rewarded/expired resolution's (re-encoded
    chosen candidate, logged reward), in log order. Expired resolutions
    train with their *logged* reward; `default_reward` here only governs
    the rebuilt state's future expiries. Serialize the result and ship it."""
    if not (1 <= bits <= 24):
        raise ValueError("bits must be between 1 and 24")
    state = new_bandit(
        1 << bits,
        horizon=horizon,
        default_reward=default_reward,
        epsilon=epsilon,
        forgetting=forgetfulness,
    )
    model = state.model
    pending = {}
    absorbed = 0
    for e in events:
        if isinstance(e, DecisionEvent):
            r = e.record
            if not (0 <= r.chosen < len(e.candidates)):
                raise ValueError(f"line {e.line}: chosen index out of range")
            pending[r.decision_id] = encode(
                e.context, e.candidates[r.chosen][0], e.candidates[r.chosen][1], bits
            )
        else:
            x = pending.pop(e.decision_id, None)
            if x is None or e.kind == CENSORED:
                continue
            if e.kind not in (REWARDED, EXPIRED):
                raise ValueError(f"line {e.line}: unknown resolution kind")
            if e.reward is None or not math.isfinite(e.reward):
                raise ValueError(f"line {e.line}: reward must be a finite number")
            model = update(model, x, e.reward)
            absorbed += 1
    return replace(state, model=model, model_version=absorbed)
