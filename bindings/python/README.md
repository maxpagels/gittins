# gittins

An opinionated, highly optimised contextual bandit engine.

- **Online by nature.** Learns one observation at a time, in O(1) work and
  fixed memory, as long as open decisions are regularly resolved.
- **Non-stationarity is expected.** The engine adapts as the relationship
  between context and feedback drifts, and never learns anything it cannot
  eventually unlearn.
- **Dynamic actions and context.** The candidate set can change on every
  call; you never declare the number of actions up front.
- **Safe reward handling.** Rewards may arrive late or never. Constructing
  invalid training data is made hard by design.
- **Speed and determinism.** Bit-identical results across platforms and
  language bindings, enforced by a golden test corpus.
- **Bring your own model.** Swap in your own scoring or exploration and
  inherit everything else.

Full documentation and user guide: **[docs.getgittins.dev](https://docs.getgittins.dev)**

## Install

```sh
pip install gittins
```

## Usage

The state is an opaque handle updated **in place** — calls return only their
result, and every alias of the handle observes the current state. Persist it
with `serialize`, which returns one plain string.

```python
import time
import gittins

state = gittins.create(bits=8, horizon=3600.0)  # model size, seconds to resolve

candidates = [
    ("banner-sale", {"discount": 0.2}),
    ("banner-new", {"discount": 0.0}),
    ("banner-plain", {}),
]
context = {"device": "mobile", "hour": 14}

record = gittins.decide(state, context, candidates, time.time(), "web-1")
arm_id = candidates[record.chosen][0]
# ... your code: act on the choice, then report the outcome

gittins.learn(state, record.decision_id, 1.0, time.time())
gittins.expire(state, time.time())  # resolve anything past its horizon

open("bandit.txt", "w").write(gittins.serialize(state))
```

Feature values are typed by what you pass: strings are categorical, ints,
floats and bools numeric, `None` absent. Anything else raises `ValueError`
naming the feature.

### Bringing your own model

`decide` takes optional `score` and `explore` callbacks, and `learn`/`expire`
take `train`. Each crosses the Python/Rust boundary once per call — with all
candidates, all estimates, or the one resolved record — never once per
candidate, so the one boundary crossing per decision is preserved.

### Offline policy evaluation

`log_line` renders a decision record or resolution as one canonical
experience-log line. Append it verbatim; it is exactly what the `gittins`
CLI's `verify` / `eval` / `sweep` / `replay` consume.

```python
with open("decisions.jsonl", "a") as log:
    log.write(gittins.log_line(record) + "\n")
```

## API

| Function | Purpose |
| --- | --- |
| `create(bits, horizon, default_reward=0.0, epsilon=..., forgetfulness=...)` | New state |
| `decide(state, context, candidates, t, salt, score=None, explore=None)` | Choose; returns a `DecisionRecord` |
| `learn(state, decision_id, reward, t, train=None)` | Resolve one decision; returns a `Resolution` or `None` |
| `expire(state, t, train=None)` | Resolve everything past its horizon |
| `serialize(state)` / `deserialize(text)` | State as one plain string |
| `model_bits(state)` | The model's size in bits |
| `log_line(record_or_resolution)` | One canonical experience-log line |

## License

MIT. Named after John Gittins, whose index (1974) established that
exploration has a precise, computable value.
