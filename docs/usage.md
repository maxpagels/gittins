# Using gittins

Everything goes through one module, `gittins` — the Rust engine as a Python
package. It has nine functions, and the four steps below are the whole
integration: no schema to define, nothing to register, no background
machinery. (Until the wheel is on PyPI, install it from the repo with
`pip install ./bindings/python`. The pure-Python reference implementation
exposes the same nine functions as `gittins_reference.api`, so everything
on this page works there too.)

Two things to know before the code makes sense:

- **The state is a handle, updated in place.** `create` gives you one;
  `decide`, `learn`, and `expire` update it as they go and hand back just
  their results. To save, copy, or roll back, snapshot it with
  `serialize` — the whole state is one plain string.
- **You supply the time.** Pass your own clock's `t` (seconds, e.g.
  `time.time()`) into `decide` and `expire`. The engine never looks at a
  clock itself, so any run can be replayed exactly.

## Setting up

```python
import gittins

state = gittins.create(bits=8, horizon=3600.0)
```

`bits` sets how much room the model has for features: it learns in a space
of 2**bits slots (here 256), and every feature and arm is hashed into it.
That's why nothing needs registering — a feature name or arm the model has
never seen before just works the first time it shows up.

`horizon` answers "how long do we wait for a reward?". If a decision gets
no reward within `horizon` seconds, it's treated as having earned
`default_reward` (0.0 unless you say otherwise). There are two more
optional settings, `epsilon` (how much it explores) and `forgetfulness` (how
fast old evidence fades); the defaults are meant to be left alone.

## Making a decision

```python
context = {"device": "mobile", "hour": 14}          # whatever you know right now
candidates = [                                       # (arm id, that arm's features)
    ("banner-sale", {"discount": 0.2}),
    ("banner-new", {"discount": 0.0}),
    ("banner-plain", {}),
]

record = gittins.decide(state, context, candidates, t=1_752_000_000.0, salt="agent-1")

chosen_arm_id = candidates[record.chosen][0]         # act on this
```

Feature dicts take strings for categories, numbers for quantities (bools
count as 1/0), and `None` for "not available this time". `salt` is the
agent's name — if you run several agents, give each its own so their
decision ids never collide.

What you get back is a **record** of the decision, not just a pick: its id,
which candidate was chosen, and the probability it was chosen with. Keep
these records (append them to a log file, say). The id is how you report
the outcome later, and the log itself becomes the dataset for judging other
strategies offline.

## Reporting outcomes

The model learns only when you resolve a decision, and each decision can
only be resolved once:

```python
# The reward came in (late or out of order is fine):
resolution = gittins.learn(state, record.decision_id, reward=1.0)

# Call this regularly with the current time; decisions that waited past
# the horizon are trained as default_reward:
resolutions = gittins.expire(state, t=1_752_003_600.0)

# Throw a decision out of training, but keep that fact on record
# (say, an outage corrupted the outcome):
resolution = gittins.censor(state, record.decision_id)
```

Each call tells you what it did, so you can log it. Reporting the same
decision twice does nothing and returns `None` — no double-counting. And a
missing reward is never quietly counted as zero: it only becomes a value
when `expire` runs, at the deadline you chose.

## Bringing your own model or exploration rule

The three optional callbacks are the whole story: `score` and `explore`
on `decide`, `train` on `learn` and `expire`. Each one replaces exactly
one built-in piece — everything else (feature encoding, deterministic
sampling, decision records, the exactly-once reward ledger, offline
evaluation) keeps working unchanged. Callbacks are passed per call and
never stored, so saving and loading the state is exactly as before.

To swap in your own prediction model, supply `score` (called once per
decision, with the same `context` and `candidates` you passed in; return
one estimated reward per candidate) and `train` (called once per resolved
decision with the decision's record and its reward — the engine has
already matched the reward to the right decision, exactly once):

```python
model = MyModel()  # anything with predict/update

record = gittins.decide(
    state, context, candidates, t=..., salt="agent-1",
    score=lambda ctx, cands: [model.predict(ctx, action) for _, action in cands],
)

gittins.learn(
    state, record.decision_id, reward=1.0,
    train=lambda rec, reward: model.update(rec.decision_id, reward),
)
# Give expire the same callback so timed-out decisions train too:
gittins.expire(state, t=..., train=lambda rec, reward: model.update(rec.decision_id, reward))
```

To swap in your own exploration rule, supply `explore`: it gets the
estimates (yours or the built-in model's) plus the configured `epsilon`,
and returns one probability per candidate. The engine still does the
random draw itself — deterministically, from the decision's own RNG
stream — and logs the probability the choice was made with, so replay
and offline evaluation stay trustworthy:

```python
def explore(estimates, epsilon):  # e.g. more uniform than epsilon-greedy
    k = len(estimates)
    best = max(range(k), key=lambda i: estimates[i])
    return [0.5 + 0.5 / k if i == best else 0.5 / k for i in range(k)]

record = gittins.decide(state, context, candidates, t=..., salt="agent-1", explore=explore)
```

Two rules to remember. The probabilities must be nonnegative and sum
to 1 — the engine rejects anything else, because a wrong distribution
would poison the decision log. And `train` runs *after* a decision is
marked resolved: if your callback crashes, that one observation is lost
(loudly — the exception reaches you), but it can never be trained twice.

## Logging decisions, and asking "what would have happened?"

Logging is appending what each call returns, verbatim. The record
`decide` hands back carries everything the decision was made over (the
context, the candidates, the encoding declaration), and `log_line`
turns it — or any resolution — into one canonical log line:

```python
with open("decisions.jsonl", "a") as f:
    record = gittins.decide(state, context, candidates, t=..., salt="agent-1")
    f.write(gittins.log_line(record) + "\n")
    # ...later, when outcomes arrive:
    resolution = gittins.learn(state, record.decision_id, reward=1.0)
    f.write(gittins.log_line(resolution) + "\n")
    for r in gittins.expire(state, t=...):
        f.write(gittins.log_line(r) + "\n")
```

That file — decisions and resolutions in the order they happened — is a
complete offline-evaluation dataset. (One nuance: the inputs ride only
on the record `decide` returns; the engine's state never stores them,
so log decisions when you make them.)

The `gittins` command-line tool (built from `bindings/cli`) then answers
the tuning questions offline, from the log alone — no bindings, no
notebook required. Logs may be gzip-compressed; the tool detects it by
content:

```sh
# is the log internally consistent? (recomputes every hash; exits nonzero if not)
gittins verify --log decisions.jsonl.gz

# how would a different configuration have done on this exact traffic?
gittins eval --log decisions.jsonl.gz --bits 8 --epsilon 0.1

# compare a whole grid in one table
gittins sweep --log decisions.jsonl.gz --bits 8 --epsilon 0.02,0.05,0.1 --forgetfulness 0.999,0.995

# any log-consuming command accepts --hard-fail: run the full verify
# pass first and refuse to continue (exit 1) on any violation
gittins eval --log decisions.jsonl.gz --bits 8 --epsilon 0.1 --hard-fail

# rebuild a deployable state from the log (fleet pooling: merge logs, replay, ship)
gittins replay --log decisions.jsonl.gz --bits 8 --horizon 3600 > bandit.state
```

`eval` reports IPS and SNIPS estimates of mean reward next to the logged
policy's realized mean — and always with the diagnostics (effective
sample size, largest importance weight) that tell you how much to trust
them. Every command is deterministic: the same log and settings produce
the same numbers on every machine. And every command streams: the log is
read one line at a time (gzip included), so its size is bounded by your
disk, not your memory. The same functions are available in Python as
`gittins_reference.ope` (`read_log`, `verify`, `evaluate`, `replay` —
`read_log` is a generator) for notebook use.

## Saving and loading

The whole state becomes one plain string (hex-encoded), and the exact same
string comes out on any machine — put it in a file, a database column,
localStorage, or version control as-is; there is never a byte-handling or
encoding step on your side:

```python
from pathlib import Path

Path("bandit.state").write_text(gittins.serialize(state))
# ... later, or on another machine:
state = gittins.deserialize(Path("bandit.state").read_text())
```

`deserialize` checks everything (a checksum plus every internal
consistency rule) and raises `ValueError` rather than load anything
corrupt. The format is shared across implementations — a state saved by
the Rust engine loads in the pure-Python reference or in a browser and
vice versa — and is small enough and stable enough to commit to version
control alongside your code.
