# Using gittins

Everything goes through one module, `gittins` — the Rust engine as a Python
package. It has eight functions, and the four steps below are the whole
integration: no schema to define, nothing to register, no background
machinery. (Until the wheel is on PyPI, install it from the repo with
`pip install ./bindings/python`. The pure-Python reference implementation
exposes the same eight functions as `gittins_reference.api`, so everything
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
optional settings, `epsilon` (how much it explores) and `forgetting` (how
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
