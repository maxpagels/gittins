# Using the reference API

Everything goes through one module, `gittins_reference.api`. It has eight
functions, and the four steps below are the whole integration — there is no
schema to define, nothing to register, and no background machinery.

Two things to know before the code makes sense:

- **The bandit's state is a plain value.** Every call returns a new state;
  keep the latest one. Saving, copying, or rolling back a bandit is just
  handling that value.
- **You supply the time.** Pass your own clock's `t` (seconds, e.g.
  `time.time()`) into `decide` and `expire`. The engine never looks at a
  clock itself, so any run can be replayed exactly.

## Setting up

```python
from gittins_reference import api

state = api.create(bits=8, horizon=3600.0)
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

record, state = api.decide(state, context, candidates, t=1_752_000_000.0, salt="agent-1")

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
resolution, state = api.learn(state, record.decision_id, reward=1.0)

# Call this regularly with the current time; decisions that waited past
# the horizon are trained as default_reward:
resolutions, state = api.expire(state, t=1_752_003_600.0)

# Throw a decision out of training, but keep that fact on record
# (say, an outage corrupted the outcome):
resolution, state = api.censor(state, record.decision_id)
```

Each call tells you what it did, so you can log it. Reporting the same
decision twice does nothing and returns `None` — no double-counting. And a
missing reward is never quietly counted as zero: it only becomes a value
when `expire` runs, at the deadline you chose.

## Saving and loading

The whole state becomes one byte string, and the exact same bytes come out
on any machine:

```python
from pathlib import Path

Path("bandit.state").write_bytes(api.serialize(state))
# ... later, or on another machine:
state = api.deserialize(Path("bandit.state").read_bytes())
```

`deserialize` checks everything (a checksum plus every internal
consistency rule) and raises `ValueError` rather than load anything
corrupt. The file format is shared with the Rust core, so a state saved
here loads there unchanged — small enough and stable enough to commit to
version control alongside your code.
