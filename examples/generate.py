"""Regenerates decisions.jsonl: a deterministic 1000-decision example
experience log (spec/ope.md), written the way an app writes one —
`api.log_line` output, verbatim, in arrival order. Rewards arrive three
decisions late, every 97th outcome is censored, some rewards never
arrive (the regular expiry sweep resolves them), and the reward rule is
learnable: segment "a" converts on the sale banner, segment "b" on the
plain banner in the evening.

    uv run python examples/generate.py

Try it against the CLI:

    gittins verify --log examples/decisions.jsonl
    gittins eval   --log examples/decisions.jsonl --bits 8
    gittins sweep  --log examples/decisions.jsonl --bits 8 --epsilon 0.02,0.05,0.1

Evaluating the logging configuration (bits 8, epsilon 0.05,
forgetfulness 0.999) shows the self-evaluation identity: ips == snips ==
logged mean, ess == resolved, max weight 1.
"""

from gittins_reference import api

state = api.create(bits=8, horizon=1800.0, epsilon=0.05, forgetfulness=0.999)
catalog = [
    ("banner-sale", {"discount": 0.2}),
    ("banner-new", {"discount": 0.0}),
    ("banner-plain", {}),
]
T0 = 1_752_000_000.0
lines = []
pending = []


def reward_for(record, context):
    arm = catalog[record.chosen][0]
    if context["seg"] == "a":
        return 1.0 if arm == "banner-sale" else 0.0
    return 1.0 if arm == "banner-plain" and context["hour"] >= 18 else 0.0


for i in range(1000):
    t = T0 + i * 30.0
    context = {"hour": float(i % 24), "seg": "a" if i % 3 else "b"}
    record = api.decide(state, context, catalog, t=t, salt="example")
    lines.append(api.log_line(record))
    pending.append((record, context))
    if len(pending) > 3:  # rewards arrive three decisions late
        due, due_context = pending.pop(0)
        if i % 97 == 0:  # an outage corrupted this outcome
            lines.append(api.log_line(api.censor(state, due.decision_id)))
        elif i % 11 != 0:  # and some rewards never arrive at all
            resolution = api.learn(state, due.decision_id, reward_for(due, due_context))
            lines.append(api.log_line(resolution))
    if i % 50 == 49:  # the regular expiry sweep
        lines.extend(api.log_line(r) for r in api.expire(state, t))
lines.extend(api.log_line(r) for r in api.expire(state, T0 + 1000 * 30.0 + 1800.0))

with open("examples/decisions.jsonl", "w", newline="\n") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote examples/decisions.jsonl: {len(lines)} lines")
