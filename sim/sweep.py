"""`python -m sim.sweep`: sweep the engine's epsilon default across the battery.

The engine deliberately has no user knobs; its fixed constants are settled
by evidence instead. This driver produces the evidence for
`DEFAULT_EPSILON` (exploration.py's uniform exploration mass): it re-runs
every battery environment — and the event-time configurations — with the
engine constructed at each candidate value, and prints a markdown report of
normalized regret per (environment, value) with the per-environment best
marked and the no-ledger epsilon-0.1 baseline alongside as the "would
something dumber beat us" reference.

The summary applies the battery's pass criteria to each candidate as a
would-be default: within 10% of the per-environment best on at least 90%
of environments, never catastrophically worse (2x) anywhere, and beating
the baseline overall. A chosen value lands in exploration.py as the new
constant, with the golden vectors regenerated and the diff reviewed like
code.

This is a decision-making tool, run by hand — it is not part of CI.
`ridge` and the default forgetting rate still await sweeps of their own.
"""

import math
import sys
import time

from sim.__main__ import (
    BITS,
    ENVIRONMENTS,
    EVENT_CONFIGS,
    EVENT_DURATION,
    EVENT_ENV,
    EVENT_TRAFFIC,
    ROUNDS,
    SEEDS,
)
from sim.event_runner import run_events
from sim.metrics import median_iqr, normalized_regret
from sim.policies import EpsilonGreedyPolicy, GittinsPolicy
from sim.runner import run

EPSILONS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

# Pass criteria for a would-be default.
NEAR_BEST = 1.1
CATASTROPHIC = 2.0


def cells() -> "list[tuple[str, callable, callable]]":
    """(label, gittins run, baseline run) per battery cell: the round-based
    environments plus the event-time configurations, each returning one
    median normalized regret over the battery's seeds."""
    out = []
    for env in ENVIRONMENTS:

        def rounds_cell(epsilon, env=env, baseline=False):
            regrets = []
            for seed in SEEDS:
                policy = (
                    EpsilonGreedyPolicy(0.1, bits=BITS)
                    if baseline
                    else GittinsPolicy(bits=BITS, epsilon=epsilon)
                )
                regrets.append(normalized_regret(run(env, policy, seed, ROUNDS)))
            return median_iqr(regrets)[0]

        out.append((env.name, rounds_cell, lambda env=env: rounds_cell(0.0, env, baseline=True)))
    for label, delay, horizon in EVENT_CONFIGS:

        def event_cell(epsilon, delay=delay, horizon=horizon, baseline=False):
            regrets = []
            for seed in SEEDS:
                policy = (
                    EpsilonGreedyPolicy(0.1, bits=BITS)
                    if baseline
                    else GittinsPolicy(bits=BITS, horizon=horizon, epsilon=epsilon)
                )
                regrets.append(
                    normalized_regret(
                        run_events(EVENT_ENV, policy, seed, EVENT_TRAFFIC, delay, EVENT_DURATION)
                    )
                )
            return median_iqr(regrets)[0]

        out.append(
            (
                f"event: {label}",
                event_cell,
                lambda d=delay, h=horizon: event_cell(0.0, d, h, baseline=True),
            )
        )
    return out


def main() -> None:
    start = time.time()
    table: "dict[str, dict[float, float]]" = {}
    baseline: "dict[str, float]" = {}
    battery = cells()

    for epsilon in EPSILONS:
        for label, gittins_cell, _ in battery:
            table.setdefault(label, {})[epsilon] = gittins_cell(epsilon)
    for label, _, baseline_cell in battery:
        baseline[label] = baseline_cell()

    print("### DEFAULT_EPSILON sweep")
    print()
    print(
        f"Median normalized regret over seeds {list(SEEDS)}; round environments at "
        f"{ROUNDS} rounds, event configurations as in the battery. Bold marks each "
        f"environment's best value; eps-0.1 is the same-model, no-ledger baseline."
    )
    print()
    header = " | ".join(f"{e:g}" for e in EPSILONS)
    print(f"| environment | {header} | eps-0.1 |")
    print("|---|" + "---|" * (len(EPSILONS) + 1))
    for label, row in table.items():
        best = min(row.values())
        cells_text = " | ".join(
            f"**{v:.3f}**" if v == best else f"{v:.3f}" for v in row.values()
        )
        print(f"| {label} | {cells_text} | {baseline[label]:.3f} |")

    print()
    print("### As a default")
    print()
    print(
        f"Per candidate: environments within {NEAR_BEST:g}x of that environment's "
        f"best swept value, worst ratio to best, environments beaten by the "
        f"baseline, and the mean regret across all environments."
    )
    print()
    print("| DEFAULT_EPSILON | near-best | worst ratio | loses to eps-0.1 | mean regret |")
    print("|---|---|---|---|---|")
    n = len(table)
    for epsilon in EPSILONS:
        ratios = []
        near = 0
        loses = 0
        for label, row in table.items():
            best = min(row.values())
            ratio = row[epsilon] / best if best > 0 else math.inf if row[epsilon] > 0 else 1.0
            ratios.append(ratio)
            near += ratio <= NEAR_BEST
            loses += row[epsilon] > baseline[label]
        mean = sum(row[epsilon] for row in table.values()) / n
        print(
            f"| {epsilon:g} | {near}/{n} | {max(ratios):.2f} | {loses}/{n} | {mean:.3f} |"
        )
    print()
    print(f"_{time.time() - start:.0f}s on {sys.platform}, Python {sys.version.split()[0]}._")


if __name__ == "__main__":
    main()
