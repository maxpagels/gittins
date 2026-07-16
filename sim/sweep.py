"""`python -m sim.sweep`: sweep the engine's gamma scale across the battery.

The engine deliberately has no user knobs; its fixed constants are settled
by evidence instead. This driver produces the evidence for `GAMMA_SCALE`
(decide.py's greediness schedule, gamma = GAMMA_SCALE / mean uncertainty):
it re-runs every battery environment — and the event-time configurations —
with the engine patched to each candidate value, and prints a markdown
report of normalized regret per (environment, value) with the
per-environment best marked and epsilon-greedy 0.1 alongside as the
"would something dumber beat us" reference.

The summary applies the battery's pass criteria to each candidate as a
would-be default: within 10% of the per-environment best on at least 90%
of environments, never catastrophically worse (2x) anywhere, and beating
epsilon-greedy overall. A chosen value lands in decide.py as the new
constant, with the golden vectors regenerated and the diff reviewed like
code.

This is a decision-making tool, run by hand — it is not part of CI, and
patching the engine's constant is exactly what it exists to do. The
schedule's aggregate (mean vs min/max), `ridge`, and the default half-life
still await sweeps of their own.
"""

import math
import sys
import time

import gittins_reference.decide as decide_mod
from sim.__main__ import (
    BITS,
    ENVIRONMENTS,
    EVENT_CONFIGS,
    EVENT_DURATION,
    EVENT_ENV,
    EVENT_HALF_LIFE,
    EVENT_TRAFFIC,
    ROUNDS,
    SEEDS,
)
from sim.event_runner import run_events
from sim.metrics import median_iqr, normalized_regret
from sim.policies import EpsilonGreedyPolicy, GittinsPolicy
from sim.runner import run

GAMMA_SCALES = [0.25, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]

# Pass criteria for a would-be default.
NEAR_BEST = 1.1
CATASTROPHIC = 2.0


def cells() -> "list[tuple[str, callable, callable]]":
    """(label, gittins run, epsilon run) per battery cell: the round-based
    environments plus the event-time configurations, each returning one
    median normalized regret over the battery's seeds."""
    out = []
    for env in ENVIRONMENTS:

        def rounds_cell(env=env, eps=False):
            regrets = []
            for seed in SEEDS:
                policy = (
                    EpsilonGreedyPolicy(0.1, bits=BITS) if eps else GittinsPolicy(bits=BITS)
                )
                regrets.append(normalized_regret(run(env, policy, seed, ROUNDS)))
            return median_iqr(regrets)[0]

        out.append((env.name, rounds_cell, lambda env=env: rounds_cell(env, eps=True)))
    for label, delay, horizon in EVENT_CONFIGS:

        def event_cell(delay=delay, horizon=horizon, eps=False):
            regrets = []
            for seed in SEEDS:
                policy = (
                    EpsilonGreedyPolicy(0.1, bits=BITS, half_life=EVENT_HALF_LIFE)
                    if eps
                    else GittinsPolicy(bits=BITS, half_life=EVENT_HALF_LIFE, horizon=horizon)
                )
                regrets.append(
                    normalized_regret(
                        run_events(EVENT_ENV, policy, seed, EVENT_TRAFFIC, delay, EVENT_DURATION)
                    )
                )
            return median_iqr(regrets)[0]

        out.append(
            (f"event: {label}", event_cell, lambda d=delay, h=horizon: event_cell(d, h, eps=True))
        )
    return out


def main() -> None:
    start = time.time()
    table: "dict[str, dict[float, float]]" = {}
    epsilon: "dict[str, float]" = {}
    battery = cells()

    original = decide_mod.GAMMA_SCALE
    try:
        for scale in GAMMA_SCALES:
            decide_mod.GAMMA_SCALE = scale
            for label, gittins_cell, _ in battery:
                table.setdefault(label, {})[scale] = gittins_cell()
    finally:
        decide_mod.GAMMA_SCALE = original
    for label, _, epsilon_cell in battery:
        epsilon[label] = epsilon_cell()

    print("### GAMMA_SCALE sweep")
    print()
    print(
        f"Median normalized regret over seeds {list(SEEDS)}; round environments at "
        f"{ROUNDS} rounds, event configurations as in the battery. Bold marks each "
        f"environment's best value; epsilon-0.1 is the same-model baseline. "
        f"Current constant: {original:g}."
    )
    print()
    header = " | ".join(f"{s:g}" for s in GAMMA_SCALES)
    print(f"| environment | {header} | eps-0.1 |")
    print("|---|" + "---|" * (len(GAMMA_SCALES) + 1))
    for label, row in table.items():
        best = min(row.values())
        cells_text = " | ".join(
            f"**{v:.3f}**" if v == best else f"{v:.3f}" for v in row.values()
        )
        print(f"| {label} | {cells_text} | {epsilon[label]:.3f} |")

    print()
    print("### As a default")
    print()
    print(
        f"Per candidate: environments within {NEAR_BEST:g}x of that environment's "
        f"best swept value, worst ratio to best, environments beaten by epsilon, "
        f"and the mean regret across all environments."
    )
    print()
    print("| GAMMA_SCALE | near-best | worst ratio | loses to eps | mean regret |")
    print("|---|---|---|---|---|")
    n = len(table)
    for scale in GAMMA_SCALES:
        ratios = []
        near = 0
        loses = 0
        for label, row in table.items():
            best = min(row.values())
            ratio = row[scale] / best if best > 0 else math.inf if row[scale] > 0 else 1.0
            ratios.append(ratio)
            near += ratio <= NEAR_BEST
            loses += row[scale] > epsilon[label]
        mean = sum(row[scale] for row in table.values()) / n
        print(
            f"| {scale:g} | {near}/{n} | {max(ratios):.2f} | {loses}/{n} | {mean:.3f} |"
        )
    print()
    print(f"_{time.time() - start:.0f}s on {sys.platform}, Python {sys.version.split()[0]}._")


if __name__ == "__main__":
    main()
