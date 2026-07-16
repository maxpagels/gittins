"""`python -m sim`: run the batteries and print markdown tables.

Two tables: the round-based environment battery (every world x every
comparator), and the event-time battery (daily traffic and delayed rewards
driving the same policies through the event-queue runner, with phase-split
regret and the reward-plumbing diagnostics). CI appends the output to the
GitHub job summary, so every run is readable directly in the Actions UI.
Runs are seeded and paired (see sim.rand): the numbers are exactly
reproducible for a given harness version, but they are statistics, not
golden vectors — nothing here is bit-pinned, and these tables are a
diagnostic, not a gate. The sweep driver and the full battery report
belong to PR 14.
"""

import sys
import time

from sim.environments import (
    AbruptShiftEnvironment,
    ActionFeatureEnvironment,
    ChurnEnvironment,
    DriftEnvironment,
    DropoutEnvironment,
    LinearEnvironment,
    NeedleEnvironment,
    XorEnvironment,
)
from sim.event_runner import run_events
from sim.metrics import final_window_regret, median_iqr, normalized_regret, phase_regret, rmse
from sim.policies import (
    EpsilonGreedyPolicy,
    GittinsPolicy,
    GreedyPolicy,
    OraclePolicy,
    UniformPolicy,
)
from sim.runner import run
from sim.traffic import DAY, HOUR, DailyTraffic, ExponentialDelay, NextMorningDelay, phase

SEEDS = range(5)
ROUNDS = 1500
BITS = 8

ENVIRONMENTS = [
    LinearEnvironment(k=5),
    LinearEnvironment(k=20),
    XorEnvironment(k=4),
    XorEnvironment(k=10),
    NeedleEnvironment(k=10),
    ActionFeatureEnvironment(k=16),
    # PR 12: non-stationary, churn, and missing-feature worlds. The policy
    # half-life is 1000 rounds: shift period 375 is well inside it (four
    # epochs per 1500-round run), drift period 500 is seasonal (three full
    # cycles), churn's events land at rounds 400/800/1100.
    AbruptShiftEnvironment(k=5, period=375),
    DriftEnvironment(k=5, period=500),
    ChurnEnvironment(k=8),
    DropoutEnvironment(k=5, p_drop=0.3),
]

POLICIES = [
    OraclePolicy,
    UniformPolicy,
    lambda: GreedyPolicy(bits=BITS),
    lambda: EpsilonGreedyPolicy(0.05, bits=BITS),
    lambda: EpsilonGreedyPolicy(0.1, bits=BITS),
    lambda: GittinsPolicy(bits=BITS),
]

# The event-time battery (PR 13): linear-k5 under daily traffic (two peaks,
# overnight trough, one burst window a day) with wall-clock half-life and
# horizon. Config A: fast rewards with a tail past the 2h horizon (~7%
# expire). Config B: every reward lands the next morning — the horizon must
# span the night, so the ledger holds a whole day of open decisions.
EVENT_ENV = LinearEnvironment(k=5)
EVENT_TRAFFIC = DailyTraffic(peak_rate=0.04, trough_rate=0.002, bursts=1)
EVENT_DURATION = 2 * DAY
EVENT_HALF_LIFE = 6 * HOUR
EVENT_CONFIGS = [
    ("exp 45m, horizon 2h", ExponentialDelay(45 * 60.0), 2 * HOUR),
    ("next morning, horizon 26h", NextMorningDelay(), 26 * HOUR),
]


def event_policies(horizon: float) -> list:
    return [
        OraclePolicy,
        UniformPolicy,
        lambda: GreedyPolicy(bits=BITS, half_life=EVENT_HALF_LIFE),
        lambda: EpsilonGreedyPolicy(0.05, bits=BITS, half_life=EVENT_HALF_LIFE),
        lambda: EpsilonGreedyPolicy(0.1, bits=BITS, half_life=EVENT_HALF_LIFE),
        lambda: GittinsPolicy(bits=BITS, half_life=EVENT_HALF_LIFE, horizon=horizon),
    ]


def spread(values: "list[float]") -> str:
    median, q1, q3 = median_iqr(values)
    return f"{median:.3f} [{q1:.3f}, {q3:.3f}]"


def main() -> None:
    start = time.time()
    print("### Environment battery")
    print()
    print(
        f"{ROUNDS} rounds, seeds {list(SEEDS)}, bits={BITS}. Normalized regret: "
        "0 = oracle, 1 = uniform (median [IQR] over seeds). RMSE is late-run "
        "(final half) prediction error vs the oracle means."
    )
    print()
    print("| environment | policy | normalized regret | final 10% | RMSE (late) |")
    print("|---|---|---|---|---|")
    for env in ENVIRONMENTS:
        for make in POLICIES:
            regrets = []
            finals = []
            errors = []
            for seed in SEEDS:
                policy = make()
                result = run(env, policy, seed, ROUNDS)
                regrets.append(normalized_regret(result))
                finals.append(final_window_regret(result))
                e = rmse(result, first=ROUNDS // 2)
                if e is not None:
                    errors.append(e)
            err = spread(errors) if errors else "—"
            print(
                f"| {env.name} | {policy.name} | {spread(regrets)} "
                f"| {spread(finals)} | {err} |"
            )
    print()
    print("### Event-time battery")
    print()
    print(
        f"{EVENT_ENV.name}, {EVENT_DURATION / DAY:g} days of daily traffic "
        f"(peak {EVENT_TRAFFIC.peak_rate:g}/s, trough {EVENT_TRAFFIC.trough_rate:g}/s, "
        f"{EVENT_TRAFFIC.bursts} burst/day), half-life {EVENT_HALF_LIFE / HOUR:g}h, "
        f"seeds {list(SEEDS)}. Regret is normalized (median [IQR]); phase columns are "
        "medians; expired is the fraction of decisions the engine's horizon expired; "
        "open is the ledger high-water mark."
    )
    print()
    print(
        "| delay / horizon | policy | normalized regret "
        "| peak | trough | morning | expired | open |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for label, delay, horizon in EVENT_CONFIGS:
        for make in event_policies(horizon):
            regrets = []
            phases: "dict[str, list[float]]" = {}
            expired = []
            opened = []
            for seed in SEEDS:
                policy = make()
                result = run_events(EVENT_ENV, policy, seed, EVENT_TRAFFIC, delay, EVENT_DURATION)
                regrets.append(normalized_regret(result))
                for p, v in phase_regret(result, phase).items():
                    phases.setdefault(p, []).append(v)
                expired.append(result.expired / len(result.times))
                opened.append(result.max_open)
            cells = " | ".join(
                f"{median_iqr(phases[p])[0]:.3f}" if p in phases else "—"
                for p in ("peak", "trough", "morning")
            )
            print(
                f"| {label} | {policy.name} | {spread(regrets)} | {cells} "
                f"| {median_iqr(expired)[0]:.1%} | {median_iqr(opened)[0]:.0f} |"
            )
    print()
    print(f"_{time.time() - start:.0f}s on {sys.platform}, Python {sys.version.split()[0]}._")


if __name__ == "__main__":
    main()
