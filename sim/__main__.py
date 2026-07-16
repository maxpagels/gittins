"""`python -m sim`: run the environment battery and print a markdown table.

Every world x every comparator, round-based. CI appends the output to the
GitHub job summary, so every run is readable directly in the Actions UI.
Runs are seeded and paired (see sim.rand): the numbers are exactly
reproducible for a given harness version, but they are statistics, not
golden vectors — nothing here is bit-pinned, and the table is a
diagnostic, not a gate. The sweep driver that settles the engine's
constants is sim/sweep.py.
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
from sim.metrics import final_window_regret, median_iqr, normalized_regret, rmse
from sim.policies import (
    EpsilonGreedyPolicy,
    GittinsPolicy,
    GreedyPolicy,
    OraclePolicy,
    UniformPolicy,
)
from sim.runner import run

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
    # Non-stationary, churn, and missing-feature worlds. The model's
    # effective window is ~1000 updates: shift period 375 is well inside it
    # (four epochs per 1500-round run), drift period 500 is seasonal (three
    # full cycles), churn's events land at rounds 400/800/1100.
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

def spread(values: "list[float]") -> str:
    median, q1, q3 = median_iqr(values)
    return f"{median:.3f} [{q1:.3f}, {q3:.3f}]"


def main() -> None:
    start = time.time()
    decisions = 0
    print("### Environment battery (Python Reference implementation)")
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
                decisions += len(result.regret)
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
    elapsed = time.time() - start
    print(
        f"_{elapsed:.0f}s on {sys.platform}, Python {sys.version.split()[0]}; "
        f"{decisions:,} decisions total, {decisions / elapsed:,.0f} decisions/s._"
    )


if __name__ == "__main__":
    main()
