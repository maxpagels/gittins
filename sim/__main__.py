"""`python -m sim`: run the stationary battery and print a markdown table.

CI appends the output to the GitHub job summary, so every run of the
battery is readable directly in the Actions UI. Runs are seeded and paired
(see sim.rand): the numbers are exactly reproducible for a given harness
version, but they are statistics, not golden vectors — nothing here is
bit-pinned, and this table is a diagnostic, not a gate. The sweep driver
and the full battery report belong to PR 14.
"""

import sys
import time

from sim.environments import LinearEnvironment, XorEnvironment
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
    XorEnvironment(k=4),
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
    print("### Stationary battery")
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
    print(f"_{time.time() - start:.0f}s on {sys.platform}, Python {sys.version.split()[0]}._")


if __name__ == "__main__":
    main()
