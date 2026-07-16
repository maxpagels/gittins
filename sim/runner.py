"""The runner: one policy through one environment for one seed.

Round by round: fetch the environment's round, let the policy choose, draw
the stochastic reward, hand it back. For the engine this loop *is* the real
public path end to end. Because rounds are pure functions of
(environment, seed, t), every policy run on the same seed faces identical
contexts, candidates, and noise draws — comparisons are paired.

The result keeps per-round series, computed from the oracle means so regret
is exact:

    regret[t]      — best mean minus the chosen candidate's mean
    normalizer[t]  — best mean minus the average mean: the expected
                     per-round regret of uniform random, the unit that
                     metrics.normalized_regret divides by
    sq_error[t]    — mean squared prediction error across the round's
                     candidates (None for model-free policies), separating
                     model quality from policy quality
    reward[t]      — the realized (noisy) reward the policy trained on
"""

from dataclasses import dataclass

from sim.environments import Environment
from sim.policies import Policy


@dataclass(frozen=True)
class RunResult:
    environment: str
    policy: str
    seed: int
    regret: tuple[float, ...]
    normalizer: tuple[float, ...]
    sq_error: "tuple[float, ...] | None"
    reward: tuple[float, ...]


def run(env: Environment, policy: Policy, seed: int, rounds: int) -> RunResult:
    if rounds < 1:
        raise ValueError("need at least one round")
    policy.begin(seed)
    regret = []
    normalizer = []
    sq_error: "list[float] | None" = None
    reward = []
    for t in range(rounds):
        rd = env.round(seed, t)
        chosen, estimates = policy.choose(rd, seed)
        r = env.reward(seed, t, rd, chosen)
        policy.observe(r)

        k = len(rd.means)
        best = max(rd.means)
        regret.append(best - rd.means[chosen])
        normalizer.append(best - sum(rd.means) / k)
        reward.append(r)
        if estimates is not None:
            if sq_error is None:
                sq_error = []
            total = 0.0
            for i in range(k):
                total += (estimates[i] - rd.means[i]) ** 2
            sq_error.append(total / k)
    return RunResult(
        environment=env.name,
        policy=policy.name,
        seed=seed,
        regret=tuple(regret),
        normalizer=tuple(normalizer),
        sq_error=None if sq_error is None else tuple(sq_error),
        reward=tuple(reward),
    )
