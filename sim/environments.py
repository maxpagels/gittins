"""The environment protocol and the stationary battery.

An environment presents each round `t` as a context dict plus candidate
dicts (arm id + action features) — exactly the shape the public encoder
takes — and returns a stochastic reward for the chosen candidate. It also
exposes the oracle expected reward of every candidate (`Round.means`), so
regret is exact, not estimated; the runner keeps the means away from every
policy except the oracle comparator.

Rounds are pure functions of (environment name, seed, t) via sim.rand, so
every policy driven on the same seed sees identical rounds. The reward
noise draw is likewise keyed by (name, seed, t) alone — shared across
candidates (common random numbers), which tightens paired comparisons.

This module holds the two stationary environments (PR 11); the
non-stationary, churn, and missing-feature batteries follow in PR 12.
"""

from dataclasses import dataclass

from sim.rand import gaussian, randint, stream, uniform


@dataclass(frozen=True)
class Round:
    """One decision round as the environment presents it. `means` is the
    oracle truth — the expected reward of each candidate — carried here for
    exact regret; policies never see it (the oracle comparator excepted)."""

    t: float
    context: dict
    arm_ids: tuple[str, ...]
    actions: tuple[dict, ...]
    means: tuple[float, ...]


class Environment:
    """Protocol: subclasses set `name` and `noise` and implement `round`.
    The reward rule is shared: chosen mean plus seeded gaussian noise."""

    name: str
    noise: float

    def round(self, seed: int, t: int) -> Round:
        raise NotImplementedError

    def reward(self, seed: int, t: int, rd: Round, chosen: int) -> float:
        key = stream(self.name, seed, f"reward:{t}")
        return rd.means[chosen] + self.noise * gaussian(key, 0)


class LinearEnvironment(Environment):
    """Stationary, well-specified: the model's home turf.

    Expected reward is  base_a + sum_j w[a][j] * x_j  over numeric context
    features x drawn fresh each round — exactly the dimensions the hashed
    outer-product encoding produces (identity x bias is the per-arm
    intercept, context x identity the per-arm slopes), so the linear model
    is correctly specified up to hash collisions. The floor's cost is
    measured here.

    The hidden parameters are a pure function of (name, seed): bases in
    [-0.5, 0.5), slopes in [-1, 1) scaled by 1/n_features so means stay
    roughly unit-scale at any feature count.
    """

    def __init__(self, k: int, n_features: int = 3, noise: float = 0.1):
        if k < 1 or n_features < 1:
            raise ValueError("need at least one arm and one feature")
        self.name = f"linear-k{k}-f{n_features}"
        self.k = k
        self.n_features = n_features
        self.noise = noise
        self._params: "dict[int, tuple[list[float], list[list[float]]]]" = {}

    def params(self, seed: int) -> "tuple[list[float], list[list[float]]]":
        """(per-arm base, per-arm slope vector) for one seed; memoized."""
        if seed not in self._params:
            key = stream(self.name, seed, "params")
            scale = 1.0 / self.n_features
            bases = []
            weights = []
            c = 0
            for _ in range(self.k):
                bases.append(uniform(key, c, -0.5, 0.5))
                c += 1
                w = []
                for _ in range(self.n_features):
                    w.append(uniform(key, c, -scale, scale))
                    c += 1
                weights.append(w)
            self._params[seed] = (bases, weights)
        return self._params[seed]

    def round(self, seed: int, t: int) -> Round:
        bases, weights = self.params(seed)
        key = stream(self.name, seed, f"context:{t}")
        x = [uniform(key, j, -1.0, 1.0) for j in range(self.n_features)]
        means = tuple(
            bases[a] + sum(weights[a][j] * x[j] for j in range(self.n_features))
            for a in range(self.k)
        )
        return Round(
            t=float(t),
            context={f"f{j}": x[j] for j in range(self.n_features)},
            arm_ids=tuple(f"arm{a}" for a in range(self.k)),
            actions=({},) * self.k,
            means=means,
        )


class XorEnvironment(Environment):
    """Stationary, misspecified: the graceful-degradation check.

    Two binary categorical context features; an arm's expected reward is
    HIGH when the arm's parity matches their XOR, LOW otherwise. The
    encoding carries each feature's interaction with the arm but never
    their product, so conditioned on either feature alone both parities
    look identical — no linear model on these features can separate them.
    The engine should hover near chance between the parities without
    destabilizing (never catastrophically below uniform).
    """

    HIGH = 0.75
    LOW = 0.25

    def __init__(self, k: int, noise: float = 0.1):
        if k < 2:
            raise ValueError("need at least two arms for two parities")
        self.name = f"xor-k{k}"
        self.k = k
        self.noise = noise

    def round(self, seed: int, t: int) -> Round:
        key = stream(self.name, seed, f"context:{t}")
        s1 = randint(key, 0, 2)
        s2 = randint(key, 1, 2)
        parity = s1 ^ s2
        means = tuple(self.HIGH if a % 2 == parity else self.LOW for a in range(self.k))
        return Round(
            t=float(t),
            context={"s1": str(s1), "s2": str(s2)},
            arm_ids=tuple(f"arm{a}" for a in range(self.k)),
            actions=({},) * self.k,
            means=means,
        )
