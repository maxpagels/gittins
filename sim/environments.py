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

This module holds the stationary environments and the non-stationary,
churn, and missing-feature ones; traffic and reward delays (the event-time
battery) live in sim/traffic.py.
"""

import math
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

    def round(self, seed: int, t: float) -> Round:
        raise NotImplementedError

    def reward(self, seed: int, t: float, rd: Round, chosen: int) -> float:
        key = stream(self.name, seed, f"reward:{t}")
        return rd.means[chosen] + self.noise * gaussian(key, 0)


def linear_params(
    name: str, seed: int, label: str, k: int, n_features: int
) -> "tuple[list[float], list[list[float]]]":
    """One hidden linear world — (per-arm base, per-arm slope vector) — as a
    pure function of (name, seed, label): bases in [-0.5, 0.5), slopes in
    [-1, 1) scaled by 1/n_features so means stay roughly unit-scale at any
    feature count. The label lets one environment hold several worlds
    (epochs, rotation endpoints)."""
    key = stream(name, seed, label)
    scale = 1.0 / n_features
    bases = []
    weights = []
    c = 0
    for _ in range(k):
        bases.append(uniform(key, c, -0.5, 0.5))
        c += 1
        w = []
        for _ in range(n_features):
            w.append(uniform(key, c, -scale, scale))
            c += 1
        weights.append(w)
    return bases, weights


def linear_means(
    bases: list[float], weights: "list[list[float]]", x: list[float]
) -> tuple[float, ...]:
    """base_a + w[a] . x for every arm."""
    return tuple(
        bases[a] + sum(weights[a][j] * x[j] for j in range(len(x)))
        for a in range(len(bases))
    )


class LinearEnvironment(Environment):
    """Stationary, well-specified: the model's home turf.

    Expected reward is  base_a + sum_j w[a][j] * x_j  over numeric context
    features x drawn fresh each round — exactly the dimensions the hashed
    outer-product encoding produces (identity x bias is the per-arm
    intercept, context x identity the per-arm slopes), so the linear model
    is correctly specified up to hash collisions. The floor's cost is
    measured here.
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
            self._params[seed] = linear_params(self.name, seed, "params", self.k, self.n_features)
        return self._params[seed]

    def round(self, seed: int, t: float) -> Round:
        bases, weights = self.params(seed)
        key = stream(self.name, seed, f"context:{t}")
        x = [uniform(key, j, -1.0, 1.0) for j in range(self.n_features)]
        return Round(
            t=float(t),
            context={f"f{j}": x[j] for j in range(self.n_features)},
            arm_ids=tuple(f"arm{a}" for a in range(self.k)),
            actions=({},) * self.k,
            means=linear_means(bases, weights, x),
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

    def round(self, seed: int, t: float) -> Round:
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


class NeedleEnvironment(Environment):
    """Stationary, context-free: the pure exploration stress test.

    Every arm pays BASE except one hidden needle arm — a pure function of
    (name, seed) — which pays BASE + gap. There is no context and no action
    feature to generalize from, so finding the needle is a matter of
    sampling arms; with many arms and a small gap this isolates the
    exploration rule from the model. Greedy can lock onto a lucky early
    draw and never leave it; a sound explorer's regret must keep shrinking.
    """

    BASE = 0.5

    def __init__(self, k: int, gap: float = 0.2, noise: float = 0.1):
        if k < 2:
            raise ValueError("need at least two arms to hide a needle")
        if not (gap > 0.0):
            raise ValueError("gap must be positive")
        self.name = f"needle-k{k}-g{gap:g}"
        self.k = k
        self.gap = gap
        self.noise = noise

    def needle(self, seed: int) -> int:
        """The needle arm's index for one seed."""
        return randint(stream(self.name, seed, "needle"), 0, self.k)

    def round(self, seed: int, t: float) -> Round:
        needle = self.needle(seed)
        means = tuple(
            self.BASE + self.gap if a == needle else self.BASE for a in range(self.k)
        )
        return Round(
            t=float(t),
            context={},
            arm_ids=tuple(f"arm{a}" for a in range(self.k)),
            actions=({},) * self.k,
            means=means,
        )


class ActionFeatureEnvironment(Environment):
    """Stationary, well-specified through action features: the
    generalization-across-arms check.

    Each arm carries a numeric action-feature vector z — drawn once per
    (name, seed) and presented to the policy in its action dict — and the
    expected reward is the bilinear form x . W z over the round's fresh
    context vector x, with W hidden. Every term is a context x action
    product, exactly the interaction dimensions the hashed outer-product
    encoding produces, so the model is correctly specified — but only
    through the action features: with many arms and few features, evidence
    must transfer between arms that share structure, not accrue per
    identity (whose true weight is zero). The first battery environment to
    exercise the encoder's action namespace at all.
    """

    def __init__(self, k: int, n_features: int = 3, noise: float = 0.1):
        if k < 1 or n_features < 1:
            raise ValueError("need at least one arm and one feature")
        self.name = f"actions-k{k}-f{n_features}"
        self.k = k
        self.n_features = n_features
        self.noise = noise
        self._params: "dict[int, tuple[list[list[float]], list[list[float]]]]" = {}

    def params(self, seed: int) -> "tuple[list[list[float]], list[list[float]]]":
        """(per-arm action vectors, hidden weight matrix) for one seed;
        memoized. Weights are scaled 1/n_features so means stay roughly
        unit-scale at any feature count, matching LinearEnvironment."""
        if seed not in self._params:
            key = stream(self.name, seed, "params")
            n = self.n_features
            scale = 1.0 / n
            c = 0
            arms = []
            for _ in range(self.k):
                arms.append([uniform(key, c + j, -1.0, 1.0) for j in range(n)])
                c += n
            weights = []
            for _ in range(n):
                weights.append([uniform(key, c + j, -scale, scale) for j in range(n)])
                c += n
            self._params[seed] = (arms, weights)
        return self._params[seed]

    def round(self, seed: int, t: float) -> Round:
        arms, weights = self.params(seed)
        n = self.n_features
        key = stream(self.name, seed, f"context:{t}")
        x = [uniform(key, j, -1.0, 1.0) for j in range(n)]
        means = tuple(
            sum(x[i] * weights[i][j] * arms[a][j] for i in range(n) for j in range(n))
            for a in range(self.k)
        )
        return Round(
            t=float(t),
            context={f"f{j}": x[j] for j in range(n)},
            arm_ids=tuple(f"arm{a}" for a in range(self.k)),
            actions=tuple({f"z{j}": arms[a][j] for j in range(n)} for a in range(self.k)),
            means=means,
        )


class AbruptShiftEnvironment(Environment):
    """Non-stationary, well-specified within each epoch: the recovery check.

    A LinearEnvironment world that is redrawn from scratch every `period`
    rounds — epoch `t // period` has its own hidden bases and slopes, all
    pure functions of (name, seed, epoch). At each boundary everything the
    model believes is stale at once. Run with `period` both long and short
    relative to the model's effective window (~1/(1 - forgetting) updates):
    this is where the engine's recovery claim — stale evidence is
    outweighed within about one window — is measured, as post-shift
    recovery.
    """

    def __init__(self, k: int, period: int, n_features: int = 3, noise: float = 0.1):
        if k < 1 or n_features < 1:
            raise ValueError("need at least one arm and one feature")
        if period < 1:
            raise ValueError("period must be at least one round")
        self.name = f"shift-k{k}-p{period}"
        self.k = k
        self.period = period
        self.n_features = n_features
        self.noise = noise
        self._params: "dict[tuple[int, int], tuple[list[float], list[list[float]]]]" = {}

    def params(self, seed: int, epoch: int) -> "tuple[list[float], list[list[float]]]":
        if (seed, epoch) not in self._params:
            self._params[(seed, epoch)] = linear_params(
                self.name, seed, f"params:{epoch}", self.k, self.n_features
            )
        return self._params[(seed, epoch)]

    def round(self, seed: int, t: float) -> Round:
        bases, weights = self.params(seed, int(t // self.period))
        key = stream(self.name, seed, f"context:{t}")
        x = [uniform(key, j, -1.0, 1.0) for j in range(self.n_features)]
        return Round(
            t=float(t),
            context={f"f{j}": x[j] for j in range(self.n_features)},
            arm_ids=tuple(f"arm{a}" for a in range(self.k)),
            actions=({},) * self.k,
            means=linear_means(bases, weights, x),
        )


class DriftEnvironment(Environment):
    """Non-stationary, smooth: the regime where any fixed forgetting rate
    is a compromise.

    The hidden parameters rotate continuously between two independently
    drawn linear worlds:  p(t) = cos(theta) * p0 + sin(theta) * p1  with
    theta = 2*pi*t / period. A `period` much longer than the run is slow
    drift (the world creeps away from everything learned); a period inside
    the run is seasonal (old evidence becomes right again — a model that
    has forgotten it pays a re-learning tax every cycle).
    """

    def __init__(self, k: int, period: int, n_features: int = 3, noise: float = 0.1):
        if k < 1 or n_features < 1:
            raise ValueError("need at least one arm and one feature")
        if period < 1:
            raise ValueError("period must be at least one round")
        self.name = f"drift-k{k}-p{period}"
        self.k = k
        self.period = period
        self.n_features = n_features
        self.noise = noise
        self._params: "dict[int, tuple]" = {}

    def params(self, seed: int) -> tuple:
        """The two rotation endpoints for one seed; memoized."""
        if seed not in self._params:
            self._params[seed] = (
                linear_params(self.name, seed, "params:0", self.k, self.n_features),
                linear_params(self.name, seed, "params:1", self.k, self.n_features),
            )
        return self._params[seed]

    def round(self, seed: int, t: float) -> Round:
        (b0, w0), (b1, w1) = self.params(seed)
        theta = 2.0 * math.pi * t / self.period
        c, s = math.cos(theta), math.sin(theta)
        bases = [c * b0[a] + s * b1[a] for a in range(self.k)]
        weights = [
            [c * w0[a][j] + s * w1[a][j] for j in range(self.n_features)]
            for a in range(self.k)
        ]
        key = stream(self.name, seed, f"context:{t}")
        x = [uniform(key, j, -1.0, 1.0) for j in range(self.n_features)]
        return Round(
            t=float(t),
            context={f"f{j}": x[j] for j in range(self.n_features)},
            arm_ids=tuple(f"arm{a}" for a in range(self.k)),
            actions=({},) * self.k,
            means=linear_means(bases, weights, x),
        )


class ChurnEnvironment(Environment):
    """Arm churn: arms are born and die mid-run; the candidate list changes
    shape under the policy.

    Context-free arms with hidden means in [0.25, 0.75) (positive, so the
    reward-rate recovery metric is meaningful). Three scripted events, in
    absolute rounds so the schedule is independent of run length:

    - the best arm disappears at `absent[0]` (the policy must fall back to
      the runner-up),
    - it returns at `absent[1]` (rediscovery: its faded evidence must win
      back traffic via whatever exploration remains),
    - a brand-new arm strictly better than everything (best + newcomer_gap)
      is born at `newcomer_at` (cold-start regret of a strong stranger).

    The encoding needs no registration for any of this — a returning or
    newborn arm is usable the moment its id first appears.
    """

    def __init__(
        self,
        k: int,
        absent: "tuple[int, int]" = (400, 800),
        newcomer_at: int = 1100,
        newcomer_gap: float = 0.1,
        noise: float = 0.1,
    ):
        if k < 2:
            raise ValueError("need at least two arms to survive a disappearance")
        if not (0 <= absent[0] < absent[1]):
            raise ValueError("absence window must be a nonempty range")
        self.name = f"churn-k{k}"
        self.k = k
        self.absent = absent
        self.newcomer_at = newcomer_at
        self.newcomer_gap = newcomer_gap
        self.noise = noise
        self._means: "dict[int, list[float]]" = {}

    def arm_means(self, seed: int) -> list[float]:
        """The k resident arms' hidden means for one seed; memoized."""
        if seed not in self._means:
            key = stream(self.name, seed, "means")
            self._means[seed] = [uniform(key, a, 0.25, 0.75) for a in range(self.k)]
        return self._means[seed]

    def round(self, seed: int, t: float) -> Round:
        means = self.arm_means(seed)
        best = max(range(self.k), key=means.__getitem__)
        alive = [
            a
            for a in range(self.k)
            if not (a == best and self.absent[0] <= t < self.absent[1])
        ]
        ids = [f"arm{a}" for a in alive]
        mu = [means[a] for a in alive]
        if t >= self.newcomer_at:
            ids.append("newcomer")
            mu.append(means[best] + self.newcomer_gap)
        return Round(
            t=float(t),
            context={},
            arm_ids=tuple(ids),
            actions=({},) * len(ids),
            means=tuple(mu),
        )


class DropoutEnvironment(Environment):
    """Missing features: the world is linear in x, but the policy sees a
    damaged view of it.

    Each true feature is absent from the presented context with probability
    `p_drop`, independently per round and per feature (the hashed encoding
    treats absent as no token — occasionally the whole context is empty),
    and `n_distractors` irrelevant features with fresh random values are
    always present. Oracle means come from the full x — the environment
    knows the world, the policy doesn't — so regret prices what the missing
    information costs; the check is that partial contexts and distractors
    degrade the model gracefully, never destabilize it.
    """

    def __init__(
        self,
        k: int,
        n_features: int = 3,
        p_drop: float = 0.3,
        n_distractors: int = 2,
        noise: float = 0.1,
    ):
        if k < 1 or n_features < 1:
            raise ValueError("need at least one arm and one feature")
        if not (0.0 <= p_drop < 1.0):
            raise ValueError("p_drop must be in [0, 1)")
        self.name = f"dropout-k{k}-f{n_features}-p{p_drop:g}"
        self.k = k
        self.n_features = n_features
        self.p_drop = p_drop
        self.n_distractors = n_distractors
        self.noise = noise
        self._params: "dict[int, tuple[list[float], list[list[float]]]]" = {}

    def params(self, seed: int) -> "tuple[list[float], list[list[float]]]":
        if seed not in self._params:
            self._params[seed] = linear_params(self.name, seed, "params", self.k, self.n_features)
        return self._params[seed]

    def round(self, seed: int, t: float) -> Round:
        bases, weights = self.params(seed)
        key = stream(self.name, seed, f"context:{t}")
        x = [uniform(key, j, -1.0, 1.0) for j in range(self.n_features)]
        # Draws beyond the features: dropout coins, then distractor values.
        context = {
            f"f{j}": x[j]
            for j in range(self.n_features)
            if uniform(key, self.n_features + j) >= self.p_drop
        }
        for d in range(self.n_distractors):
            context[f"d{d}"] = uniform(key, 2 * self.n_features + d, -1.0, 1.0)
        return Round(
            t=float(t),
            context=context,
            arm_ids=tuple(f"arm{a}" for a in range(self.k)),
            actions=({},) * self.k,
            means=linear_means(bases, weights, x),
        )
