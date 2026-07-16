"""The comparator policies, all driven through one identical loop (runner.py).

`gittins` is the engine under test, run on the real public path — encode
(PR 8) -> decide (PR 6) -> ledger learn/expire (PR 7) — never the layers in
isolation, so hashing, the gamma schedule, the floor, and decay are all in
the loop together. The baselines bracket and challenge it:

    oracle          — argmax of the true means: the regret upper bound
    uniform         — uniform random: the regret lower bound; together these
                      normalize regret to [0 (oracle) .. 1 (uniform)]
    greedy          — gittins with gamma -> inf and no floor: same model,
                      same encoding, pure argmax of the estimates
    epsilon-greedy  — same model again, exploration replaced by an
                      epsilon coin: the "would something dumber beat us"
                      check (PROGRESS.md suggests eps in {0.05, 0.1})

A policy is reset per run with `begin(seed)`; `choose` returns the chosen
candidate index plus its per-candidate reward estimates (None for the
model-free policies), which the runner uses for the prediction-RMSE metric;
`observe` hands back the reward. All policy randomness comes from sim.rand
streams keyed by (policy name, seed, t), so runs replay exactly.
"""

from gittins_reference.decide import decide, new_bandit
from gittins_reference.encoding import encode
from gittins_reference.ledger import expire, learn
from gittins_reference.model import factorize, new_model, predict_factored, update
from sim.environments import Round
from sim.rand import randint, stream, uniform


class Policy:
    name: str

    def begin(self, seed: int) -> None:
        """Reset all run state; called once before each run."""

    def choose(self, rd: Round, seed: int) -> "tuple[int, list[float] | None]":
        """(chosen candidate index, per-candidate reward estimates or None)."""
        raise NotImplementedError

    def observe(self, reward: float) -> None:
        """The reward for the most recent `choose`."""


def argmax(values: "list[float] | tuple[float, ...]") -> int:
    """First maximum — the same tie-breaking rule as exploration.py."""
    best = 0
    for i in range(1, len(values)):
        if values[i] > values[best]:
            best = i
    return best


class OraclePolicy(Policy):
    """Upper bound; the only policy allowed to read Round.means."""

    name = "oracle"

    def choose(self, rd: Round, seed: int) -> "tuple[int, list[float] | None]":
        return argmax(rd.means), None


class UniformPolicy(Policy):
    """Lower bound: uniform over the candidates, no model."""

    name = "uniform"

    def choose(self, rd: Round, seed: int) -> "tuple[int, list[float] | None]":
        key = stream(self.name, seed, f"choose:{rd.t}")
        return randint(key, 0, len(rd.arm_ids)), None


class GittinsPolicy(Policy):
    """The engine, on the real public path. Rewards are handed to the
    ledger's `learn` by decision id; the `expire` sweep runs with every
    round's time, as a real event loop would (nothing expires here — sim
    rewards are immediate — but the sweep stays in the loop)."""

    def __init__(self, bits: int = 8, half_life: float = 1000.0, horizon: float = 10.0):
        self.name = "gittins"
        self.bits = bits
        self.half_life = half_life
        self.horizon = horizon

    def begin(self, seed: int) -> None:
        self.state = new_bandit(2**self.bits, self.half_life, t=0.0, horizon=self.horizon)
        self.salt = f"{self.name}:{seed}"
        self._open = ""

    def choose(self, rd: Round, seed: int) -> "tuple[int, list[float] | None]":
        _, self.state = expire(self.state, rd.t)
        candidates = [
            encode(rd.context, rd.arm_ids[i], rd.actions[i], self.bits)
            for i in range(len(rd.arm_ids))
        ]
        record, self.state = decide(self.state, candidates, rd.t, self.salt)
        self._open = record.decision_id
        # Metric-only read: the same estimates decide just scored with
        # (the model can't have changed between), recomputed because decide
        # deliberately logs only the chosen candidate.
        f = factorize(self.state.model, rd.t)
        estimates = [predict_factored(f, x)[0] for x in candidates]
        return record.chosen, estimates

    def observe(self, reward: float) -> None:
        _, self.state = learn(self.state, self._open, reward)


class ModelPolicy(Policy):
    """Shared machinery for the model-based baselines: the same
    per-coordinate ridge on the same hashed encoding as the engine, trained
    on every reward immediately — only the exploration rule differs."""

    def __init__(self, bits: int = 8, half_life: float = 1000.0):
        self.bits = bits
        self.half_life = half_life

    def begin(self, seed: int) -> None:
        self.model = new_model(2**self.bits, self.half_life, t=0.0)
        self._x: list[float] = []
        self._t = 0.0

    def estimates(self, rd: Round) -> "tuple[list[list[float]], list[float]]":
        candidates = [
            encode(rd.context, rd.arm_ids[i], rd.actions[i], self.bits)
            for i in range(len(rd.arm_ids))
        ]
        f = factorize(self.model, rd.t)
        return candidates, [predict_factored(f, x)[0] for x in candidates]

    def picked(self, rd: Round, candidates: "list[list[float]]", chosen: int) -> None:
        self._x = candidates[chosen]
        self._t = rd.t

    def observe(self, reward: float) -> None:
        self.model = update(self.model, self._x, reward, self._t)


class GreedyPolicy(ModelPolicy):
    """gamma -> inf, no floor: always the argmax estimate."""

    def __init__(self, bits: int = 8, half_life: float = 1000.0):
        super().__init__(bits, half_life)
        self.name = "greedy"

    def choose(self, rd: Round, seed: int) -> "tuple[int, list[float] | None]":
        candidates, estimates = self.estimates(rd)
        chosen = argmax(estimates)
        self.picked(rd, candidates, chosen)
        return chosen, estimates


class EpsilonGreedyPolicy(ModelPolicy):
    """Argmax estimate, except a uniform candidate with probability eps."""

    def __init__(self, eps: float, bits: int = 8, half_life: float = 1000.0):
        super().__init__(bits, half_life)
        if not (0.0 <= eps <= 1.0):
            raise ValueError("eps must be in [0, 1]")
        self.eps = eps
        self.name = f"epsilon-{eps}"

    def choose(self, rd: Round, seed: int) -> "tuple[int, list[float] | None]":
        candidates, estimates = self.estimates(rd)
        key = stream(self.name, seed, f"choose:{rd.t}")
        if uniform(key, 0) < self.eps:
            chosen = randint(key, 1, len(candidates))
        else:
            chosen = argmax(estimates)
        self.picked(rd, candidates, chosen)
        return chosen, estimates
