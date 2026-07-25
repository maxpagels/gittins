"""The comparator policies, all driven through one identical loop (runner.py).

`gittins` is the engine under test, run on the real public path — encode
-> decide -> ledger learn/expire — never the layers in
isolation, so hashing, the epsilon-greedy rule, the tie split, and
forgetting are all in the loop together. The baselines bracket and
challenge it:

    oracle          — argmax of the true means: the regret upper bound
    uniform         — uniform random: the regret lower bound; together these
                      normalize regret to [0 (oracle) .. 1 (uniform)]
    greedy          — gittins with epsilon = 0 and first-max ties: same
                      model, same encoding, pure argmax of the estimates
    epsilon-greedy  — same model again, exploration as an epsilon coin
                      outside the engine: no ledger, first-max ties — what
                      the engine's plumbing is priced against

A policy is reset per run with `begin(seed)`; `choose` returns the chosen
candidate index plus its per-candidate reward estimates (None for the
model-free policies), which the runner uses for the prediction-RMSE metric;
`observe` hands back the reward. All policy randomness comes from sim.rand
streams keyed by (policy name, seed, t), so runs replay exactly.
"""

from gittins_reference.decide import decide, new_bandit
from gittins_reference.encoding import encode
from gittins_reference.exploration import DEFAULT_EPSILON
from gittins_reference.ledger import expire, learn
from gittins_reference.model import (
    DEFAULT_FORGETTING,
    factorize,
    new_model,
    predict_factored,
    update,
)
from sim.environments import Round
from sim.rand import randint, stream, uniform


class Policy:
    """One decision per round via `choose`, its reward via `observe`."""

    name: str

    def begin(self, seed: int) -> None:
        """Reset all run state; called once before each run."""

    def choose(self, rd: Round, seed: int) -> "tuple[int, list[float] | None]":
        """(chosen candidate index, per-candidate reward estimates or None)."""
        raise NotImplementedError

    def observe(self, reward: float) -> None:
        """The reward for the most recent `choose`; no-op for model-free
        policies."""


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
    """The engine, on the real public path: rewards are handed to the
    ledger's `learn` by decision id, and the `expire` sweep runs with every
    round's time, as a real event loop would. Rewards are immediate in the
    round runner, so nothing actually expires — the sweep is kept because
    it is part of the path a production loop drives."""

    def __init__(
        self,
        bits: int = 8,
        forgetting: float = DEFAULT_FORGETTING,
        horizon: float = 10.0,
        epsilon: float = DEFAULT_EPSILON,
    ):
        self.name = "gittins"
        self.bits = bits
        self.forgetting = forgetting
        self.horizon = horizon
        self.epsilon = epsilon

    def begin(self, seed: int) -> None:
        self.state = new_bandit(
            2**self.bits,
            horizon=self.horizon,
            epsilon=self.epsilon,
            forgetting=self.forgetting,
        )
        self.salt = f"{self.name}:{seed}"

    def choose(self, rd: Round, seed: int) -> "tuple[int, list[float] | None]":
        _, self.state = expire(self.state, rd.t)
        candidates = [
            encode(rd.context, rd.arm_ids[i], rd.actions[i], self.bits)
            for i in range(len(rd.arm_ids))
        ]
        record, self.state = decide(self.state, candidates, rd.t, self.salt)
        self._decision_id = record.decision_id
        self._t = rd.t
        # Metric-only read: the same estimates decide just scored with
        # (the model can't have changed between), recomputed because decide
        # deliberately logs only the chosen candidate.
        f = factorize(self.state.model)
        estimates = [predict_factored(f, x)[0] for x in candidates]
        return record.chosen, estimates

    def observe(self, reward: float) -> None:
        _, self.state = learn(self.state, self._decision_id, reward, self._t)


class ModelPolicy(Policy):
    """Shared machinery for the model-based baselines: the same
    per-coordinate forgetting ridge on the same hashed encoding as the
    engine, trained on every reward exactly as the engine trains — only the
    exploration rule differs, and there is no ledger anywhere."""

    def __init__(self, bits: int = 8, forgetting: float = DEFAULT_FORGETTING):
        self.bits = bits
        self.forgetting = forgetting

    def begin(self, seed: int) -> None:
        self.model = new_model(2**self.bits, self.forgetting)

    def estimates(self, rd: Round) -> "tuple[list, list[float]]":
        """(sparse candidates, their estimates) for one round."""
        candidates = [
            encode(rd.context, rd.arm_ids[i], rd.actions[i], self.bits)
            for i in range(len(rd.arm_ids))
        ]
        f = factorize(self.model)
        return candidates, [predict_factored(f, x)[0] for x in candidates]

    def observe(self, reward: float) -> None:
        self.model = update(self.model, self._chosen_x, reward)


class GreedyPolicy(ModelPolicy):
    """epsilon = 0, first-max ties: always the argmax estimate."""

    def __init__(self, bits: int = 8, forgetting: float = DEFAULT_FORGETTING):
        super().__init__(bits, forgetting)
        self.name = "greedy"

    def choose(self, rd: Round, seed: int) -> "tuple[int, list[float] | None]":
        candidates, estimates = self.estimates(rd)
        chosen = argmax(estimates)
        self._chosen_x = candidates[chosen]
        return chosen, estimates


class EpsilonGreedyPolicy(ModelPolicy):
    """Argmax estimate, except a uniform candidate with probability eps."""

    def __init__(self, eps: float, bits: int = 8, forgetting: float = DEFAULT_FORGETTING):
        super().__init__(bits, forgetting)
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
        self._chosen_x = candidates[chosen]
        return chosen, estimates
