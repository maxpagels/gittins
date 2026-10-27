"""An opinionated, highly optimised contextual bandit engine.

The state is an opaque handle updated in place: calls return only their
result, and every alias of the handle observes the current state. Persist
it with `serialize`.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Literal, TypeAlias, final

# Mapping of object: mypy infers {"a": "x", "b": 1} as dict[str, object], and
# decide already rejects bad values with a ValueError naming the feature.
_Features: TypeAlias = Mapping[str, object]
_Candidate: TypeAlias = tuple[str, _Features]
"""An `(arm_id, action_features)` pair."""

@final
class BanditState:
    """The bandit's state: an opaque handle, updated in place by decide/learn/
    expire; persist it with serialize/deserialize."""

@final
class DecisionRecord:
    """One decision, self-contained: what was chosen, with what probability,
    under which id — everything outcome reporting and offline evaluation
    need. Records returned by `decide` additionally carry the inputs the
    decision was made over (`bits`, `context`, `candidates` — the caller's
    own objects), so they log as-is via `log_line`; records that crossed the
    state boundary (a `train` callback's) carry None there."""

    @property
    def decision_id(self) -> str:
        """`"{salt}:{seq}"` — pass it to `learn` to report the outcome."""
    @property
    def t(self) -> float:
        """The time the decision was made at."""
    @property
    def candidate_hash(self) -> int: ...
    @property
    def chosen(self) -> int:
        """Index of the chosen candidate in the list passed to `decide`."""
    @property
    def features(self) -> list[tuple[int, float]]:
        """The chosen candidate's encoded sparse (index, value) pairs."""
    @property
    def propensity(self) -> float:
        """The probability the chosen candidate had of being chosen."""
    @property
    def model_version(self) -> int:
        """Observations the model had absorbed when the decision was made."""
    @property
    def salt(self) -> str: ...
    @property
    def bits(self) -> int | None: ...
    @property
    def context(self) -> _Features | None: ...
    @property
    def candidates(self) -> Sequence[_Candidate] | None: ...

@final
class Resolution:
    """One resolution event: how a decision left the ledger. `reward` is what
    the model trained with."""

    @property
    def decision_id(self) -> str: ...
    @property
    def kind(self) -> Literal["rewarded", "expired"]: ...
    @property
    def reward(self) -> float: ...

def create(
    bits: int,
    horizon: float,
    default_reward: float = 0.0,
    epsilon: float = 0.05,
    forgetfulness: float = 0.999,
) -> BanditState:
    """A fresh bandit.

    `bits` (1–24) sizes the model: features hash into 2**bits weights.
    `horizon` is how long, in the units of `t`, a decision waits for its
    reward before `expire` resolves it with `default_reward`. `epsilon` is
    the probability mass spent exploring uniformly; `forgetfulness` is the
    per-update decay of past observations (closer to 1 remembers longer).
    """

def decide(
    state: BanditState,
    context: _Features,
    candidates: Sequence[_Candidate],
    t: float,
    salt: str,
    score: Callable[[_Features, Sequence[_Candidate]], Sequence[float]] | None = None,
    explore: Callable[[list[float], float], Sequence[float]] | None = None,
) -> DecisionRecord:
    """Choose one of `candidates` for `context` at time `t`.

    `candidates` is a list of `(arm_id, action_features)` pairs and may
    change on every call. `salt` namespaces decision ids (`"{salt}:{seq}"`);
    use a distinct one per state writing to the same log. The decision
    joins the state's open ledger until `learn` or `expire` resolves it;
    `candidates[record.chosen]` is the choice. Feature values: strings are
    categorical, ints/floats/bools numeric, None absent.

    `score(context, candidates)` replaces the built-in estimates and
    `explore(estimates, epsilon)` the epsilon-greedy distribution; each
    returns one number per candidate.
    """

def learn(
    state: BanditState,
    decision_id: str,
    reward: float,
    t: float,
    train: Callable[[DecisionRecord, float], object] | None = None,
) -> Resolution | None:
    """The resolution, or None if the id is unknown or already resolved.

    The engine classifies against the horizon at time `t`: a reward arriving
    at or past the record's `t + horizon` resolves as expired with the
    default reward. `train` is the BYO training tap: it replaces the
    built-in update and fires after the resolution commits, exactly once,
    with the classified reward.
    """

def expire(
    state: BanditState,
    t: float,
    train: Callable[[DecisionRecord, float], object] | None = None,
) -> tuple[Resolution, ...]:
    """Every decision past its horizon at time `t`, resolved as expired, in
    ledger order. `train` as on `learn`, fired per due record."""

def serialize(state: BanditState) -> str:
    """The current state as one plain hex string — storable anywhere text
    goes, with no byte handling on the caller's side."""

def deserialize(data: str) -> BanditState:
    """The state a `serialize` string encodes."""

def model_bits(state: BanditState) -> int:
    """The `bits` the state was created with."""

def log_line(item: DecisionRecord | Resolution) -> str:
    """One canonical experience-log line for a decision record or
    resolution — append it to your log verbatim; it is exactly what the
    `gittins` CLI's verify/eval/replay consume. Decision records must be the
    ones `decide` returned (they carry the inputs)."""
