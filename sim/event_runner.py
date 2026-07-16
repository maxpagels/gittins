"""The event-time runner: one policy through one environment, driven by a
time-ordered event stream instead of rounds (PR 13).

Decision arrivals (from a traffic model) and reward arrivals (each
decision's reward lands `delay` later) interleave in one queue. At every
event the policy's expiry sweep runs first with that event's time — exactly
a production event loop at simulated speed — so learning happens mid-
traffic from whatever state exists at that instant, some rewards land many
decisions late, and rewards slower than the engine's horizon arrive after
their decision has already expired (a structural no-op: the ledger has
spent the record).

Arrival times, contexts, rewards, and delays are all pure functions of
(name, seed) — never of the policy — so event-time comparisons are paired
exactly like round-based ones. The reward value is drawn at decision time
(the world decides the outcome then); the delay only sets when the policy
gets to see it.

The result carries the same per-decision series as the round runner's
RunResult — regret/normalizer/best/sq_error/reward, indexed by decision in
arrival order — so every round metric (normalized regret, final window,
RMSE, recovery time) applies unchanged, plus the decision timestamps (for
phase splits) and the reward-plumbing accounting:

    resolved    — reward deliveries within the run (a delivery for an
                  already-expired decision is a no-op but still counts:
                  resolved + in_flight == decisions, always)
    expired     — decisions the policy's own horizon expired
    in_flight   — rewards still undelivered when the run ended
    max_open    — the ledger-occupancy high-water mark (0 for unledgered
                  policies): does the horizon bound hold under peak load?
"""

import heapq
from dataclasses import dataclass

from sim.environments import Environment
from sim.policies import Policy

DECISION = 0  # heap tiebreak: at equal times, decide before resolving
REWARD = 1


@dataclass(frozen=True)
class EventRunResult:
    environment: str
    policy: str
    seed: int
    times: tuple[float, ...]
    regret: tuple[float, ...]
    normalizer: tuple[float, ...]
    best: tuple[float, ...]
    sq_error: "tuple[float, ...] | None"
    reward: tuple[float, ...]
    resolved: int
    expired: int
    in_flight: int
    max_open: int


def run_events(
    env: Environment, policy: Policy, seed: int, traffic, delay, duration: float
) -> EventRunResult:
    if not (duration > 0.0):
        raise ValueError("duration must be positive")
    policy.begin(seed)
    queue = [
        (t, DECISION, i, None) for i, t in enumerate(traffic.arrivals(seed, duration))
    ]
    heapq.heapify(queue)

    times = []
    regret = []
    normalizer = []
    best_means = []
    sq_error: "list[float] | None" = None
    reward = []
    resolved = 0
    in_flight = 0
    max_open = 0

    while queue:
        t, kind, i, payload = heapq.heappop(queue)
        policy.sweep(t)
        if kind == REWARD:
            ref, r = payload
            policy.resolve(ref, r)
            resolved += 1
        else:
            rd = env.round(seed, t)
            chosen, estimates, ref = policy.decide_at(rd, seed)
            r = env.reward(seed, t, rd, chosen)
            lands = t + delay.draw(seed, i, t)
            if lands < duration:
                heapq.heappush(queue, (lands, REWARD, i, (ref, r)))
            else:
                in_flight += 1

            k = len(rd.means)
            best = max(rd.means)
            times.append(t)
            regret.append(best - rd.means[chosen])
            normalizer.append(best - sum(rd.means) / k)
            best_means.append(best)
            reward.append(r)
            if estimates is not None:
                if sq_error is None:
                    sq_error = []
                total = 0.0
                for j in range(k):
                    total += (estimates[j] - rd.means[j]) ** 2
                sq_error.append(total / k)
        max_open = max(max_open, policy.open_count())

    policy.sweep(duration)
    return EventRunResult(
        environment=env.name,
        policy=policy.name,
        seed=seed,
        times=tuple(times),
        regret=tuple(regret),
        normalizer=tuple(normalizer),
        best=tuple(best_means),
        sq_error=None if sq_error is None else tuple(sq_error),
        reward=tuple(reward),
        resolved=resolved,
        expired=policy.expired_count(),
        in_flight=in_flight,
        max_open=max_open,
    )
