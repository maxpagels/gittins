"""The event-time harness: traffic, delays, the event-queue runner,
and the phase-split metrics.

Same testing philosophy as test_sim.py: nothing is bit-pinned, but every
run is a pure function of (environment, traffic, delay, policy, seed), so
the assertions are exact-reproducibility checks, structural accounting
invariants, and loose evidence-based bounds on the same fixed seeds.
"""

from collections import Counter

from sim.environments import LinearEnvironment
from sim.event_runner import EventRunResult, run_events
from sim.metrics import normalized_regret, phase_regret
from sim.policies import (
    EpsilonGreedyPolicy,
    GittinsPolicy,
    GreedyPolicy,
    OraclePolicy,
    UniformPolicy,
)
from sim.traffic import (
    DAY,
    HOUR,
    ConstantDelay,
    DailyTraffic,
    ExponentialDelay,
    NextMorningDelay,
    phase,
)

BITS = 6
HORIZON = 2 * HOUR


def traffic():
    return DailyTraffic(peak_rate=0.04, trough_rate=0.002, bursts=1)


def engine(horizon=HORIZON):
    return GittinsPolicy(bits=BITS, horizon=horizon)


class TestTraffic:
    def test_arrivals_replay_and_increase(self):
        a = traffic().arrivals(0, DAY)
        b = traffic().arrivals(0, DAY)
        assert a == b
        assert all(0.0 < s < t < DAY for s, t in zip(a, a[1:]))
        assert traffic().arrivals(1, DAY) != a

    def test_rate_follows_the_daily_curve(self):
        # Peak hours must see far more traffic than the overnight trough —
        # the plan asks for a 10-100x swing (here 20x by construction).
        counts = Counter(phase(t) for t in traffic().arrivals(0, 5 * DAY))
        assert counts["peak"] > 10 * counts["trough"]

    def test_intensity_is_bounded_by_its_rates(self):
        tr = DailyTraffic(peak_rate=0.04, trough_rate=0.002)  # no bursts
        for t in range(0, int(DAY), 900):
            assert 0.002 <= tr.intensity(0, float(t)) <= 0.04 + 1e-12

    def test_bursts_multiply_the_rate(self):
        tr = traffic()
        start = tr.burst_windows(0, 0)[0]
        assert 0.0 <= start <= DAY - tr.burst_length
        assert tr.intensity(0, start) == 5.0 * tr.base_intensity(start)

    def test_phases_cover_the_day(self):
        assert phase(0.0) == "trough"
        assert phase(6 * HOUR) == "morning"
        assert phase(10 * HOUR) == "peak"
        assert phase(14 * HOUR) == "day"
        assert phase(19 * HOUR) == "peak"
        assert phase(DAY + 1.0) == "trough"  # the cycle repeats


class TestDelays:
    def test_constant(self):
        assert ConstantDelay(30.0).draw(0, 5, 1000.0) == 30.0

    def test_exponential_is_positive_and_replayable(self):
        d = ExponentialDelay(600.0)
        draws = [d.draw(0, i, 0.0) for i in range(200)]
        assert draws == [d.draw(0, i, 0.0) for i in range(200)]
        assert all(v > 0.0 for v in draws)
        assert 300.0 < sum(draws) / len(draws) < 1200.0  # loose mean check

    def test_next_morning_lands_next_day_at_hour(self):
        d = NextMorningDelay(hour=9.0, jitter=HOUR)
        for t in [2 * HOUR, 20 * HOUR, DAY + 13 * HOUR]:
            lands = t + d.draw(0, 0, t)
            assert int(lands // DAY) == int(t // DAY) + 1
            hour = (lands % DAY) / HOUR
            assert 9.0 <= hour < 10.0
        # Correlated with time of day: the evening waits less.
        assert d.draw(0, 0, 20 * HOUR) < d.draw(0, 0, 2 * HOUR)


class TestEventRunner:
    def test_replays_exactly(self):
        args = (LinearEnvironment(k=3), 7, traffic(), ExponentialDelay(2700.0), DAY)
        env, seed, tr, delay, duration = args
        a = run_events(env, engine(), seed, tr, delay, duration)
        b = run_events(env, engine(), seed, tr, delay, duration)
        assert a == b

    def test_reward_accounting(self):
        # Every decision's reward is either delivered in the run or still in
        # flight at the end — nothing is lost. Deliveries for decisions the
        # horizon already expired are no-ops but still deliveries.
        env = LinearEnvironment(k=5)
        result = run_events(env, engine(), 0, traffic(), ExponentialDelay(2700.0), DAY)
        assert result.resolved + result.in_flight == len(result.times)
        # The 45-minute mean has a real tail past the 2h horizon: expiry
        # does actual work here, and the ledger never grows unbounded.
        assert result.expired > 0
        assert 0 < result.max_open < len(result.times)

    def test_zero_delay_equals_the_round_loop(self):
        # With every reward delivered instantly, the event runner is the
        # round-based loop at wall-clock times: driving choose/observe by
        # hand over the same arrivals gives the identical regret series.
        env = LinearEnvironment(k=4)
        tr = traffic()
        result = run_events(env, engine(), 3, tr, ConstantDelay(0.0), DAY)
        policy = engine()
        policy.begin(3)
        regret = []
        for t in tr.arrivals(3, DAY):
            rd = env.round(3, t)
            chosen, _ = policy.choose(rd, 3)
            policy.observe(env.reward(3, t, rd, chosen))
            regret.append(max(rd.means) - rd.means[chosen])
        assert list(result.regret) == regret
        assert result.expired == 0 and result.in_flight == 0

    def test_next_morning_fills_the_ledger_overnight(self):
        # With every reward landing the next morning, the horizon must span
        # the night: by the end of day two the ledger's high-water mark is
        # an entire day's decisions, far beyond the fast-reward config's.
        env = LinearEnvironment(k=5)
        slow = run_events(env, engine(26 * HOUR), 0, traffic(), NextMorningDelay(), 2 * DAY)
        fast = run_events(env, engine(), 0, traffic(), ExponentialDelay(2700.0), 2 * DAY)
        assert slow.max_open > 5 * fast.max_open

    def test_policies_learn_under_delay(self):
        # Delayed, out-of-order rewards must still teach: both the engine
        # and the baselines end well inside uniform on the linear world.
        env = LinearEnvironment(k=5)
        for make in [
            lambda: GreedyPolicy(bits=BITS),
            lambda: EpsilonGreedyPolicy(0.1, bits=BITS),
            engine,
        ]:
            regrets = [
                normalized_regret(
                    run_events(env, make(), s, traffic(), ExponentialDelay(2700.0), DAY)
                )
                for s in [0, 1, 2]
            ]
            regrets.sort()
            assert regrets[1] < 0.9
        # ...and the paired bounds still anchor the scale.
        assert normalized_regret(
            run_events(env, OraclePolicy(), 0, traffic(), ExponentialDelay(2700.0), DAY)
        ) == 0.0
        u = normalized_regret(
            run_events(env, UniformPolicy(), 0, traffic(), ExponentialDelay(2700.0), DAY)
        )
        assert 0.8 < u < 1.2


class TestPhaseRegret:
    def test_splits_by_phase(self):
        # Two decisions in the trough (one perfect, one unit regret), one
        # in the peak (half a unit): the buckets are independent ratios.
        result = EventRunResult(
            environment="e",
            policy="p",
            seed=0,
            times=(1.0, 2.0, 10 * HOUR),
            regret=(0.0, 1.0, 0.5),
            normalizer=(1.0, 1.0, 1.0),
            best=(1.0, 1.0, 1.0),
            sq_error=None,
            reward=(1.0, 0.0, 0.5),
            resolved=3,
            expired=0,
            in_flight=0,
            max_open=0,
        )
        split = phase_regret(result, phase)
        assert split == {"trough": 0.5, "peak": 0.5}

    def test_engine_run_covers_the_phases(self):
        result = run_events(
            LinearEnvironment(k=5), engine(), 0, traffic(), ExponentialDelay(2700.0), DAY
        )
        split = phase_regret(result, phase)
        assert set(split) == {"peak", "trough", "morning", "day"}
        assert all(v >= 0.0 for v in split.values())
