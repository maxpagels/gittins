"""Traffic: when decisions arrive, and how late their rewards are.

Decision arrival *times* are part of the world (R4): reward delays race
the engine's wall-clock horizon, so the event-time battery needs an
arrival process. `DailyTraffic` is a non-homogeneous Poisson process with a daily
intensity curve — morning and evening peaks (Gaussian bumps), an overnight
trough, the rate swinging peak/trough-fold — plus optional burst windows
that multiply the rate for a stretch. Delay models say how long each
decision's reward takes to come home; `NextMorningDelay` is the
conversion-that-lands-tomorrow case, correlated with time of day.

Everything is a pure function of (name, seed) through sim.rand streams, so
arrival times and delays are identical for every policy on the same seed —
event-time comparisons stay paired. Sampling uses `math.log` (like
sim.rand's gaussian, not bit-portable across platforms; sims are
statistical).
"""

import math

from sim.rand import stream, uniform

DAY = 86400.0
HOUR = 3600.0

# The intensity curve's shape: Gaussian bumps at the two daily peaks.
PEAK_HOURS = (10.0, 19.0)
PEAK_WIDTH = 3.0  # hours

# Phase buckets for the phase-split metrics, in hours of day: the overnight
# trough, the first stretch after it (the morning re-exploration cost), and
# the bump tops. Everything else is ordinary daytime.
TROUGH_HOURS = (0.0, 5.0)
MORNING_HOURS = (5.0, 9.0)
PEAK_PHASE_HOURS = ((9.0, 12.0), (17.0, 21.0))


class DailyTraffic:
    """Decision arrivals over repeating days, by Poisson thinning.

    intensity(t) = trough_rate + (peak_rate - trough_rate) * s(hour), with
    s the sum of the two peak bumps (clamped to 1), times a burst factor
    when t falls inside one of the day's burst windows. Bursts are drawn
    per (name, seed, day): `bursts` windows of `burst_length` seconds at
    uniform start times, each multiplying the rate by `burst_factor`.
    """

    def __init__(
        self,
        peak_rate: float,
        trough_rate: float,
        bursts: int = 0,
        burst_factor: float = 5.0,
        burst_length: float = 900.0,
    ):
        if not (0.0 < trough_rate <= peak_rate):
            raise ValueError("need 0 < trough_rate <= peak_rate")
        self.name = f"traffic-p{peak_rate:g}-t{trough_rate:g}-b{bursts}"
        self.peak_rate = peak_rate
        self.trough_rate = trough_rate
        self.bursts = bursts
        self.burst_factor = burst_factor
        self.burst_length = burst_length
        self._windows: "dict[tuple[int, int], list[float]]" = {}

    def base_intensity(self, t: float) -> float:
        """The burst-free arrival rate at wall-clock time t (events/s)."""
        hour = (t % DAY) / HOUR
        s = 0.0
        for peak in PEAK_HOURS:
            s += math.exp(-(((hour - peak) / PEAK_WIDTH) ** 2))
        return self.trough_rate + (self.peak_rate - self.trough_rate) * min(s, 1.0)

    def burst_windows(self, seed: int, day: int) -> list[float]:
        """Start times of the day's burst windows; memoized."""
        if (seed, day) not in self._windows:
            key = stream(self.name, seed, f"bursts:{day}")
            self._windows[(seed, day)] = [
                day * DAY + uniform(key, b, 0.0, DAY - self.burst_length)
                for b in range(self.bursts)
            ]
        return self._windows[(seed, day)]

    def in_burst(self, seed: int, t: float) -> bool:
        return any(
            start <= t < start + self.burst_length
            for start in self.burst_windows(seed, int(t // DAY))
        )

    def intensity(self, seed: int, t: float) -> float:
        rate = self.base_intensity(t)
        if self.bursts and self.in_burst(seed, t):
            rate *= self.burst_factor
        return rate

    def arrivals(self, seed: int, duration: float) -> list[float]:
        """Decision timestamps in (0, duration), by thinning: candidate
        points from a homogeneous process at the intensity's upper bound,
        each kept with probability intensity/bound."""
        bound = self.peak_rate * (self.burst_factor if self.bursts else 1.0)
        key = stream(self.name, seed, "arrivals")
        times = []
        t = 0.0
        c = 0
        while True:
            t += -math.log(1.0 - uniform(key, c)) / bound
            c += 1
            if t >= duration:
                return times
            if uniform(key, c) * bound < self.intensity(seed, t):
                times.append(t)
            c += 1


def phase(t: float) -> str:
    """The traffic phase of wall-clock time t, for the phase-split metrics:
    'trough' (overnight), 'morning' (the first stretch after the trough,
    where the re-exploration cost shows), 'peak' (the bump tops), or 'day'.
    A fixed property of the daily shape, not of any one traffic instance."""
    hour = (t % DAY) / HOUR
    if TROUGH_HOURS[0] <= hour < TROUGH_HOURS[1]:
        return "trough"
    if MORNING_HOURS[0] <= hour < MORNING_HOURS[1]:
        return "morning"
    for lo, hi in PEAK_PHASE_HOURS:
        if lo <= hour < hi:
            return "peak"
    return "day"


class ConstantDelay:
    """Every reward takes exactly `delay` seconds."""

    def __init__(self, delay: float):
        if delay < 0.0:
            raise ValueError("delay must be nonnegative")
        self.name = f"const-{delay:g}s"
        self.delay = delay

    def draw(self, seed: int, index: int, t: float) -> float:
        return self.delay


class ExponentialDelay:
    """Memoryless delays with the given mean: most rewards come back fast,
    a tail takes several means — the tail past the horizon expires."""

    def __init__(self, mean: float):
        if not (mean > 0.0):
            raise ValueError("mean must be positive")
        self.name = f"exp-{mean:g}s"
        self.mean = mean

    def draw(self, seed: int, index: int, t: float) -> float:
        key = stream(self.name, seed, f"delay:{index}")
        return -self.mean * math.log(1.0 - uniform(key, 0))


class NextMorningDelay:
    """The reward lands the next day at `hour`, plus jitter — conversions
    that are only counted the following morning. Delay is correlated with
    the decision's time of day: an evening decision waits half as long as a
    morning one, and everything decided today resolves in one batch."""

    def __init__(self, hour: float = 9.0, jitter: float = HOUR):
        if not (0.0 <= hour < 24.0):
            raise ValueError("hour must be within the day")
        self.name = f"morning-{hour:g}h"
        self.hour = hour
        self.jitter = jitter

    def draw(self, seed: int, index: int, t: float) -> float:
        key = stream(self.name, seed, f"delay:{index}")
        lands = (int(t // DAY) + 1) * DAY + self.hour * HOUR + uniform(key, 0, 0.0, self.jitter)
        return lands - t
