"""Regret metrics over run results.

The headline number is normalized cumulative regret: cumulative exact regret
divided by the cumulative expected regret of uniform random on the same
rounds, so 0 = oracle and 1 = uniform, comparable across environments, arm
counts, and reward scales. Aggregation over seeds is median and IQR —
sims are statistical, distributions matter more than means.
"""

import math

from sim.runner import RunResult


def normalized_regret(result: RunResult, first: int = 0, last: "int | None" = None) -> float:
    """Cumulative regret over rounds [first, last) as a fraction of uniform
    random's expected cumulative regret on the same rounds (0 = oracle,
    1 = uniform). Rounds where every candidate is equally good contribute
    nothing to either sum; if the whole slice is like that, regret is 0."""
    total = math.fsum(result.regret[first:last])
    unit = math.fsum(result.normalizer[first:last])
    if unit == 0.0:
        return 0.0
    return total / unit


def final_window_regret(result: RunResult, window_fraction: float = 0.1) -> float:
    """Normalized regret over just the final window of the run — the
    steady-state rate, where the floor's cost shows once learning is done."""
    if not (0.0 < window_fraction <= 1.0):
        raise ValueError("window_fraction must be in (0, 1]")
    n = len(result.regret)
    window = max(1, int(n * window_fraction))
    return normalized_regret(result, first=n - window)


def rmse(result: RunResult, first: int = 0, last: "int | None" = None) -> "float | None":
    """Root-mean-square prediction error vs the oracle means over rounds
    [first, last); None for model-free policies. Separates model quality
    from policy quality."""
    if result.sq_error is None:
        return None
    window = result.sq_error[first:last]
    return math.sqrt(math.fsum(window) / len(window))


def quantile(values: "list[float]", q: float) -> float:
    """The q-quantile with linear interpolation (numpy's default rule)."""
    if not values:
        raise ValueError("need at least one value")
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo])


def median_iqr(values: "list[float]") -> "tuple[float, float, float]":
    """(median, 25th percentile, 75th percentile) over seeds."""
    return quantile(values, 0.5), quantile(values, 0.25), quantile(values, 0.75)
