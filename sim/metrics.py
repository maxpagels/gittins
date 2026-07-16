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
    steady-state rate, where the epsilon mass's cost shows once learning is
    done."""
    if not (0.0 < window_fraction <= 1.0):
        raise ValueError("window_fraction must be in (0, 1]")
    n = len(result.regret)
    window = max(1, int(n * window_fraction))
    return normalized_regret(result, first=n - window)


def recovery_time(
    result: RunResult, event: int, window: int = 100, fraction: float = 0.9
) -> "int | None":
    """Rounds after `event` until the policy's rolling expected reward rate
    regains `fraction` of the oracle's — the post-shift/birth/return
    recovery measure. Rates are means over the forward window [r, r+window):
    the oracle's is `best`, the policy's is `best - regret`. Returns the
    first r - event that qualifies, or None if the run ends first.

    Meaningful when means are positive (a fraction of a negative oracle
    rate is not a target); the churn battery draws its means that way.
    """
    if window < 1:
        raise ValueError("window must be at least one round")
    n = len(result.regret)
    for r in range(event, n - window + 1):
        oracle = math.fsum(result.best[r : r + window]) / window
        policy = oracle - math.fsum(result.regret[r : r + window]) / window
        if policy >= fraction * oracle:
            return r - event
    return None


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
