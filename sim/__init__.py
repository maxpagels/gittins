"""Simulation harness for benchmarking the reference (Phase 0.5, PROGRESS.md).

Pure Python, zero runtime dependencies, like the reference — but *not* part of
`gittins_reference` and never bit-pinned: sims are statistical. Every run is
nonetheless seeded and exactly reproducible, because all environment and
policy randomness comes from the reference's own counter RNG keyed by
(name, seed, t) — see sim.rand.

The pieces:

    environments  — the environment protocol (context + candidates in,
                    stochastic reward out, oracle expected rewards exposed
                    so regret is exact) and the stationary battery
    policies      — the comparators, all driven through one identical loop:
                    gittins on the real encode -> decide -> ledger path,
                    bracketed by oracle/uniform and challenged by
                    greedy/epsilon-greedy
    runner        — one (environment, policy, seed) episode
    metrics       — normalized regret, final-window rate, prediction RMSE,
                    median/IQR aggregation over seeds
"""
