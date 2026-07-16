"""Decision-cycle benchmark: the Python binding vs the pure-Python
reference on identical work, over a grid of candidate-set sizes and
context widths, full cycle (decide + learn) per round. Run by CI into the
job summary so boundary-crossing regressions in the binding are visible
next to the battery tables.

Each cell is time-boxed (at least MIN_SECONDS of sampling, at most
MAX_ROUNDS rounds), so heavy shapes don't blow the CI budget and light
shapes still sample enough rounds to be stable.

    python bindings/python/bench.py
"""

import time

BITS = 8
ARM_COUNTS = (5, 10, 100)
FEATURE_COUNTS = (5, 10, 100)  # context features: one categorical + numerics
MIN_SECONDS = 0.4
MAX_ROUNDS = 2_000
CONTEXT_VARIANTS = 50  # prebuilt so dict construction stays out of the timing


def catalog_for(arms: int):
    """`arms` candidates, each with two action features."""
    return [(f"arm{a}", {"z0": (a % 7) * 0.25, "z1": (a % 3) * 0.5}) for a in range(arms)]


def contexts_for(n_features: int):
    """Rotating context dicts with exactly `n_features` entries each:
    one categorical plus n-1 numerics, values varying per variant."""
    out = []
    for i in range(CONTEXT_VARIANTS):
        context = {"seg": "abcd"[i % 4]}
        for j in range(n_features - 1):
            context[f"f{j}"] = ((i + j) % 10) * 0.1
        out.append(context)
    return out


def drive(create, decide, learn, contexts, catalog) -> float:
    """Seconds per full decision cycle through one implementation,
    time-boxed sampling."""
    state = create(bits=BITS, horizon=1e9)
    for i in range(5):  # warm up (fills the reference's hash cache, too)
        record = decide(state, contexts[i % len(contexts)], catalog, float(i), "warm")
        learn(state, record.decision_id, 1.0)
    rounds = 0
    start = time.perf_counter()
    while rounds < MAX_ROUNDS:
        i = rounds
        record = decide(state, contexts[i % len(contexts)], catalog, float(i), "bench")
        learn(state, record.decision_id, 1.0 if i % 3 else 0.0)
        rounds += 1
        if time.perf_counter() - start >= MIN_SECONDS:
            break
    return (time.perf_counter() - start) / rounds


def cell(seconds: float) -> str:
    return f"{seconds * 1e6:,.0f} µs ({1 / seconds:,.0f}/s)"


def main() -> None:
    import gittins

    try:
        from gittins_reference import api
    except ImportError:
        api = None

    print("### Decision-cycle benchmark (Python binding)")
    print()
    print(
        f"bits={BITS} (dim {1 << BITS}); context features counted per row "
        "(one categorical + numerics); every arm carries 2 action features "
        "on top; each cell is a full decide + learn cycle, per decision."
    )
    print()
    print("| arms | context features | binding (Rust core) | reference (pure Python) | speedup |")
    print("|---|---|---|---|---|")
    for arms in ARM_COUNTS:
        catalog = catalog_for(arms)
        for n_features in FEATURE_COUNTS:
            contexts = contexts_for(n_features)
            bound = drive(gittins.create, gittins.decide, gittins.learn, contexts, catalog)
            if api is None:
                print(f"| {arms} | {n_features} | {cell(bound)} | not installed | — |")
                continue
            pure = drive(api.create, api.decide, api.learn, contexts, catalog)
            print(
                f"| {arms} | {n_features} | {cell(bound)} | {cell(pure)} "
                f"| {pure / bound:,.1f}x |"
            )


if __name__ == "__main__":
    main()
