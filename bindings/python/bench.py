"""Decision-cycle benchmark: the Python binding vs the pure-Python
reference on identical work — the roadmap's headline case (bits=8, so a
256-dimensional model, with 100 candidate arms), full cycle (decide +
learn) per round. Run by CI into the job summary so boundary-crossing
regressions in the binding are visible next to the battery tables.

    python bindings/python/bench.py
"""

import time

ROUNDS = 2_000
BITS = 8
ARMS = 100

CATALOG = [(f"arm{a}", {"z0": (a % 7) * 0.25, "z1": (a % 3) * 0.5}) for a in range(ARMS)]


def drive(create, decide, learn, rounds: int) -> float:
    """Seconds for `rounds` full decision cycles through one implementation."""
    state = create(bits=BITS, horizon=1e9)
    start = time.perf_counter()
    for i in range(rounds):
        context = {"f0": (i % 10) * 0.1, "f1": (i % 5) * 0.2, "seg": "a" if i % 2 else "b"}
        record, state = decide(state, context, CATALOG, float(i), "bench")
        _, state = learn(state, record.decision_id, 1.0 if i % 3 else 0.0)
    return time.perf_counter() - start


def report(name: str, seconds: float) -> None:
    per = seconds / ROUNDS
    print(f"| {name} | {per * 1e6:,.0f} µs | {1 / per:,.0f} |")


def main() -> None:
    print("### Decision-cycle benchmark (Python binding)")
    print()
    print(
        f"bits={BITS} (dim {1 << BITS}), {ARMS} arms with action features, "
        f"{ROUNDS} rounds of decide + learn each."
    )
    print()
    print("| implementation | per decision | decisions/s |")
    print("|---|---|---|")

    import gittins

    drive(gittins.create, gittins.decide, gittins.learn, 50)  # warm up
    report("binding (Rust core)", drive(gittins.create, gittins.decide, gittins.learn, ROUNDS))

    try:
        from gittins_reference import api
    except ImportError:
        print("| reference (pure Python) | not installed | — |")
        return
    report("reference (pure Python)", drive(api.create, api.decide, api.learn, ROUNDS))


if __name__ == "__main__":
    main()
