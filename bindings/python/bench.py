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


def cell(seconds: "float | None") -> str:
    if seconds is None:
        return "not built"
    return f"{seconds * 1e6:,.0f} µs ({1 / seconds:,.0f}/s)"


def wasm_grid() -> "dict[tuple[int, int], float] | None":
    """Per-cell seconds for the wasm binding under Node, on the identical
    workload (bindings/wasm/bench.cjs), or None if the Node build of the
    wasm package (`wasm-pack build --target nodejs --out-dir pkg-node
    bindings/wasm`) or Node itself is unavailable."""
    import json
    import shutil
    import subprocess
    from pathlib import Path

    wasm_dir = Path(__file__).resolve().parents[1] / "wasm"
    node = shutil.which("node")
    if node is None or not (wasm_dir / "pkg-node" / "gittins_wasm.js").exists():
        return None
    config = json.dumps(
        {
            "bits": BITS,
            "arm_counts": list(ARM_COUNTS),
            "feature_counts": list(FEATURE_COUNTS),
            "min_seconds": MIN_SECONDS,
            "max_rounds": MAX_ROUNDS,
            "variants": CONTEXT_VARIANTS,
        }
    )
    run = subprocess.run(
        [node, str(wasm_dir / "bench.cjs"), config], capture_output=True, text=True
    )
    if run.returncode != 0:
        return None
    return {(r["arms"], r["features"]): r["seconds"] for r in json.loads(run.stdout)}


def main() -> None:
    import gittins

    try:
        from gittins_reference import api
    except ImportError:
        api = None
    wasm = wasm_grid()

    print("### Decision-cycle benchmark (bindings)")
    print()
    print(
        f"bits={BITS} (dim {1 << BITS}); context features counted per row "
        "(one categorical + numerics); every arm carries 2 action features "
        "on top; each cell is a full decide + learn cycle, per decision. "
        "Speedups are vs the pure-Python reference (wheel / wasm)."
    )
    print()
    print(
        "| arms | context features | wheel (Rust core) | wasm (Node) "
        "| reference (pure Python) | speedup |"
    )
    print("|---|---|---|---|---|---|")
    for arms in ARM_COUNTS:
        catalog = catalog_for(arms)
        for n_features in FEATURE_COUNTS:
            contexts = contexts_for(n_features)
            bound = drive(gittins.create, gittins.decide, gittins.learn, contexts, catalog)
            w = None if wasm is None else wasm.get((arms, n_features))
            if api is None:
                print(f"| {arms} | {n_features} | {cell(bound)} | {cell(w)} | not installed | — |")
                continue
            pure = drive(api.create, api.decide, api.learn, contexts, catalog)
            speedup = f"{pure / bound:,.1f}x"
            speedup += " / —" if w is None else f" / {pure / w:,.1f}x"
            print(
                f"| {arms} | {n_features} | {cell(bound)} | {cell(w)} | {cell(pure)} "
                f"| {speedup} |"
            )


if __name__ == "__main__":
    main()
