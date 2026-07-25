"""Decision-cycle benchmark: the Python binding vs the pure-Python
reference on identical work, over a grid of candidate-set sizes and
context widths, full cycle (decide + learn) per round. Run by CI into the
job summary so boundary-crossing regressions in the binding are visible
next to the battery tables.

Each cell is time-boxed (at least MIN_SECONDS of sampling, at most
MAX_ROUNDS rounds), so heavy shapes don't blow the CI budget and light
shapes still sample enough rounds to be stable.

    python bindings/python/bench.py [--json OUT.json]
    python bindings/python/bench.py --compare BASE.json HEAD.json

`--json` writes the per-cell seconds next to the printed table;
`--compare` prints a delta table between two such dumps (CI uses this to
show a PR's numbers against its merge base, measured on the same runner).
The GITTINS_BENCH_ROOT environment variable points the core-binary and
wasm-package lookups at another checkout, so one harness can measure two
revisions' artifacts — the venv it runs under supplies that revision's
wheel and reference.
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
        learn(state, record.decision_id, 1.0, float(i))
    rounds = 0
    start = time.perf_counter()
    while rounds < MAX_ROUNDS:
        i = rounds
        record = decide(state, contexts[i % len(contexts)], catalog, float(i), "bench")
        learn(state, record.decision_id, 1.0 if i % 3 else 0.0, float(i))
        rounds += 1
        if time.perf_counter() - start >= MIN_SECONDS:
            break
    return (time.perf_counter() - start) / rounds


def cell(seconds: "float | None") -> str:
    if seconds is None:
        return "not built"
    return f"{seconds * 1e6:,.0f} µs ({1 / seconds:,.0f}/s)"


def bench_root():
    """The checkout whose built artifacts (core binary, wasm package) are
    measured: this file's repo unless GITTINS_BENCH_ROOT points elsewhere."""
    import os
    from pathlib import Path

    override = os.environ.get("GITTINS_BENCH_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


def wasm_grid() -> "dict[tuple[int, int], float] | None":
    """Per-cell seconds for the wasm binding under Node, on the identical
    workload (bindings/wasm/bench.cjs), or None if the Node build of the
    wasm package (`wasm-pack build --target nodejs --out-dir pkg-node
    bindings/wasm`) or Node itself is unavailable."""
    import json
    import shutil
    import subprocess

    wasm_dir = bench_root() / "bindings" / "wasm"
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


def core_grid() -> "dict[tuple[int, int], float] | None":
    """Per-cell seconds for the native Rust core with no binding boundary
    at all (core/src/bin/bench.rs) — the floor the bindings are measured
    against. Builds the binary on demand if cargo is available; returns
    None when neither the binary nor cargo can be had."""
    import shutil
    import subprocess

    root = bench_root()
    binary = root / "core" / "target" / "release" / "bench"
    if not binary.exists():
        cargo = shutil.which("cargo")
        if cargo is None:
            return None
        built = subprocess.run(
            [cargo, "build", "--release", "--bin", "bench",
             "--manifest-path", str(root / "core" / "Cargo.toml")],
            capture_output=True,
        )
        if built.returncode != 0 or not binary.exists():
            return None
    run = subprocess.run(
        [
            str(binary),
            str(BITS),
            str(MIN_SECONDS),
            str(MAX_ROUNDS),
            str(CONTEXT_VARIANTS),
            ",".join(map(str, ARM_COUNTS)),
            ",".join(map(str, FEATURE_COUNTS)),
        ],
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        return None
    out = {}
    for line in run.stdout.splitlines():
        arms, features, seconds = line.split()
        out[(int(arms), int(features))] = float(seconds)
    return out


def measure() -> "list[dict]":
    """Every cell of the grid, as one dict per (arms, features) shape with
    the per-implementation seconds (None where an implementation is not
    built or not installed)."""
    import gittins

    try:
        from gittins_reference import api
    except ImportError:
        api = None
    core = core_grid()
    wasm = wasm_grid()

    rows = []
    for arms in ARM_COUNTS:
        catalog = catalog_for(arms)
        for n_features in FEATURE_COUNTS:
            contexts = contexts_for(n_features)
            rows.append(
                {
                    "arms": arms,
                    "features": n_features,
                    "core": None if core is None else core.get((arms, n_features)),
                    "wheel": drive(
                        gittins.create, gittins.decide, gittins.learn, contexts, catalog
                    ),
                    "wasm": None if wasm is None else wasm.get((arms, n_features)),
                    "reference": None
                    if api is None
                    else drive(api.create, api.decide, api.learn, contexts, catalog),
                }
            )
    return rows


IMPLS = (
    ("core", "core (native Rust)"),
    ("wheel", "wheel (Python)"),
    ("wasm", "wasm (Node)"),
    ("reference", "reference (pure Python)"),
)


def print_markdown_table(
    header: "list[str]", rows: "list[list[str]]", small: bool = False
) -> None:
    """Pad every column to its widest cell so the table lines up in a
    terminal and in the raw job summary; the `---:` separators keep it
    valid (right-aligned) markdown — same convention as the CLI's sweep.
    `small` wraps every cell in `<sub>`, the one font-size lever GitHub's
    markdown renderer allows (it strips CSS), for wide tables."""
    if small:
        header = [f"<sub>{h}</sub>" for h in header]
        rows = [[f"<sub>{c}</sub>" for c in row] for row in rows]
    widths = [len(h) for h in header]
    for row in rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))
    line = lambda cells: "| " + " | ".join(c.rjust(w) for c, w in zip(cells, widths)) + " |"
    print(line(header))
    print(line(["-" * (w - 1) + ":" for w in widths]))
    for row in rows:
        print(line(row))


def print_table(rows: "list[dict]") -> None:
    print("### Decision-cycle benchmark")
    print()
    print(
        f"bits={BITS} (dim {1 << BITS}); context features counted per row "
        "(one categorical + numerics); every arm carries 2 action features "
        "on top; each cell is a full decide + learn cycle, per decision. "
        "Speedups are vs the pure-Python reference (core / wheel / wasm); "
        "the native core has no binding boundary at all, so the wheel/wasm "
        "gaps to it are the boundary tax."
    )
    print()
    table = []
    for r in rows:
        c, bound, w, pure = r["core"], r["wheel"], r["wasm"], r["reference"]
        if pure is None:
            reference, speedup = "not installed", "—"
        else:
            reference = cell(pure)
            speedup = " / ".join(
                "—" if seconds is None else f"{pure / seconds:,.1f}x"
                for seconds in (c, bound, w)
            )
        table.append(
            [str(r["arms"]), str(r["features"]), cell(c), cell(bound), cell(w),
             reference, speedup]
        )
    print_markdown_table(
        ["arms", "context features", "core (native Rust)", "wheel (Python)",
         "wasm (Node)", "reference (pure Python)", "speedup"],
        table,
    )


def delta_cell(base: "float | None", head: "float | None") -> str:
    if head is None and base is None:
        return "—"
    if head is None:
        return "removed"
    if base is None:
        return f"{cell(head)} (new)"
    change = (head - base) / base * 100.0
    return f"{base * 1e6:,.0f} → {head * 1e6:,.0f} ({change:+,.0f}%)"


def print_compare(base_rows: "list[dict]", head_rows: "list[dict]") -> None:
    """The delta table CI appends to the job summary: per cell and per
    implementation, µs per cycle on the merge base → on the PR head, both
    measured on the same runner. Negative percentages are faster."""
    base = {(r["arms"], r["features"]): r for r in base_rows}
    print("### Decision-cycle benchmark: PR vs main")
    print()
    print(
        "Each cell is µs per full decide + learn cycle, merge base → PR "
        "head, measured back to back on the same runner with the same "
        "harness. Negative percentages are faster. Cells are time-boxed "
        f"samples ({MIN_SECONDS}s), so differences within ~10% are usually "
        "runner noise, not regressions."
    )
    print()
    table = [
        [str(head["arms"]), str(head["features"])]
        + [
            delta_cell(base.get((head["arms"], head["features"]), {}).get(key), head[key])
            for key, _ in IMPLS
        ]
        for head in head_rows
    ]
    print_markdown_table(
        ["arms", "context features"] + [label for _, label in IMPLS], table, small=True
    )


def main(argv: "list[str]") -> None:
    import json
    from pathlib import Path

    if "--compare" in argv:
        i = argv.index("--compare")
        base_path, head_path = argv[i + 1], argv[i + 2]
        base_rows = json.loads(Path(base_path).read_text(encoding="utf-8"))
        head_rows = json.loads(Path(head_path).read_text(encoding="utf-8"))
        print_compare(base_rows, head_rows)
        return

    rows = measure()
    print_table(rows)
    if "--json" in argv:
        out = Path(argv[argv.index("--json") + 1])
        out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
