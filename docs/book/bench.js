// The live benchmark mounted by [BENCH] in index.md (chapter "Fast, and
// the Same Everywhere"). Measures decide+learn cycles per second on four
// problems of increasing size, then verifies determinism: a checksum of
// the model state after 1,000 fixed decisions, compared against the value
// the same code produces on every other machine (precomputed in the repo).

import { PROBLEMS, FINGERPRINT_DECISIONS, makeProblem, cycle, fingerprintGen }
  from "./bench-core.js";

const mount = document.getElementById("bandit-bench");

// Expected fingerprints, computed offline with the same engine and the
// same bench-core.js. If a row shows a mismatch, determinism is broken.
const EXPECTED = {
  "A/B/C test": "7d14f052",
  "Banner picker": "d2df0c88",
  "Personalisation, wide": "3bfef455",
  "Recommender": "5520b087",
  "Big catalogue": "02c6fbf0",
};

function fallback(html) {
  mount.innerHTML = `<div class="demo-fallback">${html}</div>`;
}

async function boot() {
  if (typeof WebAssembly !== "object") {
    fallback("Your browser does not support WebAssembly, so the live benchmark cannot run here.");
    return;
  }
  let gittins;
  try {
    const engine = await import("./engine.js");
    await engine.ready;
    gittins = engine.gittins;
  } catch (e) {
    fallback(
      "The benchmark engine could not be loaded. If you are reading this " +
      "as a local file, serve the book over HTTP (<code>make serve</code>) " +
      "and reload this page." +
    "<br><br>Reason: <code>" +
    String((e && (e.message || e)) || "unknown")
      .replace(/&/g, "&amp;").replace(/</g, "&lt;") + "</code>"
    );
    return;
  }
  run(gittins);
}

function run(g) {
  const fmt = (x) => x >= 1e6 ? (x / 1e6).toFixed(1) + "M"
    : x >= 1e3 ? (x / 1e3).toFixed(1) + "k" : Math.round(x).toString();


  mount.innerHTML = `
    <div class="table-wrap"><table class="bench-table">
      <thead><tr>
        <th>problem</th><th>actions</th><th>features<br>per action</th>
        <th>cycles run</th><th>cycles/s</th><th>µs/cycle</th><th>state fingerprint</th>
      </tr></thead>
      <tbody>${PROBLEMS.map((p) => `
        <tr>
          <td>${p.name}</td>
          <td>${fmt(p.actions)}</td>
          <td>${p.actFeatures}</td>
          <td class="bench-n">–</td>
          <td class="bench-dps">–</td>
          <td class="bench-us">–</td>
          <td class="bench-fp">–</td>
        </tr>`).join("")}
      </tbody>
    </table></div>
    <p class="sim-line">Four problems, measured in your browser right now:
    <button type="button" class="sim-btn bench-run">run the benchmark</button>.
    One cycle is a full <em>decide</em> over every action plus a
    <em>learn</em> on the outcome. The fingerprint is a checksum of the
    model state after a fixed sequence of decisions; it
    reads exactly the same on every machine, browser, and operating system
    &mdash; <span class="sim-read bench-status">not run yet</span>.</p>
  `;
  const rows = [...mount.querySelectorAll("tbody tr")];
  const btn = mount.querySelector(".bench-run");
  const status = mount.querySelector(".bench-status");
  const raf = () => new Promise((r) => requestAnimationFrame(r));

  async function benchmark() {
    btn.disabled = true;
    for (const row of rows) {
      for (const sel of [".bench-n", ".bench-dps", ".bench-us", ".bench-fp"]) {
        row.querySelector(sel).textContent = "–";
      }
    }
    await raf();
    for (let i = 0; i < PROBLEMS.length; i++) {
      const problem = makeProblem(PROBLEMS[i]);
      status.textContent = `measuring "${problem.name}"`;
      await raf();

      // Timing: batches of ~12ms of work, UI breather between batches,
      // only active time counted, until ~350ms has been measured.
      const state = g.create(problem.bits, 3600.0);
      let t = 2_000_000_000;
      let n = 0;
      let active = 0;
      while (active < 350) {
        const b0 = performance.now();
        do {
          t += 1;
          cycle(g, state, problem, n, t);
          n += 1;
        } while (performance.now() - b0 < 12);
        active += performance.now() - b0;
        await raf();
      }
      const dps = n / (active / 1000);
      rows[i].querySelector(".bench-n").textContent = fmt(n);
      rows[i].querySelector(".bench-dps").textContent = fmt(dps);
      rows[i].querySelector(".bench-us").textContent = fmt(1e6 / dps);

      // Determinism: the fixed sequence, chunked to keep the page alive.
      status.textContent = `fingerprinting "${problem.name}"`;
      // ~12ms of work per chunk, sized from the throughput just measured.
      const chunk = Math.max(1, Math.min(200, Math.round(dps * 0.012)));
      const total = problem.fpDecisions ?? FINGERPRINT_DECISIONS;
      const it = fingerprintGen(g, problem, chunk);
      let r;
      for (;;) {
        r = it.next();
        if (r.done) break;
        status.textContent = `fingerprinting "${problem.name}" (${r.value}/${total})`;
        await raf();
      }
      const ok = r.value === EXPECTED[problem.name];
      rows[i].querySelector(".bench-fp").innerHTML =
        `<code>${r.value}</code> ${ok ? "✓" : "✗ expected " + EXPECTED[problem.name]}`;
    }
    status.textContent = "done; every fingerprint above should carry a ✓";
    btn.disabled = false;
  }

  btn.addEventListener("click", benchmark);
}

boot();
