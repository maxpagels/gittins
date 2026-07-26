// The headline throughput figure, mounted by [RATE] in index.md.
//
// Runs the real WASM engine on the same synthetic problem as the other demos
// for ten seconds and reports what it actually managed, so the number on the
// page is measured on the reader's own machine rather than quoted from ours.
//
// The rate is decisions divided by time spent *inside* the loop, not by wall
// clock. The loop deliberately yields to the browser between short bursts so
// the page stays responsive while it runs, and counting those idle gaps would
// report the yielding, not the engine.
//
// Degrades by disappearing: the clause is hidden until there is a real number
// to put in it, so with no JavaScript, no WebAssembly, or a failed load the
// line reads "Production-grade contextual bandits." and nothing is claimed.

const clause = document.getElementById("rate-clause");
const value = document.getElementById("rate-value");

// A realistically sized problem rather than a toy one: ten candidate arms
// scored against a thirty-feature context on every decision. Throughput is
// dominated by how much there is to hash and score, so a three-arm, two-feature
// world would report a number nobody could expect to see in production.
const ARMS = 10;
const FEATURES = 30;
const CONTEXT_COUNT = 4;

const ACTIONS = Array.from({ length: ARMS }, (_, a) => [
  `arm-${a}`,
  { slot: a % 3, family: `f${a % 4}` },
]);

// Fixed contexts, built once: a mix of categorical and numeric features, the
// two kinds the encoder treats differently.
const CONTEXTS = Array.from({ length: CONTEXT_COUNT }, (_, c) => {
  const ctx = {};
  for (let f = 0; f < FEATURES; f++) {
    ctx[`f${f}`] = f % 3 === 0 ? `v${(f + c) % 7}` : ((f * 7 + c * 13) % 100) / 100;
  }
  return ctx;
});

// Click probability per context and arm. Deterministic, and only there to make
// learn() do representative work — the headline measures speed, not regret.
const P = CONTEXTS.map((_, c) =>
  Array.from({ length: ARMS }, (_, a) => 0.02 + ((c * 3 + a * 5) % 11) / 100),
);

const RUN_MS = 4_000; // how long to measure for
const SLICE_MS = 8; // target work per frame; keeps the page interactive
const FIRST_BATCH = 200; // starting guess, retuned after the first frame
const MIN_BATCH = 50;
// A safety rail, not a target: tuning settles far below this. It bounds how
// long a single frame can block if the estimate ever goes wrong.
const MAX_BATCH = 20_000;
const MAX_GROWTH = 4; // per frame, so a bad reading cannot cause a huge jump

// The whole number, comma-grouped: "103,482", never "103k". The figure is the
// claim, and rounding it to two significant digits reads like marketing.
const fmt = (x) => Math.round(x).toLocaleString("en-US");

if (clause && value) run();

async function run() {
  let g;
  try {
    const engine = await import("./engine.js");
    await engine.ready;
    g = engine.gittins;
  } catch {
    return; // no engine, no claim: the clause stays hidden
  }

  const state = g.create(8, 3600.0);
  let t = 1_752_000_000;
  let decisions = 0;
  let busyMs = 0;
  let batch = FIRST_BATCH;
  const until = performance.now() + RUN_MS;

  function frame() {
    const started = performance.now();
    for (let k = 0; k < batch; k++) {
      const i = (Math.random() * CONTEXTS.length) | 0;
      t += 1;
      // A full decision cycle: choose, then resolve the reward. Resolving
      // every decision keeps the ledger bounded, so this measures steady
      // state rather than a run that slowly fills memory.
      const record = g.decide(state, CONTEXTS[i], ACTIONS, t, "rate");
      g.learn(
        state,
        record.decision_id,
        Math.random() < P[i][record.chosen] ? 1.0 : 0.0,
        t,
      );
    }
    const spent = performance.now() - started;
    decisions += batch;
    busyMs += spent;

    // Retune the batch toward SLICE_MS of work. One clock read per frame,
    // never a spin loop: a loop that polls the clock until a deadline pins a
    // core for no reason, and on a clock that does not advance — a headless
    // browser under virtual time, say — it never returns at all.
    //
    // `spent` can legitimately read 0: browsers clamp performance.now()
    // resolution, so a frame quicker than one tick is indistinguishable from
    // an instant one. Scaling by it would divide by zero and slam the batch
    // into MAX_BATCH, turning one frame into seconds of blocked main thread,
    // so an unmeasurable frame just doubles and the growth is capped either
    // way.
    if (spent >= 1) {
      const scaled = Math.round(batch * (SLICE_MS / spent));
      batch = Math.min(scaled, batch * MAX_GROWTH);
    } else {
      batch = batch * 2;
    }
    batch = Math.max(MIN_BATCH, Math.min(MAX_BATCH, batch));

    // Only claim a rate once one is measurable. Until the clock has actually
    // moved, decisions/busyMs is a division by zero, and rendering that would
    // put "InfinityM decisions/sec" on the page — briefly on a real browser
    // whose performance.now() is coarse, permanently on one whose clock is
    // frozen. The clause stays hidden instead, which is the same thing that
    // happens when there is no engine at all.
    const rate = decisions / (busyMs / 1000);
    if (Number.isFinite(rate) && rate > 0) {
      value.textContent = fmt(rate);
      clause.hidden = false;
    }

    if (performance.now() < until) requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}
