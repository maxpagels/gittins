// The non-stationarity demo mounted by [SIM-FORGET] in index.md (chapter
// "Learning to Forget"). Same live engine and chart as sim.js, but with a
// "flip the world" control that reverses every context's action rewards:
// yesterday's best action becomes today's worst. The oracle (row max) and
// the random policy (row mean) are unaffected by construction, so only the
// learner dips and recovers. Falls back to a plain message when
// WebAssembly is unavailable.

const mount = document.getElementById("bandit-sim-forget");

// Colors validated for CVD separation and contrast on the book surface.
const COLOR_BANDIT = "#b3261e";
const COLOR_RANDOM = "#2563eb";
const COLOR_ORACLE = "#777";
const INK_MUTED = "#777";
const GRID = "rgba(0,0,0,0.08)";
const FLIP_MARK = "rgba(0,0,0,0.3)";

const CONTEXTS = [
  { device: "mobile", daypart: "day" },
  { device: "mobile", daypart: "night" },
  { device: "desktop", daypart: "day" },
  { device: "desktop", daypart: "night" },
];
const ACTIONS = [["banner-sale", {}], ["banner-new", {}], ["banner-plain", {}]];
const P = [
  [0.12, 0.06, 0.02],
  [0.04, 0.10, 0.02],
  [0.03, 0.05, 0.08],
  [0.06, 0.02, 0.09],
];
// The 180: each row reversed, so per context the best action becomes the
// worst. Row max (oracle) and row mean (random) are identical either way.
const P_FLIPPED = P.map((row) => [...row].reverse());
const BEST = P.map((row) => Math.max(...row));
const Y_MAX = Math.max(...BEST) * 1.15;
const WINDOW = 2000;
const MAX_SAMPLES = 600;

function fallback(html) {
  mount.innerHTML = `<div class="demo-fallback">${html}</div>`;
}

async function boot() {
  if (typeof WebAssembly !== "object") {
    fallback("Your browser does not support WebAssembly, so the live simulation cannot run here.");
    return;
  }
  let gittins;
  try {
    const engine = await import("./engine.js");
    await engine.ready;
    gittins = engine.gittins;
  } catch (e) {
    fallback(
      "The simulation engine could not be loaded. If you are reading this " +
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
  mount.innerHTML = `
    <canvas class="sim-canvas" height="240"></canvas>
    <p class="sim-line"><button type="button" class="sim-btn sim-start">start</button>
    the engine (or <button type="button" class="sim-btn sim-reset">reset</button> it),
    let <span class="sim-nowrap"><span class="sim-swatch" style="background:${COLOR_BANDIT}"></span>Gittins</span>
    settle near the
    <span class="sim-nowrap"><span class="sim-swatch sim-swatch-line"></span>oracle</span>,
    then <button type="button" class="sim-btn sim-flip">flip the world</button>:
    every context's best action becomes its worst. Gittins dips and relearns
    from the still-arriving rewards; the oracle and the
    <span class="sim-nowrap"><span class="sim-swatch" style="background:${COLOR_RANDOM}"></span>random
    policy</span> never even notice &mdash;
    <span class="sim-read">no decisions made yet</span>.</p>
  `;
  const $ = (sel) => mount.querySelector(sel);
  const canvas = $(".sim-canvas");
  const ctx2d = canvas.getContext("2d");

  let state, tSim, steps, samples, world, flips;
  // Rolling window of each policy's expected reward per decision.
  let bufB, bufR, bufO, bufIdx, bufCount, sumB, sumR, sumO;
  let running = false;
  let recent = [];
  let hoverX = null;

  const meanB = () => (bufCount ? sumB / bufCount : 0);
  const meanR = () => (bufCount ? sumR / bufCount : 0);
  const meanO = () => (bufCount ? sumO / bufCount : 0);

  function reset() {
    state = g.create(8, 3600.0);
    tSim = 1_752_000_000;
    steps = 0;
    samples = [];
    world = P;
    flips = [];
    bufB = new Float64Array(WINDOW);
    bufR = new Float64Array(WINDOW);
    bufO = new Float64Array(WINDOW);
    bufIdx = 0;
    bufCount = 0;
    sumB = sumR = sumO = 0;
    recent = [];
    readout(0);
    draw();
  }

  function step() {
    const i = (Math.random() * CONTEXTS.length) | 0;
    tSim += 1;
    const record = g.decide(state, CONTEXTS[i], ACTIONS, tSim, "forget-sim");
    g.learn(state, record.decision_id, Math.random() < world[i][record.chosen] ? 1.0 : 0.0, tSim);

    const pB = world[i][record.chosen];
    const pR = world[i][(Math.random() * 3) | 0];
    const pO = BEST[i];
    if (bufCount === WINDOW) {
      sumB -= bufB[bufIdx];
      sumR -= bufR[bufIdx];
      sumO -= bufO[bufIdx];
    } else {
      bufCount += 1;
    }
    bufB[bufIdx] = pB;
    bufR[bufIdx] = pR;
    bufO[bufIdx] = pO;
    sumB += pB;
    sumR += pR;
    sumO += pO;
    bufIdx = (bufIdx + 1) % WINDOW;

    steps += 1;
  }

  function frame() {
    if (!running) return;
    const t0 = performance.now();
    let batch = 0;
    while (performance.now() - t0 < 8) {
      for (let k = 0; k < 200; k++) step();
      batch += 200;
    }
    samples.push({ n: steps, b: meanB(), r: meanR(), o: meanO() });
    if (samples.length > MAX_SAMPLES) samples.shift();
    const nowMs = performance.now();
    recent.push([nowMs, batch]);
    while (recent.length && nowMs - recent[0][0] > 1000) recent.shift();
    const winMs = recent.length > 1 ? nowMs - recent[0][0] : 1;
    const dps = (recent.reduce((s, x) => s + x[1], 0) / winMs) * 1000;
    readout(dps);
    draw();
    requestAnimationFrame(frame);
  }

  const fmt = (x) => x >= 1e6 ? (x / 1e6).toFixed(1) + "M"
    : x >= 1e3 ? (x / 1e3).toFixed(1) + "k" : Math.round(x).toString();
  const pct = (v) => (v * 100).toFixed(1) + "%";

  function readout(dps) {
    $(".sim-read").textContent = steps === 0
      ? "no decisions made yet"
      : `so far ${fmt(steps)} decisions at ${fmt(dps)} decisions per second ` +
        `and ${flips.length} flip${flips.length === 1 ? "" : "s"}, averaging ` +
        `${pct(meanB())} for Gittins against ${pct(meanR())} for random and ` +
        `${pct(meanO())} for the oracle`;
  }

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== w * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
    ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx2d.clearRect(0, 0, w, h);

    const ml = 44, mr = 14, mt = 14, mb = 22;
    const pw = w - ml - mr, ph = h - mt - mb;
    const n0 = samples.length ? samples[0].n : 0;
    const span = Math.max(steps - n0, 1);
    const X = (n) => ml + ((n - n0) / span) * pw;
    const Y = (v) => mt + ph - (v / Y_MAX) * ph;

    ctx2d.font = "italic 11px Georgia, serif";
    ctx2d.fillStyle = INK_MUTED;

    ctx2d.strokeStyle = GRID;
    ctx2d.lineWidth = 1;
    ctx2d.textAlign = "right";
    for (const level of [0.05, 0.1]) {
      ctx2d.beginPath();
      ctx2d.moveTo(ml, Y(level));
      ctx2d.lineTo(ml + pw, Y(level));
      ctx2d.stroke();
      ctx2d.fillText((level * 100).toFixed(0) + "%", ml - 4, Y(level) + 3);
    }
    ctx2d.strokeStyle = INK_MUTED;
    ctx2d.beginPath();
    ctx2d.moveTo(ml, Y(0));
    ctx2d.lineTo(ml + pw, Y(0));
    ctx2d.stroke();
    ctx2d.fillText("0", ml - 4, Y(0) + 3);

    ctx2d.textAlign = "left";
    ctx2d.fillText(`average reward, rolling ${fmt(WINDOW)} decisions`, ml, mt - 3);
    if (steps > 0) {
      ctx2d.textAlign = "left";
      ctx2d.fillText(fmt(n0), ml, h - 6);
      ctx2d.textAlign = "right";
      ctx2d.fillText(fmt(steps) + " decisions", ml + pw, h - 6);
    }

    // Flip markers still inside the visible window.
    ctx2d.strokeStyle = FLIP_MARK;
    ctx2d.setLineDash([4, 3]);
    ctx2d.textAlign = "left";
    for (const n of flips) {
      if (n < n0) continue;
      const x = X(n);
      ctx2d.beginPath();
      ctx2d.moveTo(x, mt);
      ctx2d.lineTo(x, mt + ph);
      ctx2d.stroke();
      ctx2d.fillText("flip", x + 4, mt + ph - 6);
    }
    ctx2d.setLineDash([]);

    const line = (key, color) => {
      ctx2d.strokeStyle = color;
      ctx2d.lineWidth = 2;
      ctx2d.beginPath();
      samples.forEach((s, k) => {
        const x = X(s.n), y = Y(s[key]);
        if (k === 0) ctx2d.moveTo(x, y); else ctx2d.lineTo(x, y);
      });
      ctx2d.stroke();
    };
    line("o", COLOR_ORACLE);
    line("r", COLOR_RANDOM);
    line("b", COLOR_BANDIT);

    if (hoverX !== null && samples.length > 0) {
      const n = n0 + ((hoverX - ml) / pw) * span;
      let s = samples[0];
      for (const cand of samples) if (Math.abs(cand.n - n) < Math.abs(s.n - n)) s = cand;
      const x = X(s.n);
      ctx2d.strokeStyle = GRID;
      ctx2d.beginPath();
      ctx2d.moveTo(x, mt);
      ctx2d.lineTo(x, mt + ph);
      ctx2d.stroke();
      for (const [key, color] of [["b", COLOR_BANDIT], ["r", COLOR_RANDOM], ["o", COLOR_ORACLE]]) {
        ctx2d.fillStyle = color;
        ctx2d.beginPath();
        ctx2d.arc(x, Y(s[key]), 4, 0, 2 * Math.PI);
        ctx2d.fill();
      }
      ctx2d.fillStyle = INK_MUTED;
      ctx2d.textAlign = x > ml + pw / 2 ? "right" : "left";
      ctx2d.fillText(
        `${fmt(s.n)}: gittins ${pct(s.b)} · random ${pct(s.r)} · oracle ${pct(s.o)}`,
        x + (x > ml + pw / 2 ? -8 : 8), mt + 12,
      );
    }
  }

  canvas.addEventListener("mousemove", (e) => {
    hoverX = e.offsetX;
    if (!running) draw();
  });
  canvas.addEventListener("mouseleave", () => {
    hoverX = null;
    if (!running) draw();
  });

  $(".sim-start").addEventListener("click", () => {
    running = !running;
    $(".sim-start").textContent = running ? "pause" : "start";
    if (running) requestAnimationFrame(frame);
  });
  $(".sim-reset").addEventListener("click", () => {
    running = false;
    $(".sim-start").textContent = "start";
    reset();
  });
  $(".sim-flip").addEventListener("click", () => {
    world = world === P ? P_FLIPPED : P;
    flips.push(steps);
    if (!running) draw();
  });

  window.addEventListener("resize", draw);
  reset();
}

boot();
