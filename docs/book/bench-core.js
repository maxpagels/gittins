// Shared, DOM-free core of the benchmark demo: the four problems, their
// deterministic generation, and the fingerprint run. Imported by bench.js
// (the page widget) and by the repo's offline checker, so both execute the
// exact same decision sequence — which is the point: the resulting state is
// bit-identical everywhere.

export const PROBLEMS = [
  { name: "A/B/C test", actions: 3, ctxFeatures: 2, actFeatures: 1, bits: 8 },
  { name: "Banner picker", actions: 10, ctxFeatures: 6, actFeatures: 3, bits: 12 },
  { name: "Personalisation, wide", actions: 20, ctxFeatures: 20, actFeatures: 10, bits: 16 },
  { name: "Recommender", actions: 50, ctxFeatures: 10, actFeatures: 200, bits: 16, fpDecisions: 250 },
  { name: "Big catalogue", actions: 1000, ctxFeatures: 5, actFeatures: 100, bits: 16, fpDecisions: 100 },
];

// Length of the fixed fingerprint run; heavy problems override it with
// spec.fpDecisions so the run stays quick in a browser.
export const FINGERPRINT_DECISIONS = 1000;

// Deterministic PRNG (integer arithmetic + one exact division), so feature
// values are the same float64s on every platform.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeProblem(spec) {
  const rnd = mulberry32(1234567 + spec.actions);
  const features = (prefix, count) => {
    const f = {};
    for (let k = 0; k < count; k++) f[prefix + k] = Math.round(rnd() * 100) / 100;
    return f;
  };
  const candidates = [];
  for (let a = 0; a < spec.actions; a++) {
    candidates.push(["action-" + a, features("f", spec.actFeatures)]);
  }
  const contexts = [];
  for (let c = 0; c < 16; c++) contexts.push(features("c", spec.ctxFeatures));
  return { ...spec, candidates, contexts };
}

// One deterministic decide+learn cycle. Rewards derive from the step and
// the choice, so the whole run is a pure function of the problem spec.
export function cycle(g, state, problem, step, t) {
  const record = g.decide(
    state, problem.contexts[step % 16], problem.candidates, t, "bench",
  );
  g.learn(state, record.decision_id, (step + record.chosen) % 3 === 0 ? 1.0 : 0.0);
}

// Fresh state, FINGERPRINT_DECISIONS fixed cycles, then a checksum of the
// serialized state. Same 8 hex chars on every machine, browser, and OS.
// The generator yields every `chunk` cycles so a page can keep its UI
// responsive; the chunking cannot affect the decision sequence.
export function* fingerprintGen(g, problem, chunk = 100) {
  const state = g.create(problem.bits, 3600.0);
  let t = 1_752_000_000;
  const decisions = problem.fpDecisions ?? FINGERPRINT_DECISIONS;
  for (let s = 0; s < decisions; s++) {
    t += 1;
    cycle(g, state, problem, s, t);
    if ((s + 1) % chunk === 0) yield s + 1;
  }
  return fnv1a(g.serialize(state));
}

export function fingerprint(g, problem) {
  const it = fingerprintGen(g, problem);
  for (;;) {
    const r = it.next();
    if (r.done) return r.value;
  }
}

export function fnv1a(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}
