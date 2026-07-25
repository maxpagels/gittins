// The wasm leg of the decision-cycle benchmark. Invoked by
// bindings/python/bench.py with the grid configuration as a JSON argument,
// so all three implementations run exactly the same workload; emits
// per-cell seconds-per-decision as JSON on stdout. Needs the Node build of
// the wasm package first:
//
//     wasm-pack build --target nodejs --out-dir pkg-node bindings/wasm
//
// The catalog and context value formulas mirror bench.py's exactly.

const path = require("node:path");
const gittins = require(path.join(__dirname, "pkg-node", "gittins_wasm.js"));

const config = JSON.parse(process.argv[2]);

function catalogFor(arms) {
  const out = [];
  for (let a = 0; a < arms; a++) {
    out.push([`arm${a}`, { z0: (a % 7) * 0.25, z1: (a % 3) * 0.5 }]);
  }
  return out;
}

function contextsFor(nFeatures) {
  const out = [];
  for (let i = 0; i < config.variants; i++) {
    const context = { seg: "abcd"[i % 4] };
    for (let j = 0; j < nFeatures - 1; j++) {
      context[`f${j}`] = ((i + j) % 10) * 0.1;
    }
    out.push(context);
  }
  return out;
}

function drive(contexts, catalog) {
  const state = gittins.create(config.bits, 1e9);
  for (let i = 0; i < 5; i++) {
    const record = gittins.decide(state, contexts[i % contexts.length], catalog, i, "warm");
    gittins.learn(state, record.decision_id, 1.0, i);
  }
  let rounds = 0;
  const start = process.hrtime.bigint();
  const budget = BigInt(Math.round(config.min_seconds * 1e9));
  while (rounds < config.max_rounds) {
    const i = rounds;
    const record = gittins.decide(state, contexts[i % contexts.length], catalog, i, "bench");
    gittins.learn(state, record.decision_id, i % 3 ? 1.0 : 0.0, i);
    rounds += 1;
    if (process.hrtime.bigint() - start >= budget) break;
  }
  return Number(process.hrtime.bigint() - start) / 1e9 / rounds;
}

const results = [];
for (const arms of config.arm_counts) {
  const catalog = catalogFor(arms);
  for (const features of config.feature_counts) {
    results.push({ arms, features, seconds: drive(contextsFor(features), catalog) });
  }
}
console.log(JSON.stringify(results));
