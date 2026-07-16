# gittins — browser / JS binding

The gittins engine compiled to WebAssembly, exposing the same eight
functions as the Python package and the reference (`spec/api.md`). Build
with [wasm-pack](https://rustwasm.github.io/wasm-pack/):

```sh
wasm-pack build --target web bindings/wasm   # emits bindings/wasm/pkg/
```

As everywhere else (`spec/api.md`): calls return their result only, and
the state handle is updated in place. Decision records and resolutions are
plain objects with the same field names as the Python API;
`candidate_hash` is a BigInt.

A bandit that lives entirely in the browser, surviving reloads through
localStorage — the engine's whole state is one byte string:

```js
import init, * as gittins from "./pkg/gittins_wasm.js";
await init();

const saved = localStorage.getItem("bandit");
let state = saved
  ? gittins.deserialize(Uint8Array.from(atob(saved), (c) => c.charCodeAt(0)))
  : gittins.create(8, 3600.0); // bits, horizon seconds

const candidates = [
  ["banner-sale", { discount: 0.2 }],
  ["banner-new", { discount: 0.0 }],
  ["banner-plain", {}],
];
const record = gittins.decide(
  state,
  { device: "mobile", hour: new Date().getHours() },
  candidates,
  Date.now() / 1000,
  "browser-1",
);
const chosenArmId = candidates[record.chosen][0];
renderBanner(chosenArmId); // your code: act on the choice

// later, when the user clicks (or doesn't):
gittins.learn(state, record.decision_id, 1.0);
gittins.expire(state, Date.now() / 1000);

localStorage.setItem("bandit", btoa(String.fromCharCode(...gittins.serialize(state))));
```

The byte format is shared across implementations: a state saved in the
browser loads in Python or Rust unchanged. Tests
(`wasm-pack test --node bindings/wasm`) replay the golden `api` and
`serialization` sections through this module — the binding acceptance gate,
and, because WASM mandates IEEE-754 semantics, the project's cross-platform
bit-identity check.
