# gittins

An opinionated, highly optimised contextual bandit engine.

- **Online by nature.** Learns one observation at a time, in O(1) work and
  fixed memory, as long as open decisions are regularly resolved.
- **Non-stationarity is expected.** The engine adapts as the relationship
  between context and feedback drifts, and never learns anything it cannot
  eventually unlearn.
- **Dynamic actions and context.** The candidate set can change on every
  call; you never declare the number of actions up front.
- **Safe reward handling.** Rewards may arrive late or never. Constructing
  invalid training data is made hard by design.
- **Speed and determinism.** Bit-identical results across platforms and
  language bindings, enforced by a golden test corpus.
- **Bring your own model.** Swap in your own scoring or exploration and
  inherit everything else.

Full documentation and user guide: **[docs.getgittins.dev](https://docs.getgittins.dev)**

## Install

```sh
npm install gittins
```

```js
import * as gittins from "gittins";
```

That works in Node (ESM or CommonJS) and under any bundler — webpack, Vite,
Rollup. There is no initialization step: the package ships a build per
environment and the right one is selected for you.

Straight from a CDN, no build step and no install, use the `/web` entry:

```js
import * as gittins from "https://esm.sh/gittins/web";
```

## Usage

Calls return their result only, and the state handle is updated in place.
Decision records and resolutions are plain objects with the same field names
as the Python API; `candidate_hash` is a BigInt.

A bandit that lives entirely in the browser, surviving reloads through
localStorage — the engine's whole state is one plain string, stored and
loaded as-is:

```js
import * as gittins from "gittins";

const saved = localStorage.getItem("bandit");
let state = saved ? gittins.deserialize(saved) : gittins.create(8, 3600.0); // bits, horizon seconds

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
gittins.learn(state, record.decision_id, 1.0, Date.now() / 1000);
gittins.expire(state, Date.now() / 1000);

localStorage.setItem("bandit", gittins.serialize(state));
```

## License

MIT. Named after John Gittins, whose index (1974) established that
exploration has a precise, computable value.
