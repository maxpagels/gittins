// The one shared engine loader. Every demo imports the engine through this
// module, which loads it through pkg/gittins.js — the same auto-initializing
// entry the npm package ships as `gittins/web`, so the demos on this page run
// the engine exactly the way the book tells readers to load it.
//
// That entry awaits init at module evaluation, and a module body runs once no
// matter how many importers it has, so wasm instantiation happens exactly
// once. This is what the loader is for: the generated init's own re-entry
// guard is not race-safe, and concurrent init() calls from several demos can
// instantiate the module repeatedly, stranding already-created bandit states
// in a dead instance.

import * as gittins from "./pkg/gittins.js";

export { gittins };
// Kept so demos can `await engine.ready` unchanged. It is already resolved by
// the time this module finishes evaluating — awaiting the dynamic import of
// this file is itself enough — but leaving it costs nothing and keeps the
// demos from having to care how initialization happens.
export const ready = Promise.resolve();
