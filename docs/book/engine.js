// The one shared engine loader. Every demo imports the engine through this
// module so wasm instantiation happens exactly once: the generated init's
// re-entry guard is not race-safe, and concurrent init() calls from
// several demos can instantiate the module repeatedly, stranding
// already-created bandit states in a dead instance.

import * as gittins from "./pkg/gittins_wasm.js";

export { gittins };
export const ready = gittins.default();
