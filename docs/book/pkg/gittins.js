// Auto-initializing entry for the `web` wasm-pack target.
//
// The raw web build exports an `init` the caller has to await before touching
// anything; the top-level await below does it once, at module evaluation, so
// importers get a module that is already live. ES module semantics also make
// this the only way it can happen: a module body runs exactly once no matter
// how many importers it has, which removes the race the generated init's own
// re-entry guard does not protect against.
//
// `export *` deliberately does not re-export init as the default, so there is
// no second, half-initialized way in.
//
// Shipped as `web/gittins.js` in the npm package (see build-npm.mjs) and
// copied next to the book's engine as `docs/book/pkg/gittins.js` (see the
// book-wasm target) — one file, so the two cannot drift.
import init from "./gittins_wasm.js";
await init();
export * from "./gittins_wasm.js";
