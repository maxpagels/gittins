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
//
// The import of that entry is deliberately kept dynamic, inside `ready`,
// rather than a static `import * as gittins from "./pkg/gittins.js"`. A static
// import would make *this* module async too, because its dependency
// top-level-awaits — and Safari resolves a dynamic `import()` of such a module
// before its body has evaluated, so a demo doing `await import("./engine.js")`
// then reading `engine.ready` hits "Cannot access 'ready' before
// initialization". Keeping the await inside a promise leaves this module
// synchronous, so both exports are initialized the moment it is imported, in
// every browser.

export let gittins;

export const ready = import("./pkg/gittins.js").then((module) => {
  gittins = module;
  return module;
});
