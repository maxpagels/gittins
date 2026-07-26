// Builds the publishable npm package, bindings/wasm/pkg-npm.
//
// wasm-pack emits one target per build, and each target has a different
// initialization story: `nodejs` and `bundler` wire the module up themselves,
// while `web` requires an explicit `await init()` because it has to fetch the
// .wasm over the network. Publishing only `web` — the obvious single choice,
// since it is what the docs demo uses — is what forces that init call onto
// every consumer, including the ones whose toolchain would have handled it.
//
// So we build all three into one package and let an `exports` map pick:
//
//   import * as gittins from "gittins"        -> node/ under Node, root under
//                                                a bundler; neither needs init
//   import * as gittins from "gittins/web"    -> web/, wrapped in a module
//                                                that awaits init() for you
//
// The `web` entry is a three-line wrapper rather than the raw wasm-pack
// output: top-level await lets the wrapper finish initializing before any
// importer's code runs, so the browser/CDN path is ceremony-free too. That
// is as far as it can go — browsers refuse to compile a >4KB wasm module
// synchronously on the main thread, so *something* has to be awaited. This
// moves the await to the import, where the language already handles it.
//
// The one field wasm-pack gets wrong for us is `name`: it takes the npm
// package name from the crate name, which has to stay `gittins-wasm` because
// it also names the emitted .js/.wasm files that the book imports by path.
// Everything else comes from bindings/wasm/Cargo.toml.

import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const crate = dirname(fileURLToPath(import.meta.url));
const out = join(crate, "pkg-npm");

rmSync(out, { recursive: true, force: true });

// The bundler build lands at the package root so it is the default entry;
// the other two go in subdirectories the exports map points at.
const targets = [
  ["bundler", out],
  ["nodejs", join(out, "node")],
  ["web", join(out, "web")],
];
for (const [target, dir] of targets) {
  execFileSync(
    "wasm-pack",
    ["build", "--release", "--target", target, "--out-dir", dir, crate],
    { stdio: ["ignore", "ignore", "inherit"] },
  );
  console.log(`built ${target} -> ${dir.replace(out, "pkg-npm")}`);
}

// The zero-ceremony browser entry. `export *` deliberately does not re-export
// init as the default, so there is no second, half-initialized way in.
writeFileSync(
  join(out, "web", "gittins.js"),
  [
    "// Auto-initializing entry: the top-level await below resolves before any",
    "// importer's code runs, so callers never call init() themselves.",
    'import init from "./gittins_wasm.js";',
    "await init();",
    'export * from "./gittins_wasm.js";',
    "",
  ].join("\n"),
);

// wasm-pack writes a full package.json into every out-dir; the nested ones
// are replaced with a single `type` field. They cannot be deleted outright:
// the root package.json says "type": "module" (the bundler build is ESM), and
// that would make Node parse the nodejs target's CommonJS as ESM and die on
// its first `exports.` assignment. A nested `type` scopes the module system
// per directory without shadowing the root exports map, which resolves
// against the package root regardless.
for (const [sub, type] of [
  ["node", "commonjs"],
  ["web", "module"],
]) {
  for (const junk of [".gitignore", "README.md", "LICENSE"]) {
    rmSync(join(out, sub, junk), { force: true });
  }
  writeFileSync(
    join(out, sub, "package.json"),
    JSON.stringify({ type }, null, 2) + "\n",
  );
}

const pkgPath = join(out, "package.json");
const pkg = JSON.parse(readFileSync(pkgPath, "utf8"));

pkg.name = "gittins";
pkg.exports = {
  ".": {
    types: "./gittins_wasm.d.ts",
    node: "./node/gittins_wasm.js",
    default: "./gittins_wasm.js",
  },
  "./web": {
    types: "./web/gittins_wasm.d.ts",
    default: "./web/gittins.js",
  },
  "./package.json": "./package.json",
};
// `files` came from the bundler build alone; ship the other two as well.
pkg.files = [...new Set([...(pkg.files ?? []), "node/", "web/"])];
pkg.sideEffects = ["./gittins_wasm.js", "./web/gittins.js", "./snippets/*"];

writeFileSync(pkgPath, JSON.stringify(pkg, null, 2) + "\n");
console.log(`package.json: name=${pkg.name} version=${pkg.version}`);
