.PHONY: book book-wasm serve ope-verify ope-eval ope-sweep ope-verify-gz ope-demo

# The CLI (bindings/cli) against the committed example experience log
# (examples/decisions.jsonl, regenerable with examples/generate.py).
# `make ope-demo` runs the whole tour in order.
GITTINS_CLI = cargo run --manifest-path bindings/cli/Cargo.toml --
OPE_LOG = examples/decisions.jsonl

ope-verify:
	$(GITTINS_CLI) verify --log $(OPE_LOG)

ope-eval:
	$(GITTINS_CLI) eval --log $(OPE_LOG) --bits 8

ope-sweep:
	$(GITTINS_CLI) sweep --log $(OPE_LOG) --bits 8 --epsilon 0.02,0.05,0.1

ope-verify-gz:
	gzip -kf $(OPE_LOG)
	$(GITTINS_CLI) verify --log $(OPE_LOG).gz

ope-demo: ope-verify ope-eval ope-sweep ope-verify-gz

book: docs/book/index.html

docs/book/index.html: docs/book/index.md docs/book/build.mjs core/Cargo.toml
	node docs/book/build.mjs

# Build the WASM engine and copy it into the book so the live demo works
# from a plain checkout. The copies in docs/book/pkg are checked in;
# rerun this target whenever the engine changes.
book-wasm:
	wasm-pack build --release --target web bindings/wasm
	mkdir -p docs/book/pkg
	cp bindings/wasm/pkg/gittins_wasm.js docs/book/pkg/
	cp bindings/wasm/pkg/gittins_wasm_bg.wasm docs/book/pkg/

serve: book
	python3 -m http.server 8000 --directory docs/book
