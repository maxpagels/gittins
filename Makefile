.PHONY: book book-wasm serve

book: docs/book/index.html

docs/book/index.html: docs/book/index.md docs/book/build.mjs
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
