.PHONY: book book-wasm npm-pkg npm-publish serve ope-verify ope-eval ope-sweep ope-verify-gz ope-demo release

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

# The publishable npm package, in bindings/wasm/pkg-npm: all three wasm-pack
# targets behind one exports map, so no consumer has to call init(). See the
# header of build-npm.mjs for why that takes three builds and not one.
#
# Its own out-dir, so it can never race the pkg/ that book-wasm overwrites
# with a differently-targeted build.
npm-pkg:
	node bindings/wasm/build-npm.mjs

# Publish that package to npm. Requires `npm login` (see the trusted-publisher
# note in .github/workflows/release.yml once the first release is out).
npm-publish: npm-pkg
	npm publish bindings/wasm/pkg-npm --access public

serve: book
	python3 -m http.server 8000 --directory docs/book

# Cut a release: bump every shipped version, commit, tag, push. The tag
# push is what triggers .github/workflows/release.yml, which builds the
# wheels and publishes to PyPI.
#
# Run from main with a clean tree. Unlike a pure-Python release this bumps
# five manifests, four lock files, and the book, so the sequence is bump
# (no commit, no tag) -> refresh locks -> rebuild the book -> one commit ->
# tag. Letting bump-my-version commit for us would put the tag on a commit
# whose Cargo.lock files and version badge still carried the old version —
# a released tag that does not build reproducibly.
#
# The golden corpus runs before anything is written: it is the contract
# that all four bindings implement the same engine, so a release that
# fails it is a release that should not exist.
release:  # usage: make release BUMP=patch|minor|major
	@test -n "$(BUMP)" || { echo "Usage: make release BUMP=patch|minor|major"; exit 1; }
	@test "$$(git rev-parse --abbrev-ref HEAD)" = main || { echo "Release from main only (on $$(git rev-parse --abbrev-ref HEAD))."; exit 1; }
	@git diff --quiet && git diff --cached --quiet || { echo "Working tree not clean — commit or stash first."; exit 1; }
	uv lock --check
	uv run pytest -q
	cargo test --manifest-path core/Cargo.toml
	uvx bump-my-version bump $(BUMP)
	@for m in core bindings/python bindings/wasm bindings/cli; do \
	  cargo metadata --manifest-path $$m/Cargo.toml --format-version 1 >/dev/null; \
	done
	# Rebuild the book so the published version badge matches the release.
	# build.mjs reads the version straight out of core/Cargo.toml, which the
	# bump just rewrote, so this needs no version argument. The build is
	# byte-deterministic, so it only shows up in the diff when something
	# actually changed. Invoked directly rather than via `make book`, whose
	# mtime rule would skip the rebuild if the timestamps happened to line up.
	node docs/book/build.mjs
	@v=$$(uvx bump-my-version show current_version); \
	git add -A; \
	git commit -m "Release v$$v"; \
	git tag -a "v$$v" -m "v$$v"
	git push --follow-tags
