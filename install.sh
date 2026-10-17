#!/bin/sh
# Installs the `gittins` command-line tool from a GitHub Release.
#
#   curl -fsSL https://github.com/maxpagels/gittins/releases/latest/download/install.sh | sh
#
# Environment:
#   GITTINS_VERSION      the release to install, e.g. 1.0.9 (default: latest)
#   GITTINS_INSTALL_DIR  where the binary goes (default: ~/.local/bin)
#
# Released alongside the binaries by .github/workflows/release.yml, which
# names the archives gittins-<target>.tar.gz and lists them in SHA256SUMS;
# the target names below must keep matching that workflow's matrix.
#
# POSIX sh, not bash: minimal containers (Alpine) have no bash. Everything
# runs inside main, called on the last line, so a download cut off halfway
# through the pipe executes nothing.

set -eu

REPO="maxpagels/gittins"

say() {
    printf 'gittins-install: %s\n' "$1" >&2
}

fail() {
    say "error: $1"
    exit 1
}

target() {
    case "$(uname -m)" in
        x86_64 | amd64) arch=x86_64 ;;
        arm64 | aarch64) arch=aarch64 ;;
        *) fail "unsupported architecture: $(uname -m)" ;;
    esac
    case "$(uname -s)" in
        # musl, statically linked: one binary for every distro, Alpine included.
        Linux) echo "$arch-unknown-linux-musl" ;;
        Darwin) echo "$arch-apple-darwin" ;;
        *) fail "unsupported OS: $(uname -s); on Windows, download the zip from https://github.com/$REPO/releases" ;;
    esac
}

download() {
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --proto '=https' --tlsv1.2 -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --https-only -O "$2" "$1"
    else
        fail "need curl or wget to download"
    fi
}

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d' ' -f1
    else
        fail "need sha256sum or shasum to verify the download"
    fi
}

main() {
    target=$(target)
    archive="gittins-$target.tar.gz"
    dir="${GITTINS_INSTALL_DIR:-$HOME/.local/bin}"
    if [ -n "${GITTINS_VERSION:-}" ]; then
        base="https://github.com/$REPO/releases/download/v${GITTINS_VERSION#v}"
    else
        base="https://github.com/$REPO/releases/latest/download"
    fi

    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT

    say "downloading $archive"
    download "$base/$archive" "$tmp/$archive" || fail "could not download $base/$archive"
    download "$base/SHA256SUMS" "$tmp/SHA256SUMS" || fail "could not download $base/SHA256SUMS"

    expected=$(awk -v f="$archive" '$2 == f { print $1 }' "$tmp/SHA256SUMS")
    [ -n "$expected" ] || fail "$archive is not listed in SHA256SUMS"
    [ "$(sha256 "$tmp/$archive")" = "$expected" ] || fail "checksum mismatch for $archive"

    tar -xzf "$tmp/$archive" -C "$tmp"
    mkdir -p "$dir"
    # Copy beside the destination, then rename over it: an upgrade never
    # leaves a half-written binary, even if gittins is running.
    cp "$tmp/gittins" "$dir/.gittins.tmp"
    chmod 755 "$dir/.gittins.tmp"
    mv -f "$dir/.gittins.tmp" "$dir/gittins"

    say "installed $("$dir/gittins" --version) to $dir/gittins"
    case ":$PATH:" in
        *":$dir:"*) ;;
        *) say "$dir is not on your PATH; add it with: export PATH=\"$dir:\$PATH\"" ;;
    esac
}

main "$@"
