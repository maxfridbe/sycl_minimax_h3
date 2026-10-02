#!/usr/bin/env bash
# Offline build: vendored tsc, vendored snabbdom, no node_modules, no network. The result (build/) is a directory of
# ES modules plus index.html and style.css; the Rust server (engine/h3-wfe) serves it as it is.
#   ./build.sh          type-check + compile
#   ./build.sh --watch  recompile on save
#   ./build.sh --check  also run the smoke test against a running server (H3_WFE=http://host:port)
set -euo pipefail
cd "$(dirname "$0")"

NODE="${NODE:-$(command -v node || echo "$HOME/.local/bin/node")}"
[ -x "$NODE" ] || { echo "need node on PATH or at \$NODE (only the build needs it)"; exit 1; }
TSC="vendor/typescript/tsc.js"
[ -f "$TSC" ] || { echo "missing $TSC - the compiler is vendored, restore it from git"; exit 1; }

if [ "${1:-}" = "--watch" ]; then
  exec "$NODE" "$TSC" -p tsconfig.json --watch
fi

echo "==> tsc (strict)"
"$NODE" "$TSC" -p tsconfig.json
echo "==> copying snabbdom, index.html, style.css"
mkdir -p build/vendor
cp -r vendor/snabbdom build/vendor/
cp index.html style.css build/
if [ "${1:-}" = "--check" ]; then
  echo "==> smoke test (parse every served module, render both tabs headlessly)"
  "$NODE" check.mjs
fi
echo "==> done: $(find build -name '*.js' | wc -l) modules, $(du -sh build | cut -f1)"
