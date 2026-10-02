#!/usr/bin/env bash
# Runs inside the container (build.sh starts it), in /src.
set -euo pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
DNNL=/opt/intel/oneapi/dnnl/2026.0

kernels() {
  echo "==> kernels: libh3sycl.so (icpx, SYCL + oneDNN)"
  icpx -O3 -fsycl -fPIC -shared -std=c++20 -Wall -I"$DNNL/include" kernels/h3sycl.cpp \
       -L"$DNNL/lib" -ldnnl -Wl,-rpath,'$ORIGIN' -o dist/libh3sycl.so
  cp -L "$DNNL/lib/libdnnl.so.3" dist/        # shipped beside the library: hosts without a shared oneDNN load it too
}
engine() {
  echo "==> engine: h3 (cargo, release)"
  cargo build --release --locked --manifest-path engine/Cargo.toml
  cp engine/target/release/h3 dist/
}
wfe() {
  echo "==> wfe: TypeScript -> dist/wfe"
  NODE=$(command -v node) wfe/build.sh
  rm -rf dist/wfe && cp -r wfe/build dist/wfe
}
tests() {
  echo "==> engine: tests and lints"
  cargo test --release --locked --manifest-path engine/Cargo.toml
  cargo clippy --release --locked --manifest-path engine/Cargo.toml -- -D warnings
}

[ $# -eq 0 ] && set -- kernels engine wfe
for what in "$@"; do
  case $what in
    kernels) kernels ;;
    engine) engine ;;
    wfe) wfe ;;
    test) tests ;;
    *) echo "build.sh: unknown target '$what' (kernels, engine, wfe, test)" >&2; exit 2 ;;
  esac
done
ls -la dist
