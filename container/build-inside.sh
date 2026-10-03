#!/usr/bin/env bash
# Runs inside the container (build.sh starts it), in /src.
set -euo pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
DNNL=${H3_DNNL:-/opt/onednn}      # oneDNN: headers in include/, the library in lib/ (the image builds 3.12 there)
# a oneDNN with container/onednn-sdpa-no-fallback.patch says so with this file: only then may attention use large
# chunks (kernels/h3sycl.cpp, attention_fused)
SDPA_FLAG=; [ -f "$DNNL/H3_SDPA_NO_FALLBACK" ] && SDPA_FLAG=-DH3S_SDPA_NO_FALLBACK

kernels() {
  echo "==> kernels: libh3sycl.so (icpx, SYCL + oneDNN)"
  icpx -O3 -fsycl -fPIC -shared -std=c++20 -Wall $SDPA_FLAG -I"$DNNL/include" kernels/h3sycl.cpp \
       -L"$DNNL/lib" -ldnnl -Wl,-rpath,'$ORIGIN' -o dist/libh3sycl.so
  cp -L "$DNNL/lib/libdnnl.so.3" dist/        # shipped beside the library: hosts without a shared oneDNN load it too
}
engine() {
  echo "==> engine: h3d (in the container) and sycl-h3 (the host's command line, static) (cargo, release)"
  cargo build --release --locked --manifest-path engine/Cargo.toml -p h3d
  cargo build --release --locked --manifest-path engine/Cargo.toml -p sycl-h3 --target x86_64-unknown-linux-musl
  cp engine/target/release/h3d dist/
  cp engine/target/x86_64-unknown-linux-musl/release/sycl-h3 dist/
  rm -f dist/h3                                  # the name before the split
  rm -rf dist/tokenizer && cp -r tokenizer dist/tokenizer   # the text encoder's tokenizer files (/app/tokenizer)
}
wfe() {
  echo "==> wfe: TypeScript -> dist/wfe"
  NODE=$(command -v node) wfe/build.sh
  rm -rf dist/wfe && cp -r wfe/build dist/wfe
}
tools() {
  echo "==> tools: oneDNN probes (gemm_bench, sdpa_probe)"
  cp -L "$DNNL/lib/libdnnl.so.3" dist/
  for t in gemm_bench sdpa_probe; do
    icpx -O2 -fsycl -std=c++20 -I"$DNNL/include" kernels/$t.cpp -L"$DNNL/lib" -ldnnl -Wl,-rpath,'$ORIGIN' -o dist/$t
  done
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
    tools) tools ;;
    test) tests ;;
    *) echo "build.sh: unknown target '$what' (kernels, engine, wfe, tools, test)" >&2; exit 2 ;;
  esac
done
ls -la dist
