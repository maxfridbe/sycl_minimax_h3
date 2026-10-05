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
# the SageAttention library (the default attention; H3S_ATTN=onednn turns it off): ARK's kernel on sycl-tla, with the flags sycl-tla wants. The
# device it is compiled for: H3_SAGE_DEVICE (bmg-g31 = the B70; bmg-g21 = B580/B60).
sage() {
  local TLA=${H3_SYCL_TLA:-/opt/sycl-tla} ARK=${H3_ARK:-/opt/ark/auto_round_kernel}
  if [ ! -f "$ARK/wrapper/include/sycl_tla_sdpa.hpp" ]; then
    echo "==> sage: skipped (no sycl-tla / ARK headers in this image: ./setup.sh)"; return 0
  fi
  echo "==> sage: libh3sage.so (icpx, sycl-tla, device ${H3_SAGE_DEVICE:-bmg-g31}; a few minutes)"
  icpx -O3 -fsycl -fPIC -shared -std=c++17 -fno-sycl-instrument-device-code -w \
       -DARK_XPU=1 -DARK_SYCL_TLA=1 -DCUTLASS_ENABLE_SYCL=1 -DSYCL_INTEL_TARGET=1 \
       -isystem "$TLA/include" -isystem "$TLA/applications" -isystem "$TLA/tools/util/include" \
       -isystem "$TLA/examples/common" -isystem "$TLA/examples/06_bmg_flash_attention" \
       -I"$ARK/wrapper/include" -I"$ARK/bestla" kernels/sage.cpp \
       -fsycl-targets=spir64 -Xs "-device ${H3_SAGE_DEVICE:-bmg-g31}" -Xspirv-translator \
       -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
       -o dist/libh3sage.so
}
engine() {
  echo "==> engine: h3d (in the container) and sycl-h3 (the host's command line, static) (cargo, release)"
  cargo build --release --locked --manifest-path engine/Cargo.toml -p h3d
  cargo build --release --locked --manifest-path engine/Cargo.toml -p sycl-h3 --target x86_64-unknown-linux-musl
  # beside, then renamed over: a running daemon or studio keeps its old file (a plain cp fails "Text file busy")
  cp engine/target/release/h3d dist/h3d.new && mv -f dist/h3d.new dist/h3d
  echo "    (a running engine daemon keeps the old h3d and cannot start workers from it: sycl-h3 stop, then sycl-h3 start)"
  cp engine/target/x86_64-unknown-linux-musl/release/sycl-h3 dist/sycl-h3.new && mv -f dist/sycl-h3.new dist/sycl-h3
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

[ $# -eq 0 ] && set -- kernels sage engine wfe
for what in "$@"; do
  case $what in
    kernels) kernels ;;
    sage) sage ;;
    engine) engine ;;
    wfe) wfe ;;
    tools) tools ;;
    test) tests ;;
    *) echo "build.sh: unknown target '$what' (kernels, sage, engine, wfe, tools, test)" >&2; exit 2 ;;
  esac
done
ls -la dist
