#!/usr/bin/env bash
# build.sh - build the engine inside the container. Output goes to dist/:
#     dist/libh3sycl.so (+ libdnnl.so.3)   the SYCL kernels      (kernels/)
#     dist/h3                              the Rust engine       (engine/)
#     dist/wfe/                            the web front end     (wfe/)
#   ./build.sh                  all three
#   ./build.sh kernels|engine|wfe ...     only those
#   ./build.sh tools            the oneDNN probes (dist/gemm_bench, dist/sdpa_probe)
#   ./build.sh test             the Rust tests and lints
# Do not build while a model is running on a small-memory box: the SYCL compile alone takes several GB of RAM.
set -euo pipefail
source "$(dirname "$0")/container/common.sh"
need_image

mkdir -p "$ROOT/dist" "$ROOT/.cache/cargo"
exec $CE run --rm "${AS_USER[@]}" -v "$ROOT:/src" -w /src -e CARGO_HOME=/src/.cache/cargo ${H3_DNNL:+-e H3_DNNL=$H3_DNNL} \
  "$IMAGE" bash /src/container/build-inside.sh "$@"
