#!/usr/bin/env bash
# run.sh - one-shot developer checks on the GPU, inside the container (h3d device, check-linear, bench-blocks ...).
# Day-to-day use goes through ./sycl-h3 (the services); this is for measuring and debugging the engine itself.
#   ./run.sh device
#   ./run.sh info /models/<checkpoint>.safetensors
#   ./run.sh load /models/<checkpoint>.safetensors
#   ./run.sh check-linear /models/<checkpoint>.safetensors
#
#   H3_BIN      another program from dist/ instead of h3d (the probes: gemm_bench, sdpa_probe)
#   H3_MODELS   the host directory with the checkpoints, seen as /models (read-only)
#   H3_OUT      the host directory for results, seen as /out (default ./out)
#
# One model per GPU: the xe driver has no out-of-memory error, so starting this beside another program that holds the
# card can stall the machine. Stop it with ./teardown.sh (which waits), never with a kill.
set -euo pipefail
source "$(dirname "$0")/container/common.sh"
need_image
[ -x "$ROOT/dist/${H3_BIN:-h3d}" ] || { echo "dist/${H3_BIN:-h3d} is not built yet: run ./build.sh" >&2; exit 1; }

mounts=(-v "$ROOT/dist:/app:ro")
[ -n "${H3_MODELS:-}" ] && mounts+=(-v "$H3_MODELS:/models:ro")
OUT=${H3_OUT:-$ROOT/out}; mkdir -p "$OUT"; mounts+=(-v "$OUT:/out")

exec $CE run --rm --name h3-engine --stop-timeout 120 "${AS_USER[@]}" "${GPU[@]}" "${mounts[@]}" \
  -e H3SYCL_LIB=/app/libh3sycl.so ${H3S_MEM_FRACTION:+-e H3S_MEM_FRACTION=$H3S_MEM_FRACTION} \
  ${ONEDNN_VERBOSE:+-e ONEDNN_VERBOSE=$ONEDNN_VERBOSE} ${H3S_PROFILE:+-e H3S_PROFILE=1} ${H3S_ATTN_TABLE_MB:+-e H3S_ATTN_TABLE_MB=$H3S_ATTN_TABLE_MB} ${H3S_ATTN_ROWS:+-e H3S_ATTN_ROWS=$H3S_ATTN_ROWS} -e H3_BIN="${H3_BIN:-h3d}" \
  "$IMAGE" bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec "/app/$H3_BIN" "$@"' h3d "$@"
