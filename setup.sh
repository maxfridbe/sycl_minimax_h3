#!/usr/bin/env bash
# setup.sh - build the container image everything else runs in (SYCL compiler + oneDNN + Rust + node).
#   ./setup.sh                 build it (a no-op when it is up to date: the layers are cached)
#   BASE=<image> ./setup.sh    start from another base image that has the oneAPI compiler and Intel's apt repository
set -euo pipefail
source "$(dirname "$0")/container/common.sh"

args=()
[ -n "${BASE:-}" ] && args+=(--build-arg "BASE=$BASE")
echo "==> $CE build -t $IMAGE"
$CE build "${args[@]}" -t "$IMAGE" -f "$ROOT/container/Containerfile" "$ROOT/container"
echo "==> done. Next: ./build.sh"
