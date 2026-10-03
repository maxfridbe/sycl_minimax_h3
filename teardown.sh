#!/usr/bin/env bash
# teardown.sh - stop the engine and remove what setup.sh and build.sh made.
#   ./teardown.sh            stop a running engine (gracefully: it finishes the kernel it is in)
#   ./teardown.sh --build    ... and remove dist/, the Rust target directory, the front end's build and the crate cache
#   ./teardown.sh --image    ... and remove the container image
#   ./teardown.sh --all      both
set -euo pipefail
source "$(dirname "$0")/container/common.sh"

build=0; image=0
for a in "$@"; do
  case $a in
    --build) build=1 ;;
    --image) image=1 ;;
    --all) build=1; image=1 ;;
    *) echo "usage: ./teardown.sh [--build] [--image] [--all]" >&2; exit 2 ;;
  esac
done

# the services first (the daemon unloads its engines at a block boundary; the web service holds nothing)
[ -x "$ROOT/dist/h3-sycl" ] && "$ROOT/dist/h3-sycl" stop --all
if $CE container inspect h3-engine >/dev/null 2>&1; then
  # stop, with time to finish: a GPU process killed inside a kernel can wedge the xe driver
  echo "==> stopping the engine (up to 120 s)"
  $CE stop -t 120 h3-engine >/dev/null
fi
if [ $build = 1 ]; then
  echo "==> removing build output"
  rm -rf "$ROOT/dist" "$ROOT/engine/target" "$ROOT/wfe/build" "$ROOT/.cache"
fi
if [ $image = 1 ] && have_image; then
  echo "==> removing the image $IMAGE"
  $CE rmi "$IMAGE" >/dev/null
fi
echo "==> done"
