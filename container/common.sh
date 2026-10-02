# Sourced by setup.sh, build.sh, run.sh and teardown.sh: which container tool, which image, how to start it.
#
#   H3_CONTAINER_ENGINE   podman (default when installed) or docker
#   H3_IMAGE              the image's name (default h3-build)

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -z "${H3_CONTAINER_ENGINE:-}" ]; then
  if command -v podman >/dev/null 2>&1; then H3_CONTAINER_ENGINE=podman
  elif command -v docker >/dev/null 2>&1; then H3_CONTAINER_ENGINE=docker
  else echo "need podman (or docker) on PATH" >&2; exit 1; fi
fi
CE=$H3_CONTAINER_ENGINE
IMAGE=${H3_IMAGE:-h3-build}

# Files the container writes into the mounted tree must belong to the caller. Rootless podman maps the container's
# root to the caller already; docker needs the caller's ids.
if [ "$CE" = podman ]; then
  AS_USER=(--security-opt label=disable)
  # the GPU's device node belongs to the host's render group; only the crun runtime can hand a rootless container
  # the caller's groups (keep-groups)
  command -v crun >/dev/null 2>&1 || echo "note: crun is not installed; rootless podman cannot reach /dev/dri without it" >&2
  GPU=(--runtime crun --device /dev/dri --group-add keep-groups)
else
  AS_USER=(--user "$(id -u):$(id -g)" -e HOME=/tmp)
  GPU=(--device /dev/dri)
  for g in render video; do
    gid=$(getent group $g | cut -d: -f3); [ -n "$gid" ] && GPU+=(--group-add "$gid")
  done
fi

have_image() { $CE image inspect "$IMAGE" >/dev/null 2>&1; }
need_image() { have_image || { echo "the image '$IMAGE' is not built yet: run ./setup.sh" >&2; exit 1; }; }
