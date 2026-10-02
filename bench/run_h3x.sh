#!/usr/bin/env bash
# run_h3x.sh <tag> [docker -e flags ...] -- <h3x.py gen args ...>
# One experiment clip through bench/h3x.py in the production image, in the foreground, log to out/<tag>.log.
# Same mounts and caches as the production launcher, but its own /out, so production's caches are not touched.
B=/mnt/2TBSSD/minimaxh3; X=$B/sycl-exp
tag=$1; shift
envs=()
while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
RG=$(getent group render|cut -d: -f3); VG=$(getent group video|cut -d: -f3)
docker rm -f h3exp >/dev/null 2>&1
docker run --rm --name h3exp --oom-score-adj 1000 --device /dev/dri -v /dev/dri:/dev/dri \
  --group-add $RG --group-add $VG --ipc=host --shm-size 8g --stop-timeout 120 \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 "${envs[@]}" \
  -v $B/repo/vendor/ComfyUI:/comfy:ro \
  -v $B/repo/vendor/ComfyUI/custom_nodes/ComfyUI-GGUF:/pkgs/comfyui_gguf:ro \
  -v $B/models:/models:ro -v $X/out:/out -v $X:/work:ro -v $B/cache:/cache \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e SYCL_CACHE_PERSISTENT=1 -e SYCL_CACHE_DIR=/cache/sycl \
  --entrypoint python3 h3-xpu:3 -u /work/h3x.py gen "$@" --out /out/$tag.mp4 > $X/out/$tag.log 2>&1
echo "exit $?" >> $X/out/$tag.log
