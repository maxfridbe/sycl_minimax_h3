#!/usr/bin/env bash
# exp9.sh - the Rust engine's latents (h3d denoise --out, from exp8's run dump) through the reference's decoders:
# the first clip whose whole denoise ran in the Rust/SYCL engine. out/i-rust.mp4, beside exp8's out/i-run.mp4.
X=/mnt/2TBSSD/minimaxh3/sycl-exp; B=/mnt/2TBSSD/minimaxh3
cd $X; source ./card.sh
L=$X/out/exp9.log; : > $L
cp $B/h3-engine/out/rust-latents.safetensors $X/out/ || { echo "no rust-latents.safetensors" >> $L; exit 1; }
card_take "exp9: decode the Rust engine's latents"
RG=$(getent group render|cut -d: -f3); VG=$(getent group video|cut -d: -f3)
docker run --rm --name h3exp --oom-score-adj 1000 --device /dev/dri -v /dev/dri:/dev/dri \
  --group-add $RG --group-add $VG --ipc=host --shm-size 8g --stop-timeout 120 \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -v $B/repo/vendor/ComfyUI:/comfy:ro -v $B/models:/models:ro -v $X/out:/out -v $X:/work:ro -v $B/cache:/cache \
  -e SYCL_CACHE_PERSISTENT=1 -e SYCL_CACHE_DIR=/cache/sycl \
  --entrypoint python3 h3-xpu:3 -u /work/h3x.py decode --latents /out/rust-latents.safetensors --out /out/i-rust.mp4 \
  > $X/out/i-rust.log 2>&1
echo "exit $?" >> $X/out/i-rust.log
tr '\r' '\n' < out/i-rust.log | grep -aE "latent:|decode total|OUT|^exit|Error|Traceback" -A2 | cut -c1-300 >> $L
card_release; echo DONE >> $L
