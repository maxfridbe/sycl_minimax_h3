#!/usr/bin/env bash
# exp10.sh - a run dump WITH keyframes for the Rust engine's keyframe check: the bakery prompt chained from exp8's
# clip the way the speech films chain - its last latent at frame 0 (--first-latent), its last second of sound at
# frame 0 (--first-audio, the audio keyframe), and its last 22 frames as a motion guide ending on the last frame
# (--guide-clip ...:22:-22). 384x288, 2 s, 8 steps, int8 on the SYCL linears, no LoRA.
X=/mnt/2TBSSD/minimaxh3/sycl-exp
cd $X; source ./card.sh
L=$X/out/exp10.log; : > $L
card_take "exp10: a run dump with keyframes"
I8=/models/kitchen/minimax_h3_fl2va_pruned_int8_convrot.safetensors
./run_h3x.sh i-kf -e H3X_PRELOAD=1 -e H3X_PREFETCH=8 -e H3X_SYCL=1 -e H3X_DUMP_RUN=/out/rundump-kf.safetensors -- \
  --prompt-file /work/prompts/bakery.txt --width 384 --height 288 --seconds 2 --seed 0 --steps 8 --no-compile --dit $I8 \
  --first-latent /out/i-run.lastlat.pt --first-audio /out/i-run.lastaud.pt --guide-clip /out/i-run.mp4:22:-22
tr '\r' '\n' < out/i-kf.log | grep -aE "run dump|first latent|first audio|motion guide|audio anchor|sampled|TOTAL|^exit|Error|Traceback" -A2 | cut -c1-300 >> $L
docker run --rm -v $X/out:/out --entrypoint chmod h3-xpu:3 644 /out/rundump-kf.safetensors /out/i-kf.mp4 2>/dev/null
card_release; echo DONE >> $L
