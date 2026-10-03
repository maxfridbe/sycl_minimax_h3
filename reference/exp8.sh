#!/usr/bin/env bash
# exp8.sh - a whole sampling run dumped for the Rust engine's denoise check (reference/h3x.py, H3X_DUMP_RUN):
# bakery prompt, 384x288, 2 s, 8 steps, seed 0, the int8 checkpoint on the SYCL linears, no LoRA; decoded to
# out/i-run.mp4 as the reference clip.
X=/mnt/2TBSSD/minimaxh3/sycl-exp
cd $X; source ./card.sh
L=$X/out/exp8.log; : > $L
card_take "exp8: a whole sampling run dumped for the Rust engine"
I8=/models/kitchen/minimax_h3_fl2va_pruned_int8_convrot.safetensors
./run_h3x.sh i-run -e H3X_PRELOAD=1 -e H3X_PREFETCH=8 -e H3X_SYCL=1 -e H3X_DUMP_RUN=/out/rundump.safetensors -- \
  --prompt-file /work/prompts/bakery.txt --width 384 --height 288 --seconds 2 --seed 0 --steps 8 --no-compile --dit $I8
tr '\r' '\n' < out/i-run.log | grep -aE "run dump|frames @|sampled|TOTAL|^exit|Error|Traceback" -A2 | cut -c1-300 >> $L
card_release; echo DONE >> $L
