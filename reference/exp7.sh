#!/usr/bin/env bash
# exp7.sh - the block dump the Rust engine is checked against: one denoiser call at a small shape (384x288, 2 s,
# 1 step), int8 checkpoint on comfy-kitchen's own int8 path, every intermediate of block 0 and the last block's output
# saved to out/blockdump.safetensors (reference/h3x.py, H3X_DUMP_BLOCK).
X=/mnt/2TBSSD/minimaxh3/sycl-exp
cd $X; source ./card.sh
L=$X/out/exp7.log; : > $L
card_take "exp7: block dump for the Rust engine's parity checks"
I8=/models/kitchen/minimax_h3_fl2va_pruned_int8_convrot.safetensors
./run_h3x.sh h-dump -e H3X_PRELOAD=1 -e H3X_PREFETCH=8 -e H3X_INT8_NATIVE=1 -e H3X_DUMP_BLOCK=/out/blockdump.safetensors -- \
  --prompt-file /work/prompts/bakery.txt --width 384 --height 288 --seconds 2 --seed 0 --steps 1 --no-compile --dit $I8
tr '\r' '\n' < out/h-dump.log | grep -aE "block dump|frames @|sampled|TOTAL|^exit|Error|Traceback" -A3 | cut -c1-200 >> $L
card_release; echo DONE >> $L
