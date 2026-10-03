#!/usr/bin/env bash
# exp11.sh - a run dump with a reference audio (ref2va voice lock) for the Rust engine's check: the bakery prompt
# spoken "in the voice of <Audio 1>", <Audio 1> = the Goodnight Borg clip's sound (Picard). 384x288, 2 s, 8 steps,
# int8 on the SYCL linears, no LoRA. The prompt is encoded fresh (the <Audio 1> label is part of it).
X=/mnt/2TBSSD/minimaxh3/sycl-exp
cd $X; source ./card.sh
L=$X/out/exp11.log; : > $L
card_take "exp11: a run dump with reference audio"
I8=/models/kitchen/minimax_h3_fl2va_pruned_int8_convrot.safetensors
./run_h3x.sh i-voice -e H3X_PRELOAD=1 -e H3X_PREFETCH=8 -e H3X_SYCL=1 -e H3X_DUMP_RUN=/out/rundump-voice.safetensors -- \
  --prompt-file /work/prompts/bakery_voice.txt --width 384 --height 288 --seconds 2 --seed 0 --steps 8 --no-compile --dit $I8 \
  --ref-audio /out/g-borg5-sycl.mp4
tr '\r' '\n' < out/i-voice.log | grep -aE "run dump|ref|conditioning|sampled|TOTAL|^exit|Error|Traceback" -A2 | cut -c1-300 >> $L
docker run --rm -v $X/out:/out --entrypoint chmod h3-xpu:3 644 /out/rundump-voice.safetensors /out/i-voice.mp4 2>/dev/null
card_release; echo DONE >> $L
