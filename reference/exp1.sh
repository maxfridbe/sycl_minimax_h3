#!/usr/bin/env bash
# exp1.sh - Phase 0, first card session:
#   A. gemm_bench: int8 against 16-bit float matrix multiply at the DiT's shapes (oneDNN)
#   B. the first-step extra, split: default / weights preloaded / preloaded + no torch.compile  (768x576, 5 s, 3 steps)
#   C. quality reference: the same clip (768x576, 5 s, 8 steps, seed 0) from the GGUF Q8_0 denoiser and from upstream's
#      int8_convrot denoiser on the plain-PyTorch ("eager") kitchen backend
X=/mnt/2TBSSD/minimaxh3/sycl-exp
cd $X; source ./card.sh
L=$X/out/exp1.log; : > $L
card_take "exp1: gemm bench, first-step split, Q8 vs int8 reference clip"
echo "took the card at $(date +%H:%M), previous LLM mode $(cat .prev-mode)" >> $L
echo "== A. gemm_bench" >> $L
timeout 900 docker run --rm --device /dev/dri -v $X:/exp h3-sycl-dev "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cd /exp && LD_LIBRARY_PATH=/opt/intel/oneapi/dnnl/2026.0/lib:\$LD_LIBRARY_PATH ./gemm_bench 5" >> $L 2>&1
P="--prompt-file /work/prompts/bakery.txt --width 768 --height 576 --seconds 5 --seed 0"
Q8=/models/engines/minimax_h3_fl2va_pruned-Q8_0.gguf
I8=/models/kitchen/minimax_h3_fl2va_pruned_int8_convrot.safetensors
sum() { f=out/$1.log; echo "$1: $(grep -E "conditioning|DiT loaded|DiT weights on device|step [0-9]+/|sampled|decoded|TOTAL|^exit|Error|error" $f | tr '\n' ';' | cut -c1-700)" >> $L; }
echo "== B. first-step split (3 steps)" >> $L
./run_h3x.sh b-default -- $P --steps 3 --dit $Q8;                         sum b-default
./run_h3x.sh b-preload -e H3X_PRELOAD=1 -- $P --steps 3 --dit $Q8;        sum b-preload
./run_h3x.sh b-preload-nocompile -e H3X_PRELOAD=1 -- $P --steps 3 --dit $Q8 --no-compile; sum b-preload-nocompile
echo "== C. quality reference (8 steps)" >> $L
./run_h3x.sh c-q8   -e H3X_DUMP_STEPS=/out/c-q8   -- $P --steps 8 --dit $Q8;   sum c-q8
./run_h3x.sh c-int8 -e H3X_DUMP_STEPS=/out/c-int8 -e H3X_PRELOAD=1 -- $P --steps 8 --dit $I8 --no-compile; sum c-int8
card_release
echo "released the card at $(date +%H:%M)" >> $L
echo DONE >> $L
