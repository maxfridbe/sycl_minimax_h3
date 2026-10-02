#!/usr/bin/env bash
# exp2.sh - the quality reference again (exp1's per-step dump crashed), then the attention-part benchmark
X=/mnt/2TBSSD/minimaxh3/sycl-exp
cd $X; source ./card.sh
while pgrep -f "exp1[.]sh" >/dev/null; do sleep 10; done
L=$X/out/exp2.log; : > $L
card_take "exp2: Q8 vs int8 reference clip, softmax bench"
echo "took the card at $(date +%H:%M), previous LLM mode $(cat .prev-mode)" >> $L
P="--prompt-file /work/prompts/bakery.txt --width 768 --height 576 --seconds 5 --seed 0"
Q8=/models/engines/minimax_h3_fl2va_pruned-Q8_0.gguf
I8=/models/kitchen/minimax_h3_fl2va_pruned_int8_convrot.safetensors
sum() { f=out/$1.log; echo "$1: $(grep -E "kitchen backends|DiT loaded|DiT weights on device|step [0-9]+/|sampled|decoded|TOTAL|^exit|Error|error" $f | tr '\n' ';' | cut -c1-900)" >> $L; }
./run_h3x.sh c-q8   -e H3X_DUMP_STEPS=/out/c-q8 -e H3X_PRELOAD=1 -- $P --steps 8 --dit $Q8 --no-compile;   sum c-q8
./run_h3x.sh c-int8 -e H3X_DUMP_STEPS=/out/c-int8 -e H3X_PRELOAD=1 -- $P --steps 8 --dit $I8 --no-compile; sum c-int8
echo "== attention parts" >> $L
timeout 900 docker run --rm --device /dev/dri -v $X:/exp h3-sycl-dev "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cd /exp && LD_LIBRARY_PATH=/opt/intel/oneapi/dnnl/2026.0/lib:\$LD_LIBRARY_PATH ./gemm_bench2 5 attn" >> $L 2>&1
card_release
echo "released the card at $(date +%H:%M)" >> $L
echo DONE >> $L
