#!/usr/bin/env bash
# exp4.sh - the weight load from a COLD page cache: today's lazy load vs 8-thread read-ahead (Q8_0 and int8), 3 steps
X=/mnt/2TBSSD/minimaxh3/sycl-exp
cd $X; source ./card.sh
L=$X/out/exp4.log; : > $L
card_take "exp4: cold weight load, with and without read-ahead"
P="--prompt-file /work/prompts/bakery.txt --width 768 --height 576 --seconds 5 --seed 0 --steps 3 --no-compile"
Q8=/models/engines/minimax_h3_fl2va_pruned-Q8_0.gguf
I8=/models/kitchen/minimax_h3_fl2va_pruned_int8_convrot.safetensors
sum() { f=out/$1.log; echo "$1: $(grep -E "prefetch|DiT weights on device|sampled|TOTAL|^exit|Error" $f | cut -c1-110 | tr '\n' ';')" >> $L; }
cold() { sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || echo "  (could not drop the page cache)" >> $L; }
cold; ./run_h3x.sh d-q8-cold -e H3X_PRELOAD=1 -- $P --dit $Q8;                            sum d-q8-cold
cold; ./run_h3x.sh d-q8-prefetch -e H3X_PRELOAD=1 -e H3X_PREFETCH=8 -- $P --dit $Q8;      sum d-q8-prefetch
cold; ./run_h3x.sh d-int8-prefetch -e H3X_PRELOAD=1 -e H3X_PREFETCH=8 -- $P --dit $I8;    sum d-int8-prefetch
card_release; echo DONE >> $L
