#!/usr/bin/env bash
# exp5.sh [kitchen] [native] [sycl] [syclsync] - the int8 denoiser with its linears on libh3sycl (comfy-kitchen backend "sycl")
# vs comfy-kitchen's own path. kitchen = ComfyUI's default on an Intel GPU (int8 weights turned back into bf16 per call),
# native = comfy-kitchen's PyTorch int8_linear, sycl = ours. Same clip as exp2's c-int8 (bakery, 768x576, 5 s, 8 steps, seed 0), no torch.compile,
# step dumps for compare.py.
X=/mnt/2TBSSD/minimaxh3/sycl-exp
cd $X; source ./card.sh
L=$X/out/exp5.log; : > $L
card_take "exp5: int8 denoiser on the sycl backend"
P="--prompt-file /work/prompts/bakery.txt --width 768 --height 576 --seconds 5 --seed 0 --steps 8 --no-compile"
I8=/models/kitchen/minimax_h3_fl2va_pruned_int8_convrot.safetensors
E="-e H3X_PRELOAD=1 -e H3X_PREFETCH=8"
sum() { echo "$1: $(tr '\r' '\n' < out/$1.log | grep -aE "sycl backend|int8 linears|Native ops|kitchen backends|DiT weights on device|sampled| 8/8 |TOTAL|^exit|Error" | cut -c1-140 | tr '\n' ';')" >> $L; }
for t in ${@:-kitchen sycl}; do
  case $t in
    kitchen)  ./run_h3x.sh f-int8-kitchen $E -e H3X_DUMP_STEPS=/out/f-int8-kitchen -- $P --dit $I8; sum f-int8-kitchen;;
    sycl)     ./run_h3x.sh f-int8-sycl $E -e H3X_SYCL=1 -e H3X_DUMP_STEPS=/out/f-int8-sycl -- $P --dit $I8; sum f-int8-sycl;;
    native)   ./run_h3x.sh f-int8-native $E -e H3X_INT8_NATIVE=1 -e H3X_DUMP_STEPS=/out/f-int8-native -- $P --dit $I8; sum f-int8-native;;
    profile)  ./run_h3x.sh f-int8-profile $E -e H3X_SYCL=1 -e H3X_SYCL_SYNC=1 -e H3X_PROFILE=1 -- ${P/--steps 8/--steps 3} --dit $I8; sum f-int8-profile
              tr '\r' '\n' < out/f-int8-profile.log | grep -a -A16 "profile (seconds" >> $L;;
    syclsync) ./run_h3x.sh f-int8-syclsync $E -e H3X_SYCL=1 -e H3X_SYCL_SYNC=1 -- $P --dit $I8; sum f-int8-syclsync;;
  esac
done
card_release; echo DONE >> $L
