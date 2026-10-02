#!/usr/bin/env bash
# exp6.sh [prod] [sycl] - the "Goodnight Borg" clip (Picard narration, prompts/borg_goodnight_5s.txt) with the production
# recipe: 768x576, realism LoRA 0.5, 8 steps, latent upscale 1.5. Clip length from SECONDS_CLIP (default 5).
#   prod = today's path: Q8_0 GGUF denoiser, torch.compile, lazy weight load
#   sycl = int8 denoiser, linears on libh3sycl, no compile, weights read ahead and loaded before step 1
X=/mnt/2TBSSD/minimaxh3/sycl-exp
cd $X; source ./card.sh
S=${SECONDS_CLIP:-5}; T=borg${S}
L=$X/out/exp6-$T.log; : > $L
card_take "exp6: goodnight borg ${S}s, production recipe"
P="--prompt-file /work/prompts/borg_goodnight_5s.txt --width 768 --height 576 --seconds $S --seed 0 --steps 8 --upscale 1.5 --lora /models/loras/h3-realism-people-t2v-i2v-r2v.safetensors:0.5"
Q8=/models/engines/minimax_h3_fl2va_pruned-Q8_0.gguf
I8=/models/kitchen/minimax_h3_fl2va_pruned_int8_convrot.safetensors
sum() { { echo "$1:"; tr '\r' '\n' < out/$1.log | grep -aE "frames @|conditioning|engine :|lora|sycl backend|int8 linears|DiT weights on device|step [0-9]+/8|sampled|latent upscaled|AUDIO decoded|video decoded|muxed|TOTAL|^exit|Error|Traceback" | grep -av "it/s\|s/it" | cut -c1-120; } >> $L; }
for t in ${@:-prod sycl}; do
  case $t in
    prod) ./run_h3x.sh g-$T-prod -e H3X_DUMP_STEPS=/out/g-$T-prod -- $P --dit $Q8; sum g-$T-prod;;
    sycl) ./run_h3x.sh g-$T-sycl -e H3X_PRELOAD=1 -e H3X_PREFETCH=8 -e H3X_SYCL=1 -e H3X_DUMP_STEPS=/out/g-$T-sycl -- $P --no-compile --dit $I8; sum g-$T-sycl;;
  esac
done
card_release; echo DONE >> $L
