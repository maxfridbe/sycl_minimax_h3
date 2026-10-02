#!/usr/bin/env bash
X=/mnt/2TBSSD/minimaxh3/sycl-exp
cd $X; source ./card.sh
L=$X/out/exp3.log; : > $L
card_take "exp3: host->device copy rates"
B=/mnt/2TBSSD/minimaxh3; RG=$(getent group render|cut -d: -f3); VG=$(getent group video|cut -d: -f3)
docker run --rm --name h3exp --oom-score-adj 1000 --device /dev/dri -v /dev/dri:/dev/dri --group-add $RG --group-add $VG \
  --ipc=host -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 -v $B/models:/models:ro -v $X:/work:ro \
  --entrypoint python3 h3-xpu:3 -u /work/h2d_bench.py 2>&1 | grep -v Warn >> $L
card_release; echo DONE >> $L
