#!/usr/bin/env bash
# card.sh - take / release the B70 for an experiment. Sourced by the experiment scripts.
#   card_take <what>   waits for /mnt/2TBSSD/CARD_LOCK to be absent, writes it, remembers the front end's LLM mode,
#                      sets it to none and waits for the card to be empty
#   card_release       restores that LLM mode and removes the lock
# The lock is shared with the Strata session on this machine (owner line: "<session> <epoch> <what>").
LOCK=/mnt/2TBSSD/CARD_LOCK
ME=minimax_sycl
rpc() { curl -s -m 10 -X POST "http://localhost:8090/rpc/$1" -H "content-type: application/json" -d "$2"; }
card_take() {
  while [ -e $LOCK ] && ! grep -q "^$ME " $LOCK; do sleep 20; done
  echo "$ME $(date +%s) $1" > $LOCK
  PREV=$(rpc llm.mode '{}' | python3 -c 'import sys,json; print(json.load(sys.stdin)["result"]["mode"])' 2>/dev/null || echo none)
  echo "$PREV" > /mnt/2TBSSD/minimaxh3/sycl-exp/.prev-mode
  rpc llm.mode '{"mode":"none"}' >/dev/null
  # wait until the card is really empty - never start beside another model (the xe driver has no out-of-memory: a
  # second model evicts VRAM into host RAM and hangs the box). No timeout: a front end that is mid-start finishes
  # the start first and only then honours "none".
  while true; do
    u=$(python3 -c 'import json; print(json.load(open("/run/gpustat.json"))["vram_used_mb"])' 2>/dev/null || echo 99999)
    s=$(pgrep -fc "server(_intel)?[.]py --engine" || true)
    c=$(docker ps --format '{{.Names}}' | grep -c 'strata-sycl\|vllm\|h3gen' || true)
    [ "$u" -lt 2000 ] && [ "${s:-0}" = 0 ] && [ "${c:-0}" = 0 ] && break
    sleep 5
  done
}
card_release() {
  PREV=$(cat /mnt/2TBSSD/minimaxh3/sycl-exp/.prev-mode 2>/dev/null || echo none)
  rpc llm.mode "{\"mode\":\"$PREV\"}" >/dev/null
  rm -f $LOCK
}
