#!/usr/bin/env bash
# with-card.sh <what> -- <command ...>   take the shared card (card.sh), run the command, give the card back.
# For builds and engine runs on a box where another program normally holds the GPU.
source "$(dirname "$0")/card.sh"
what=$1; shift; [ "$1" = "--" ] && shift
card_take "$what"
"$@"; rc=$?
card_release
exit $rc
