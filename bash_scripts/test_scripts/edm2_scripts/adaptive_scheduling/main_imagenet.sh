#!/bin/bash
set -euo pipefail

SCRIPT_PATH="$1"
# bash bash_scripts/test_scripts/edm2_scripts/adaptive_scheduling/main_imagenet.sh bash_scripts/test_scripts/edm2_scripts/adaptive_scheduling/adaptive_scheduling_imagenet.sh

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "ERROR: script not found: $SCRIPT_PATH"
  exit 1
fi

PRESETS=("edm2-img512-s-fid" "edm2-img512-m-fid" "edm2-img512-l-fid")

for PRESET in "${PRESETS[@]}"; do
  echo "======================================="
  echo "PRESET=${PRESET}"
  echo "======================================="

  PRESET=$PRESET \
  bash "$SCRIPT_PATH"
done