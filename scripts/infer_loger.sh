#!/usr/bin/env bash
# Run LOGER + FSM inference on a single image or a folder.
set -euo pipefail
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate loger
else
  source activate loger
fi
cd "$(dirname "$0")/.."
python predict_loger.py "$@"
