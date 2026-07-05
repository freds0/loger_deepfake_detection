#!/usr/bin/env bash
# Train OSDFD. Extra args are forwarded to Hydra.
set -euo pipefail
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate loger
else
  source activate loger
fi
cd "$(dirname "$0")/.."
python train.py "$@"
