#!/usr/bin/env bash
# Train LOGER + FSM. Extra args are forwarded to Hydra.
set -euo pipefail
#if command -v conda >/dev/null 2>&1; then
#  eval "$(conda shell.bash hook)"
#  conda activate loger
#else
#  source activate loger
#fi
cd "$(dirname "$0")/.."
python train_loger.py "$@" \
	--config-name loger_fsm_ntire \
	data.root=data/NTIRE-RobustAIGenDetection-train \
	trainer.max_epochs=100
