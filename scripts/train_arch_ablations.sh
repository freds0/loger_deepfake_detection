#!/usr/bin/env bash
# Train every LOGER+FSM architecture variation on the full combined dataset
# (configs/data/combined.yaml -- all 8 preprocessed datasets: FF++, Celeb-DF-v2,
# SDFVD, Comprehensive, DF40, DFBench, HydraFake, HiDF), so they can be
# compared under an identical data/step budget.
#
# Each variant is `--config-name loger_fsm_combined` plus the same override(s)
# used by its configs/ntire_v2_*.yaml counterpart (see docs/ablations_ntire.md
# for the analogous NTIRE-only protocol this mirrors, Group 1). resume is
# always disabled per-run so variants never accidentally resume each other's
# checkpoint (they'd otherwise share resume.dir: outputs/loger_fsm_combined).
#
# Requires the manifests under data/manifests/*.csv to exist already --
# run scripts/pre_process.sh first (a subset is fine: records_from_manifest
# skips any manifest that hasn't been generated yet, with a warning).
#
# Usage:
#   scripts/train_arch_ablations.sh                       # all 12 variants, full trainer.max_steps (30000)
#   scripts/train_arch_ablations.sh --steps 8000           # override the step budget for every variant
#   scripts/train_arch_ablations.sh --only base dinov2     # run a subset (see VARIANT_NAMES below for names)
#   scripts/train_arch_ablations.sh --dry-run              # ~10-step smoke run per variant, fast sanity check
set -uo pipefail

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate loger
else
  source activate loger
fi
cd "$(dirname "$0")/.."

VARIANT_NAMES=(base topk05 topk20 topk40 topk100 learnfusion nofused head512 head1024 dinov2 dinov3 siglip2large)

# Echoes the override args for one variant (relative to loger_fsm_combined's
# defaults), or nothing for "base". Unknown names print to stderr and fail.
overrides_for() {
  case "$1" in
    base)          ;;
    topk05)        echo "model.topk_ratio=0.05" ;;
    topk20)        echo "model.topk_ratio=0.2" ;;
    topk40)        echo "model.topk_ratio=0.4" ;;
    topk100)       echo "model.topk_ratio=1.0" ;;
    learnfusion)   echo "model.learnable_fusion=true" ;;
    nofused)       echo "loss.fused_weight=0.0" ;;
    head512)       echo "model.head_hidden_dim=512" ;;
    head1024)      echo "model.head_hidden_dim=1024" ;;
    dinov2)        echo "backbone=dinov2_base" ;;
    dinov3)        echo "backbone=dinov3_base" ;;   # needs transformers>=4.56
    siglip2large)  echo "backbone=siglip2_large peft=loger_lora" ;;
    *) echo "Unknown variant: $1 (see VARIANT_NAMES)" >&2; return 1 ;;
  esac
}

SELECTED=()
STEPS=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --only)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        SELECTED+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ${#SELECTED[@]} -eq 0 ]]; then
  SELECTED=("${VARIANT_NAMES[@]}")
fi

FAILED=()
for name in "${SELECTED[@]}"; do
  echo
  echo "=== [$name] training (full combined dataset) ==="

  overrides_str="$(overrides_for "$name")"
  if [[ $? -ne 0 ]]; then
    FAILED+=("$name")
    continue
  fi
  extra=()
  [[ -n "$overrides_str" ]] && read -r -a extra <<< "$overrides_str"

  run_overrides=(resume.enabled=false "${extra[@]}")
  [[ -n "$STEPS" ]] && run_overrides+=("trainer.max_steps=$STEPS")
  if [[ $DRY_RUN -eq 1 ]]; then
    run_overrides+=(
      trainer.max_steps=10 trainer.val_check_interval=10 trainer.limit_val_batches=2
      'data.max_records_per_split={train: 64, val: 32, test: 32}'
    )
  fi

  if ! python train_loger.py --config-name loger_fsm_combined "${run_overrides[@]}"; then
    echo "[$name] FAILED" >&2
    FAILED+=("$name")
  fi
done

echo
echo "=== Summary ==="
for name in "${SELECTED[@]}"; do
  if printf '%s\n' "${FAILED[@]:-}" | grep -qx "$name"; then
    echo "  $name: FAILED"
  else
    echo "  $name: ok -> outputs/loger_fsm_combined/<date>/<time>/ (see hydra run dir printed above)"
  fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  exit 1
fi
