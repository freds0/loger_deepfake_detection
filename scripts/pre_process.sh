#!/usr/bin/env bash
# Run every dataset preprocessing script, building the manifests consumed by
# configs/data/combined.yaml. Each dataset is independent: a failure in one
# is reported and skipped rather than aborting the rest.
#
# Usage:
#   scripts/pre_process.sh                 # run all datasets, full extraction
#   scripts/pre_process.sh --dry-run        # tiny per-dataset sample, fast sanity check
#   scripts/pre_process.sh --only ffpp celebdf   # run a subset (see DATASETS below)
set -uo pipefail

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate loger
else
  source activate loger
fi
cd "$(dirname "$0")/.."

DATASETS=(ffpp celebdf sdfvd comprehensive df40 dfbench)
SELECTED=()
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
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
  SELECTED=("${DATASETS[@]}")
fi

run_ffpp() {
  local extra=()
  [[ $DRY_RUN -eq 1 ]] && extra=(--limit-videos 2 --frames-train 3 --frames-val 2 --frames-test 2)
  python scripts/preprocess_ffpp.py \
    --video-root "/home/ubuntu/fred-experiments/DATASETS/DeepFake/FaceForensics++_C23" \
    "${extra[@]}"
}

run_celebdf() {
  local extra=()
  [[ $DRY_RUN -eq 1 ]] && extra=(--limit-videos 2 --frames-train 3 --frames-val 2 --frames-test 2)
  python scripts/preprocess_celebdf.py "${extra[@]}"
}

run_sdfvd() {
  local extra=()
  [[ $DRY_RUN -eq 1 ]] && extra=(--limit-videos 2 --frames-train 3 --frames-val 2 --frames-test 2)
  python scripts/preprocess_sdfvd.py "${extra[@]}"
}

run_comprehensive() {
  local extra=()
  [[ $DRY_RUN -eq 1 ]] && extra=(--limit 5)
  python scripts/preprocess_comprehensive.py "${extra[@]}"
}

run_df40() {
  local extra=()
  [[ $DRY_RUN -eq 1 ]] && extra=(--limit-per-method 2)
  python scripts/preprocess_df40.py "${extra[@]}"
}

run_dfbench() {
  local extra=()
  [[ $DRY_RUN -eq 1 ]] && extra=(--limit-per-zip 50)
  python scripts/preprocess_dfbench.py "${extra[@]}"
}

FAILED=()
for name in "${SELECTED[@]}"; do
  echo
  echo "=== [$name] preprocessing ==="
  if ! "run_${name}"; then
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
    echo "  $name: ok -> data/manifests/$name.csv"
  fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  exit 1
fi
