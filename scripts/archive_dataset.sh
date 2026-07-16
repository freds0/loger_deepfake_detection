#!/usr/bin/env bash
# Package every file actually used in training into a single tar.bz2.
#
# "Used in training" is defined by data/manifests/*.csv (produced by
# scripts/pre_process.sh / preprocess_*.py): the union of their `path` column,
# plus the manifests themselves and the domain registry, so the archive can
# be extracted elsewhere and pointed at directly with data.manifest=...
#
# Some datasets (e.g. DF40) reference images in place under the original
# dataset tree rather than a copy under data/, so those paths are included
# too — this script only reads them, never modifies anything outside the
# repo.
#
# Usage:
#   scripts/archive_dataset.sh                          # -> data/processed_dataset_<timestamp>.tar.bz2
#   scripts/archive_dataset.sh /path/to/out.tar.bz2      # explicit output path
#   scripts/archive_dataset.sh --manifests ffpp,celebdf  # archive a subset of datasets
set -euo pipefail
cd "$(dirname "$0")/.."

MANIFEST_DIR="data/manifests"
ONLY=""
OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifests)
      ONLY="$2"
      shift 2
      ;;
    *)
      OUT="$1"
      shift
      ;;
  esac
done

shopt -s nullglob
if [[ -n "$ONLY" ]]; then
  manifests=()
  IFS=',' read -ra names <<< "$ONLY"
  for name in "${names[@]}"; do
    f="$MANIFEST_DIR/$name.csv"
    if [[ -f "$f" ]]; then
      manifests+=("$f")
    else
      echo "[warn] manifest not found, skipping: $f" >&2
    fi
  done
else
  manifests=("$MANIFEST_DIR"/*.csv)
fi

if [[ ${#manifests[@]} -eq 0 ]]; then
  echo "No manifests found under $MANIFEST_DIR — run scripts/pre_process.sh first." >&2
  exit 1
fi

[[ -z "$OUT" ]] && OUT="data/processed_dataset_$(date +%Y%m%d_%H%M%S).tar.bz2"
mkdir -p "$(dirname "$OUT")"

FILELIST="$(mktemp)"
MISSING="$(mktemp)"
trap 'rm -f "$FILELIST" "$MISSING"' EXIT

# Manifests + domain registry themselves, so the archive is self-describing.
printf '%s\n' "${manifests[@]}" > "$FILELIST"
[[ -f "$MANIFEST_DIR/domain_registry.json" ]] && echo "$MANIFEST_DIR/domain_registry.json" >> "$FILELIST"

# Every image path referenced by any selected manifest (dedup).
for m in "${manifests[@]}"; do
  tail -n +2 "$m" | cut -d, -f1
done >> "$FILELIST"
sort -u "$FILELIST" -o "$FILELIST"

# Drop paths that no longer exist on disk (manifest may be stale).
: > "$MISSING"
KEEP="$(mktemp)"
while IFS= read -r f; do
  if [[ -f "$f" ]]; then
    echo "$f" >> "$KEEP"
  else
    echo "$f" >> "$MISSING"
  fi
done < "$FILELIST"
mv "$KEEP" "$FILELIST"

if [[ -s "$MISSING" ]]; then
  n_missing=$(wc -l < "$MISSING")
  echo "[warn] $n_missing referenced file(s) missing on disk, skipped (showing up to 5):" >&2
  head -5 "$MISSING" >&2
fi

n_files=$(wc -l < "$FILELIST")
echo "Archiving $n_files files -> $OUT"
tar -cjf "$OUT" -T "$FILELIST"
echo "Done: $(du -h "$OUT" | cut -f1) -> $OUT"
