#!/usr/bin/env bash
# Upload processed dataset manifests + their referenced files to
# s3://ermis-datasets/deepfake_dataset.
#
# Credentials are never hardcoded here: export them before running, e.g.
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#   scripts/upload_to_s3.sh
#
# By default only the newest datasets (hydrafake, hidf) are uploaded, one
# prefix per dataset: s3://ermis-datasets/deepfake_dataset/<name>/...
# Pass --manifests to pick a different subset (comma-separated, matching
# data/manifests/<name>.csv), or --all to upload every processed dataset.
#
# Uses `aws s3 sync` (not per-file cp) so already-uploaded files are skipped
# on reruns (size/mtime comparison) instead of being sent again — safe to
# re-run after adding new images, and this is what makes re-running this
# script cheap.
#
# Sync roots are chosen per manifest path:
#   - paths under data/ (built by our own preprocess_*.py into a directory
#     dedicated to that one dataset, e.g. data/hydrafake_frames) are synced
#     as a single whole directory: it holds nothing but that dataset's files,
#     so one sync call is both exact and fast.
#   - paths outside data/ (DF40, HiDF: referenced in place under the
#     read-only DATASETS/DeepFake tree, only ever read here, never modified)
#     sync per unique immediate parent directory instead, since those
#     external roots also contain sibling files/archives that are NOT part
#     of the manifest (e.g. HiDF's Real-img.zip / Fake-img.zip) and must not
#     be uploaded. This is exact for every dataset added so far (each
#     referenced folder is unambiguously real-only or fake-only, see
#     scripts/preprocess_*.py) but can mean many sync calls for datasets with
#     deep per-clip nesting (e.g. df40) — fine for occasional/explicit use,
#     just not fast.
#
# Usage:
#   scripts/upload_to_s3.sh                       # hydrafake + hidf (default)
#   scripts/upload_to_s3.sh --manifests df40,ffpp  # explicit subset
#   scripts/upload_to_s3.sh --all                  # every dataset with a manifest
#   scripts/upload_to_s3.sh --dry-run              # show what would be uploaded, no upload
set -euo pipefail
cd "$(dirname "$0")/.."

BUCKET="s3://ermis-datasets/deepfake_dataset"
MANIFEST_DIR="data/manifests"
ONLY="hydrafake,hidf"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifests)
      ONLY="$2"
      shift 2
      ;;
    --all)
      ONLY=""
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI not found — install it first (see https://docs.aws.amazon.com/cli/)." >&2
  exit 1
fi

if [[ $DRY_RUN -eq 0 && ( -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ) ]]; then
  echo "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set in the environment." >&2
  echo "Export them (or run 'aws configure') before uploading." >&2
  exit 1
fi

shopt -s nullglob
if [[ -n "$ONLY" ]]; then
  IFS=',' read -ra names <<< "$ONLY"
else
  names=()
  for f in "$MANIFEST_DIR"/*.csv; do
    names+=("$(basename "$f" .csv)")
  done
fi

if [[ ${#names[@]} -eq 0 ]]; then
  echo "No manifests found under $MANIFEST_DIR — run scripts/pre_process.sh first." >&2
  exit 1
fi

SYNC_ARGS=()
[[ $DRY_RUN -eq 1 ]] && SYNC_ARGS+=(--dryrun)

for name in "${names[@]}"; do
  manifest="$MANIFEST_DIR/$name.csv"
  if [[ ! -f "$manifest" ]]; then
    echo "[warn] manifest not found, skipping: $manifest" >&2
    continue
  fi

  echo
  echo "=== [$name] syncing to $BUCKET/$name/ ==="

  # For data/-relative paths: their dataset-dedicated root (2 path parts).
  # For absolute paths (external, shared trees): each unique parent dir.
  # Pure awk (no per-line fork) so this stays fast on 100k+-row manifests.
  dirs=$(tail -n +2 "$manifest" | cut -d, -f1 | awk -F'/' '
    {
      if ($0 ~ /^\//) {
        out = $1
        for (i = 2; i < NF; i++) out = out "/" $i
      } else {
        out = $1 "/" $2
      }
      print out
    }' | sort -u)

  n_dirs=0
  n_missing=0
  while IFS= read -r dir; do
    [[ -z "$dir" ]] && continue
    if [[ ! -d "$dir" ]]; then
      echo "[warn] directory not found, skipping: $dir" >&2
      n_missing=$((n_missing + 1))
      continue
    fi
    aws s3 sync "$dir" "$BUCKET/$name/${dir#/}" "${SYNC_ARGS[@]}"
    n_dirs=$((n_dirs + 1))
  done <<< "$dirs"

  # Manifest CSV itself, next to the files it describes.
  aws s3 cp "$manifest" "$BUCKET/$name/manifest.csv" "${SYNC_ARGS[@]}"

  echo "[$name] synced $n_dirs director(ies)$( [[ $n_missing -gt 0 ]] && echo ", $n_missing missing" )"
done

# Domain registry, once, so the bucket is self-describing for any dataset uploaded.
if [[ -f "$MANIFEST_DIR/domain_registry.json" ]]; then
  aws s3 cp "$MANIFEST_DIR/domain_registry.json" "$BUCKET/domain_registry.json" "${SYNC_ARGS[@]}"
fi

echo
echo "Done."
