#!/usr/bin/env python3
# download_faceforensics_c23.py

from pathlib import Path
from huggingface_hub import snapshot_download

DATASET_ID = "bitmind/FaceForensicsC23"
OUTPUT_DIR = Path("data/FaceForensicsC23")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=str(OUTPUT_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"Dataset downloaded to: {path}")

if __name__ == "__main__":
    main()

