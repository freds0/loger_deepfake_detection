#!/usr/bin/env python3
"""
Download do dataset:
deepfakesMSU/NTIRE-RobustAIGenDetection-train

Requisitos:
    pip install -U huggingface_hub
"""

from pathlib import Path
from huggingface_hub import snapshot_download

DATASET_ID = "deepfakesMSU/NTIRE-RobustAIGenDetection-train"
OUTPUT_DIR = Path("data/NTIRE-RobustAIGenDetection-train")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DATASET_ID} ...")

    local_path = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=str(OUTPUT_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=8,  # aumenta a velocidade do download
    )

    print("\nDownload concluído!")
    print(f"Dataset salvo em:\n{local_path}")


if __name__ == "__main__":
    main()
