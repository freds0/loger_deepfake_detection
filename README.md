# LOGER + Forgery Style Mixture — Deepfake Detection

This repository hosts two related face-forgery detection pipelines that share
the same data stack, logging, metrics and Hydra configuration system:

1. **LOGER + FSM** (primary) — a single-backbone implementation of the
   local–global detector design of
   > **LOGER: Local–Global Ensemble for Robust Deepfake Detection in the Wild**
   > (arXiv:2604.03558)

   trained with the **Forgery Style Mixture (FSM)** feature-space augmentation
   of
   > **Open-Set Deepfake Detection: A Parameter-Efficient Adaptation Method
   > with Forgery Style Mixture** (arXiv:2408.12791)

2. **OSDFD** (original re-implementation, kept intact) — the parameter-efficient
   open-set detector of the second paper on a SigLIP 2 backbone (LoRA + CDC
   adapter + FSM + Single-Center Loss).

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for a detailed description of the
model architecture and the training process.

---

## LOGER + FSM at a glance

```
image ─► VFM backbone (SigLIP2 / DINOv2 / DINOv3 / CLIP)
              │  token sequence (B, T, D)
              ▼
        Forgery Style Mixture          ◄─ train-only, fake samples only
              │
    ┌─────────┴──────────┐
    ▼ global branch      ▼ local branch
  pooled embedding     patch tokens (B, N, D)
  image classifier     patch classifier (2 logits/patch)
  d_global             d_i = l_fake − l_real
    │                  MIL top-k pooling, k = ⌊0.1·N⌋ → d_local
    └────────┬─────────┘
             ▼
   logit fusion  d = ½·d_global + ½·d_local  →  σ(d) = P(fake)
```

- **Backbones** swappable via config: `siglip2` (default), `dinov2`, `dinov3`
  (needs `transformers>=4.56`), `clip`.
- **PEFT** via config: `full` (default, matches the paper), `lora`, `frozen`.
- **FSM** mixes AdaIN feature statistics between fake samples of *different*
  forgery domains (`δ ~ Beta(0.1, 0.1)`, prob. 0.5) — training only, never at
  validation/inference. The backbone is never modified.
- **Local loss** = CE + 0.5·AUC + 0.5·MIL + reg, each term independent and
  weighted via config; global and fused logits use BCE.
- **Multi-resolution inference** via `model.eval_resolutions=[224,336,...]`.

---

## Alignment with the papers

The code was audited against both papers (`docs/`).

**Faithful to LOGER** (arXiv:2604.03558): per-patch two-logit classifier with
`d_i = l_i^fake − l_i^real`; MIL top-k pooling with `k = ⌊0.1·N⌋` (Eq. 1);
local objective `L_CE + 0.5·L_AUC + 0.5·L_MIL + L_reg` (Eq. 2 — the paper
gives no closed forms for the last three terms, so standard formulations are
used, see `ARCHITECTURE.md` §5); global two-logit MLP head
(`D→256→2`, ReLU, Dropout 0.1); logit-space fusion with uniform weights and
`p = σ(d̄)` (Eq. 3); AdamW (β₁=0.9, β₂=0.999, weight decay 1e-2) with gradient
clipping at 1.0; degradation-style training augmentation.

**Deliberate divergences from LOGER:**

- The paper is an **ensemble of five independently trained models**
  (2× DINOv3-Huge + MetaCLIP2-Huge global, 2× DINOv3-Large local) fused at
  inference; this repo trains **one backbone with a global and a local head**,
  supervised jointly (global BCE + fused BCE + local objective).
- The paper uses **full fine-tuning** and explicitly argues against LoRA;
  `peft.mode=full` is the default here too (`lora`/`frozen` remain available
  as cheaper ablations).
- The paper trains its global models with **Focal Loss** (M3: CE→Focal); all
  image-level terms here use BCE.
- Discriminative learning rates (backbone vs head) and horizontal-flip TTA are
  not implemented.
- The paper realises multi-resolution via train-low/infer-high ensemble
  members; here `model.eval_resolutions` averages one model's logits over
  several inference resolutions.
- Class imbalance: paper uses a `WeightedRandomSampler`; here real frames are
  oversampled ×4 (the OSDFD convention).

**Faithful to OSDFD/FSM** (arXiv:2408.12791): FSM exactly as Eqs. 6–9
(fake-only, cross-domain pairing, `δ ~ Beta(0.1, 0.1)`, activation prob. 0.5,
train-only, applied after the transformer blocks and before the heads); LoRA
on attention q/k/v with r=8; CDC adapter structure (Eqs. 1–2); Single-Center
Loss (Eqs. 11–13, margin 0.01, λ=1, penultimate features); Adam `lr=3e-5`
without decay, batch 48, 30k iterations, real faces oversampled ×4, no
pixel-space augmentation, 224² faces cropped with 1.3× margin.
**Divergences:** the backbone is SigLIP 2 instead of ImageNet-21K ViT-B/CLIP
(the purpose of this re-implementation), and the CDC operator defaults to the
CDCN blend `θ=0.7` — set `peft.cdc.theta=1.0` for the paper's pure
central-difference form (Eq. 3).

---

## Project structure

```
configs/
  loger_fsm_siglip2.yaml       # LOGER+FSM, SigLIP2 (main experiment)
  loger_fsm_dinov2.yaml        # LOGER+FSM, DINOv2
  loger_fsm_baseline.yaml      # LOGER without FSM (ablation)
  loger_fsm_combined.yaml      # LOGER+FSM trained across every preprocessed dataset
  config.yaml                  # OSDFD defaults
  data/combined.yaml           # manifest list for cross-dataset training
  backbone/ model/ peft/ fsm/ loss/ optimizer/ scheduler/ data/ trainer/ logger/ callbacks/
docs/                          # source papers
scripts/                       # {train,eval,infer}_loger.sh + OSDFD scripts
  pre_process.sh               # runs every preprocess_*.py below (--dry-run, --only)
  archive_dataset.sh           # tar.bz2 of every file referenced by data/manifests/*.csv
  preprocess_ffpp.py           # FaceForensics++ videos -> face crops
  preprocess_celebdf.py        # Celeb-DF-v2 videos -> face crops
  preprocess_sdfvd.py          # SDFVD 2.0 videos -> face crops
  preprocess_comprehensive.py  # Comprehensive Roop/Akool (pre-cropped zip) -> manifest
  preprocess_df40.py           # DF40 (40-method zoo) -> manifest
  preprocess_dfbench.py        # DFBench (general AI-image benchmark) -> manifest
  download/                    # dataset download helpers
src/
  data/                        # dataset, LightningDataModule, transforms, face detection
                               # manifest_utils.py (split/domain-registry helpers),
                               # video_extract.py (shared MTCNN frame extraction)
  models/                      # backbones.py (VFM abstraction), loger.py, fsm.py,
                               # lora.py, osdfd.py, cdc_adapter.py, head.py, ...
  losses/                      # loger.py (CE/AUC/MIL/reg), SCL, combined
  lightning/                   # loger_module.py, module.py (OSDFD)
  training/                    # metrics (ACC/AUC/F1/precision/recall/AP/EER/FPR/FNR)
  inference/                   # loger_predictor.py, predictor.py
  utils/                       # seeding + Lightning factory helpers
tests/                         # fast offline smoke tests (CPU, no downloads)
train_loger.py test_loger.py predict_loger.py    # LOGER entry points
train.py       test.py       predict.py          # dispatching entry points (OSDFD + LOGER)
```

---

## Installation

All work uses the conda environment **`loger`**.

```bash
conda activate loger

# PyTorch matching your CUDA (example: CUDA 12.1):
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Everything else:
pip install -r requirements.txt
```

> SigLIP 2 requires `transformers>=4.53`; DINOv3 requires `transformers>=4.56`
> (a guarded `ImportError` tells you when it is unavailable). Optional face
> detectors (`dlib`, `retina-face`) and the MTCNN preprocessing dependency
> (`facenet-pytorch`) are commented out in `requirements.txt`.

Verify the install with the offline smoke tests (no checkpoint download, runs
on CPU in seconds):

```bash
pytest tests/ -q
```

---

## Data

### Expected layout (FaceForensics++-style frames)

```
<root>/
  train/
    real/**/*.png                 # bona-fide      -> label 0, domain 0
    Deepfakes/**/*.png            # forgery domain 1
    Face2Face/**/*.png            # forgery domain 2
    FaceSwap/**/*.png             # forgery domain 3
    NeuralTextures/**/*.png       # forgery domain 4
  val/   (same structure)         # aliases like "valid"/"validation" also work
  test/  (same structure)
```

Labels, forgery domains and splits are inferred from the folder tree; distinct
manipulation subfolders become distinct **forgery source domains** for FSM.
Point the config at it with `data.root=/path/to/<root>`. For arbitrary datasets
(CDF, DFDC, WDF, …) use a manifest CSV with columns `path,label,domain[,split]`
and `data.source=manifest data.manifest=file.csv`. `data.manifest` also accepts
a **list** of CSVs, which is how the multi-dataset setup below combines
several preprocessed sources into one training set.

To produce this layout from raw FF++ videos see `scripts/preprocess_ffpp.py`
and `scripts/download/` (dataset download helpers).

### Multi-dataset training (manifest CSVs)

Six raw datasets are supported out of the box, each turned into a manifest CSV
(`path,label,domain,split`) by its own `scripts/preprocess_*.py` script:

| Dataset | Script | Raw format |
| --- | --- | --- |
| FaceForensics++ (C23) | `preprocess_ffpp.py` | videos, MTCNN face-crop |
| Celeb-DF-v2 | `preprocess_celebdf.py` | videos, MTCNN face-crop |
| SDFVD 2.0 | `preprocess_sdfvd.py` | videos, MTCNN face-crop |
| Comprehensive (Roop/Akool) | `preprocess_comprehensive.py` | zip of pre-cropped frames |
| DF40 | `preprocess_df40.py` | zip zoo, 39 usable generator methods |
| DFBench | `preprocess_dfbench.py` | 21 source zips + JSON labels (general, non-face) |

Real frames get `domain=0`; every forgery method/source registers its own
domain id via `DomainRegistry` (`data/manifests/domain_registry.json`), so FSM
sees one distinct domain per manipulation method across all datasets combined.
Splits without an official train/val/test file are assigned deterministically
by hashing an identity/video key (`src/data/manifest_utils.py::deterministic_split`),
so re-running a script never reshuffles an already-processed video across
splits.

Run everything with the orchestrator script:

```bash
scripts/pre_process.sh                       # all 6 datasets, full extraction (hours on a single GPU)
scripts/pre_process.sh --dry-run              # tiny per-dataset sample, fast sanity check
scripts/pre_process.sh --only ffpp celebdf    # run a subset (ffpp celebdf sdfvd comprehensive df40 dfbench)
```

It activates the `loger` conda env, runs each `preprocess_*.py` in turn,
reports a failure per dataset without aborting the rest, and prints a summary
of which manifests were written under `data/manifests/`. Each script can also
be run standalone (e.g. `python scripts/preprocess_df40.py --limit-per-method 5`
for a quick test) — see `--help` on each for dataset-specific options
(`--root`/`--video-root`, `--out-root`, `--manifest-out`, frame-count and
face-margin knobs for the video-based scripts).

Video-based preprocessing (`ffpp`, `celebdf`, `sdfvd`) needs
`facenet-pytorch` for MTCNN face detection — install it separately (see
`requirements.txt`); it is not a hard dependency of the rest of the repo.

`configs/data/combined.yaml` lists all 6 manifests; any that haven't been
generated yet are skipped with a warning, so training can start after running
only a subset of the preprocessing scripts. See **Training** below.

To package everything actually used in training (every path referenced by
`data/manifests/*.csv`, the manifests themselves, and the domain registry)
into a single archive:

```bash
scripts/archive_dataset.sh                          # -> data/processed_dataset_<timestamp>.tar.bz2
scripts/archive_dataset.sh /path/to/out.tar.bz2      # explicit output path
scripts/archive_dataset.sh --manifests ffpp,celebdf  # archive a subset of datasets
```

Some datasets (DF40) reference images in place under the original dataset
tree instead of a local copy, so the archive follows those paths too — the
script only reads files, it never modifies anything.

---

## Training (LOGER + FSM)

```bash
conda activate loger
python train_loger.py --config-name loger_fsm_siglip2
python train_loger.py --config-name loger_fsm_dinov2
python train_loger.py --config-name loger_fsm_baseline      # FSM off (ablation)
python train_loger.py --config-name loger_fsm_combined      # cross-dataset (see Multi-dataset training above)

# common overrides
python train_loger.py --config-name loger_fsm_siglip2 data.root=/data/ffpp_frames
python train_loger.py --config-name loger_fsm_siglip2 peft.mode=frozen      # heads only
python train_loger.py --config-name loger_fsm_siglip2 peft.mode=lora        # LoRA (cheaper ablation)
python train_loger.py --config-name loger_fsm_siglip2 fsm.enabled=false
python train_loger.py --config-name loger_fsm_siglip2 trainer.precision=bf16-mixed
python train_loger.py --config-name loger_fsm_siglip2 trainer.strategy=ddp trainer.devices=2
python train_loger.py --config-name loger_fsm_siglip2 scheduler.warmup_steps=1000
```

or `./scripts/train_loger.sh --config-name loger_fsm_siglip2 ...` (activates the
env for you).

- **Optimizer**: AdamW (`lr`, `weight_decay`, betas via `configs/optimizer/adamw.yaml`),
  cosine schedule with optional linear warmup, gradient clipping
  (`trainer.gradient_clip_val`).
- **Mixed precision**: `trainer.precision=16-mixed` (default) or `bf16-mixed`
  or `32-true`.
- **Checkpoints**: `checkpoints/best.ckpt` (highest `val/auc`) and
  `checkpoints/last.ckpt` per run.
- **Auto-resume**: with `resume.enabled=true` (default in the LOGER configs)
  training resumes from the newest `last.ckpt` under `resume.dir` (the
  experiment root, e.g. `outputs/loger_fsm_siglip2`). Disable with
  `resume.enabled=false` or resume explicitly with `ckpt_path=...`.
- **Logging**: TensorBoard **and** W&B simultaneously — train/val losses (all
  components), AUC, accuracy, precision, recall, F1, AP, EER, learning rate,
  gradient norm, epoch time, GPU memory, ROC/PR/confusion figures.

### Data augmentation

Enabled per experiment config; each transform has an **independent
probability** (`0` disables it): horizontal flip, random resize, JPEG
compression, Gaussian blur, motion blur, Gaussian noise, brightness, contrast,
color jitter, random crop (+ random erasing). Applied only to the training
split; val/test always use plain resize + normalise.

```bash
python train_loger.py --config-name loger_fsm_siglip2 \
  data.augmentation.jpeg=0.5 data.augmentation.motion_blur=0.2 \
  data.augmentation.gaussian_noise=0.15
```

---

## Evaluation

```bash
python test_loger.py --config-name loger_fsm_siglip2 \
    ckpt_path=outputs/loger_fsm_siglip2/DATE/TIME/checkpoints/best.ckpt
# or: ./scripts/eval_loger.sh ckpt_path=...
```

Reports ACC, AUC, F1, precision, recall, AP, EER, FPR, FNR, logs
confusion/ROC/PR figures and writes `predictions.csv`.

---

## Inference

```bash
# single image
python predict_loger.py --ckpt best.ckpt --input face.png

# folder (+ CSV), with face cropping
python predict_loger.py --ckpt best.ckpt --input imgs/ --face-crop --csv preds.csv
# or: ./scripts/infer_loger.sh --ckpt ... --input ...
```

Outputs prediction (`real`/`fake`), `P(fake)`, confidence and raw logit. FSM is
always disabled at inference; normalisation stats and multi-resolution settings
are recovered from the checkpoint. The generic `predict.py` auto-detects
whether a checkpoint is OSDFD or LOGER.

---

## Configuration

Everything is configured through Hydra (`configs/`); no hyperparameters are
hardcoded. Key groups: `backbone` (siglip2/dinov2/dinov3/clip), `model`
(loger/osdfd), `peft` (mode: lora/frozen/full), `fsm`, `loss`, `optimizer`,
`scheduler`, `data`, `trainer`, `logger`, `callbacks`. Override any value on
the CLI (`group.key=value`) or swap a whole group (`backbone=dinov2_base`).

```yaml
fsm:
  enabled: true
  probability: 0.5      # activation probability per forward pass
  beta_alpha: 0.1       # delta ~ Beta(0.1, 0.1)
```

---

## OSDFD (original pipeline)

The original SigLIP 2 OSDFD re-implementation is untouched and remains fully
functional through `train.py` / `test.py` / `predict.py` with the default
`config.yaml` (LoRA + CDC adapter + FSM + Single-Center Loss, frozen backbone,
`L = BCE + λ·SCL`). Paper-faithful settings: Adam, `lr=3e-5`, batch 48, 30k
steps, real faces oversampled ×4, FSM prob. 0.5 with `Beta(0.1, 0.1)`, SCL
margin 0.01. See `ARCHITECTURE.md` §OSDFD and the paper for details.

```bash
python train.py data.root=/path/to/ffpp_frames        # OSDFD training
python train.py fsm.enabled=false peft.cdc.enabled=false   # ablations
```

## License / citation

Cite the LOGER, OSDFD and SigLIP 2 papers (see `docs/`). This repository is a
re-implementation for research purposes.
