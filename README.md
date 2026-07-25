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
- Discriminative learning rates (backbone vs head) are not implemented;
  horizontal-flip TTA is available at predict time via `model.tta_hflip=true`.
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
  loger_fsm_ntire.yaml         # LOGER on NTIRE 2026 (FSM off, augmentation on)
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
  preprocess_hydrafake.py      # HydraFake (multi-generator zip benchmark) -> manifest
  preprocess_hidf.py           # HiDF (face-swap frames) -> manifest
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

### NTIRE 2026 (shard layout)

The NTIRE 2026 *Robust AI-generated Image Detection In The Wild* release is a
set of shards, each with a labels CSV and an image folder:

```
<root>/                          # default: data/NTIRE-RobustAIGenDetection-train
  shard_0/
    labels.csv                   # columns: ,image_name,label  (0=real, 1=fake)
    images/*.png|jpg
  shard_1/ ... shard_N/
```

Select it with `data=ntire` (already the default in `loger_fsm_ntire.yaml`) and
list the shards present on disk via `data.train_shards`. There is no separate
labelled val/test set: a deterministic 5% of the files (`data.val_fraction`,
by filename hash) is held out for validation. Leftover `shard_*.zip` archives
next to the extracted folders are ignored and can be deleted.

### Multi-dataset training (manifest CSVs)

Eight raw datasets are supported out of the box, each turned into a manifest
CSV (`path,label,domain,split`) by its own `scripts/preprocess_*.py` script:

| Dataset | Script | Raw format |
| --- | --- | --- |
| FaceForensics++ (C23) | `preprocess_ffpp.py` | videos, MTCNN face-crop |
| Celeb-DF-v2 | `preprocess_celebdf.py` | videos, MTCNN face-crop |
| SDFVD 2.0 | `preprocess_sdfvd.py` | videos, MTCNN face-crop |
| Comprehensive (Roop/Akool) | `preprocess_comprehensive.py` | zip of pre-cropped frames |
| DF40 | `preprocess_df40.py` | zip zoo, 39 usable generator methods |
| DFBench | `preprocess_dfbench.py` | 21 source zips + JSON labels (general, non-face) |
| HydraFake | `preprocess_hydrafake.py` | 8 zips (train/val real+fake, 4 test subsets), ~30 generator methods |
| HiDF | `preprocess_hidf.py` | pre-extracted face-swap frames, real/fake in separate folders |

Real frames get `domain=0`; every forgery method/source registers its own
domain id via `DomainRegistry` (`data/manifests/domain_registry.json`), so FSM
sees one distinct domain per manipulation method across all datasets combined.
Splits without an official train/val/test file are assigned deterministically
by hashing an identity/video key (`src/data/manifest_utils.py::deterministic_split`),
so re-running a script never reshuffles an already-processed video across
splits. Video-based scripts (`ffpp`, `celebdf`, `sdfvd`) share MTCNN frame
extraction from `src/data/video_extract.py`.

Run everything with the orchestrator script:

```bash
scripts/pre_process.sh                       # all 8 datasets, full extraction (hours on a single GPU)
scripts/pre_process.sh --dry-run              # tiny per-dataset sample, fast sanity check
scripts/pre_process.sh --only ffpp celebdf    # run a subset (ffpp celebdf sdfvd comprehensive df40 dfbench hydrafake hidf)
```

It runs each `preprocess_*.py` in turn, reports a failure per dataset without
aborting the rest, and prints a summary of which manifests were written under
`data/manifests/`. Each script can also be run standalone (e.g.
`python scripts/preprocess_df40.py --limit-per-method 5` for a quick test) —
see `--help` on each for dataset-specific options (`--root`/`--video-root`,
`--out-root`, `--manifest-out`, frame-count and face-margin knobs for the
video-based scripts).

Video-based preprocessing (`ffpp`, `celebdf`, `sdfvd`) needs
`facenet-pytorch` for MTCNN face detection — install it separately (see
`requirements.txt`); it is not a hard dependency of the rest of the repo.

`configs/data/combined.yaml` lists all 8 manifests; any that haven't been
generated yet are skipped with a warning, so training can start after running
only a subset of the preprocessing scripts. Train across every combined
dataset with `--config-name loger_fsm_combined` (see Training below).

To package everything actually used in training (every path referenced by
`data/manifests/*.csv`, the manifests themselves, and the domain registry)
into a single archive:

```bash
scripts/archive_dataset.sh                          # -> data/processed_dataset_<timestamp>.tar.bz2
scripts/archive_dataset.sh /path/to/out.tar.bz2      # explicit output path
scripts/archive_dataset.sh --manifests ffpp,celebdf  # archive a subset of datasets
```

Some datasets (DF40, HiDF) reference images in place under the original
dataset tree instead of a local copy, so the archive follows those paths too
— the script only reads files, it never modifies anything.

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
python train_loger.py --config-name loger_fsm_siglip2 tune_batch_size=true   # find data.batch_size, then exit
```

or `./scripts/train_loger.sh --config-name loger_fsm_siglip2 ...` (activates the
env for you).

### Comparing architecture variations on the full combined dataset

`scripts/train_arch_ablations.sh` trains every architecture variation (MIL
top-k ratio, logit fusion, head capacity, backbone) on top of
`loger_fsm_combined`, one Hydra override per variant — the same variations as
`configs/ntire_v2_*.yaml` (see `docs/ablations_ntire.md`), but against the
full 8-dataset manifest instead of the NTIRE subset:

```bash
scripts/train_arch_ablations.sh                       # all 12 variants, full trainer.max_steps (30000)
scripts/train_arch_ablations.sh --steps 8000           # override the step budget for every variant
scripts/train_arch_ablations.sh --only base dinov2     # run a subset
scripts/train_arch_ablations.sh --dry-run              # ~10-step smoke run per variant, fast sanity check
```

Each variant disables `resume` (they'd otherwise share
`resume.dir: outputs/loger_fsm_combined`) and logs to its own
`outputs/loger_fsm_combined/<date>/<time>/`; compare them with
`tensorboard --logdir outputs/loger_fsm_combined` on `val/auc` (EMA), same
adoption rule as the NTIRE protocol (Δ ≥ +0.10 pp).

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

### Training on NTIRE 2026

`loger_fsm_ntire.yaml` is the dedicated experiment config: FSM off (NTIRE has
no per-generator domain labels), degradation augmentation on, `bf16-mixed`,
500 warmup steps, balanced sampling and auto-resume from
`outputs/loger_fsm_ntire/`. The config defaults to shards `[0..6]` — override
`data.train_shards` to match the shards actually on disk.

```bash
# 1. validate the dataset (CSVs, corrupt images, class distribution)
python scripts/validate_dataset.py \
  --root data/NTIRE-RobustAIGenDetection-train --shards 0 1 2 3 4 5

# 2. ~100-step smoke test
python train_loger.py --config-name loger_fsm_ntire \
  'data.train_shards=[0,1,2,3,4,5]' \
  trainer.max_steps=100 trainer.val_check_interval=100 \
  trainer.limit_val_batches=20 resume.enabled=false

# 3. (optional) find the largest data.batch_size for this GPU, then exit
python train_loger.py --config-name loger_fsm_ntire \
  'data.train_shards=[0,1,2,3,4,5]' tune_batch_size=true

# 4. full training (auto-resumes from last.ckpt if interrupted)
python train_loger.py --config-name loger_fsm_ntire \
  'data.train_shards=[0,1,2,3,4,5]'
```

Monitor with `tensorboard --logdir outputs/loger_fsm_ntire`; checkpointing
tracks `val/auc` and keeps `best.ckpt` + `last.ckpt`. If you raise
`data.batch_size` well above the default 48 (step 3), scale `optimizer.lr`
roughly linearly with it.

### Multi-GPU (DDP)

```bash
python train_loger.py --config-name loger_fsm_ntire trainer.devices=8 trainer.strategy=ddp
```

- **Effective batch size** = `data.batch_size × trainer.devices` (each rank
  loads its own `data.batch_size`-sized batch). Scale `optimizer.lr` linearly
  with the effective batch if you change `trainer.devices` relative to a
  tuned single-GPU run (e.g. 2x devices -> ~2x `lr`).
- **`data.balanced_sampling`** (the NTIRE default class-rebalancing sampler,
  see P0.3) is **single-GPU only** in this version — `WeightedRandomSampler`
  isn't DDP-aware, and `train_dataloader()` raises if
  `trainer.world_size > 1` with it enabled. Either run single-device or set
  `data.balanced_sampling=false data.real_oversample=4` for multi-GPU runs.
- **`trainer.val_check_interval`** counts optimizer steps **per rank**, not
  global steps — validation frequency in wall-clock time is unaffected by
  `trainer.devices`, but the *step number* it validates at is not comparable
  across runs with a different device count.

---

## Evaluation

```bash
python test_loger.py --config-name loger_fsm_siglip2 \
    ckpt_path=outputs/loger_fsm_siglip2/DATE/TIME/checkpoints/best.ckpt
# or: ./scripts/eval_loger.sh ckpt_path=...
```

Reports ACC, AUC, F1, precision, recall, AP, EER, FPR, FNR, logs
confusion/ROC/PR figures and writes `predictions.csv`.

**Multi-resolution + horizontal-flip TTA.** Both only affect `predict_step`
(the `predictions.csv` this script writes), not the `trainer.test()` metrics
printed above; average one model's logits over several inference resolutions
and/or over `x`/`hflip(x)`:

```bash
python test_loger.py --config-name loger_fsm_siglip2 \
    ckpt_path=outputs/loger_fsm_siglip2/DATE/TIME/checkpoints/best.ckpt \
    model.eval_resolutions=[224,336,448] model.tta_hflip=true
```

Compare the "Evaluation metrics" table across `eval_resolutions=[224]` (single-
resolution baseline) vs `[224,336,448]` vs `+model.tta_hflip=true` and keep the
combination with the best val AUC/acc for the final submission.

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
