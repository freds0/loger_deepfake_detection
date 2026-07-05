# Architecture — LOGER + Forgery Style Mixture

This document describes the model architecture and the training process of the
LOGER + FSM pipeline in detail, with pointers to the implementing modules and
the config keys that control each part. The original OSDFD pipeline is
summarised at the end.

Papers:

- **LOGER: Local–Global Ensemble for Robust Deepfake Detection in the Wild**
  (arXiv:2604.03558) — dual-branch architecture, MIL top-k pooling, logit
  fusion, multi-resolution inference.
- **Open-Set Deepfake Detection: A Parameter-Efficient Adaptation Method with
  Forgery Style Mixture** (arXiv:2408.12791) — FSM module (Sec. III-C,
  Eqs. 6–9), LoRA adaptation (Sec. III-B).

---

## 1. End-to-end pipeline

```
                        pixel_values (B, 3, H, W)
                                  │
                 ┌────────────────▼─────────────────┐
                 │  VFM backbone (never modified)   │   src/models/backbones.py
                 │  siglip2 │ dinov2 │ dinov3 │ clip│   config: backbone.*, peft.mode
                 └────────────────┬─────────────────┘
                        tokens (B, T, D)
                                  │
                 ┌────────────────▼─────────────────┐
                 │  Forgery Style Mixture (FSM)     │   src/models/fsm.py
                 │  TRAIN ONLY · fake samples only  │   config: fsm.*
                 └────────────────┬─────────────────┘
                        mixed tokens (B, T, D)
                    ┌─────────────┴──────────────┐
        pool(tokens)│                            │patches(tokens)
                    ▼                            ▼
         pooled (B, D)                 patch tokens (B, N, D)
                    │                            │
     ┌──────────────▼─────────┐   ┌──────────────▼───────────────┐
     │ GLOBAL BRANCH          │   │ LOCAL BRANCH                 │
     │ ImageClassifier        │   │ PatchClassifier              │
     │ fc(D→256)→ReLU→fc(→2)  │   │ fc(D→256)→ReLU→fc(→2) /patch │
     │ d_global = l_f − l_r   │   │ d_i = l_i^fake − l_i^real    │
     └──────────────┬─────────┘   │ MIL top-k: k = ⌊0.1·N⌋       │
                    │             │ d_local = (1/k) Σ_{i∈S_k} d_i│
                    │             └──────────────┬───────────────┘
                    └─────────────┬──────────────┘
                                  ▼
                 logit fusion  d = α_g·d_global + α_l·d_local
                 (uniform α = 1/2 by default)      P(fake) = σ(d)
```

Implementation: `src/models/loger.py` (`LOGERModel`, `LOGEROutput`,
`ImageClassifier`, `PatchClassifier`, `mil_topk_pool`).

The forward pass returns all intermediate logits (`logits`, `global_logits`,
`local_logits`, `patch_diffs (B, N)`, `pooled`) so every loss term can be
supervised independently.

---

## 2. Backbone abstraction

`src/models/backbones.py` exposes one contract for every Vision Foundation
Model, selected by `backbone.name`:

```python
tokens  = backbone(pixel_values)   # (B, T, D)  full token sequence
pooled  = backbone.pool(tokens)    # (B, D)     global image embedding
patches = backbone.patches(tokens) # (B, N, D)  patch tokens only
```

| Family    | Checkpoint (default)                        | Pooling            | Prefix tokens dropped by `patches()` | Normalisation |
|-----------|---------------------------------------------|--------------------|--------------------------------------|---------------|
| `siglip2` | `google/siglip2-base-patch16-224`           | MAP attention head | none (no `[CLS]`)                    | mean/std 0.5  |
| `dinov2`  | `facebook/dinov2-base`                      | `[CLS]` token      | 1 (`[CLS]`)                          | ImageNet      |
| `dinov3`  | `facebook/dinov3-vitb16-pretrain-lvd1689m`  | `[CLS]` token      | 1 + register tokens                  | ImageNet      |
| `clip`    | `openai/clip-vit-base-patch16`              | `[CLS]` token      | 1 (`[CLS]`)                          | CLIP stats    |

Key properties:

- **FSM operates on the full token sequence** returned by `forward`, so both
  the pooled global feature and the patch tokens seen by the local branch
  reflect the mixed statistics. `pool`/`patches` are pure post-processing;
  the backbone weights are never modified.
- **Multi-resolution input**: SigLIP 2 and CLIP interpolate their positional
  embeddings when the input size differs from the checkpoint's native size;
  DINOv2 does this natively. Any square resolution divisible by the patch
  size works.
- **DINOv3** requires `transformers>=4.56`; on older versions a guarded
  `ImportError` explains the alternatives.
- Normalisation statistics per family are mirrored in the data pipeline
  (`data.normalization_mean/std`; the LOGER entry points derive them
  automatically from `backbone.name`).

### PEFT modes (`peft.mode`)

| Mode     | Backbone weights | Extra params | Trainable                          |
|----------|------------------|--------------|-------------------------------------|
| `full`   | trainable        | —            | everything (default, matches the paper) |
| `frozen` | frozen           | —            | branch heads only                  |
| `lora`   | frozen           | LoRA on attention q/k/v (`inject_lora`) | LoRA matrices + branch heads |

LoRA (`src/models/lora.py`) wraps each attention projection as
`h = W·x + (α/r)·W_up(W_down(x))` with `W_up` zero-initialised, so the adapted
model starts identical to the pre-trained one. Config:
`peft.lora.{r=8, alpha=8.0, dropout=0.0}`. With SigLIP2-base + LoRA r=8 the
model has ≈0.84 M trainable / ≈93 M total parameters.

---

## 3. Forgery Style Mixture (FSM)

`src/models/fsm.py`, config `fsm.{enabled, probability=0.5, beta_alpha=0.1}`.

FSM is a feature-space augmentation that diversifies *forgery styles* during
training so the detector generalises to unseen manipulation methods. Per
training forward pass, with probability `fsm.probability`:

1. Select the fake samples of the batch (`is_fake` mask); real samples pass
   through unchanged.
2. Pair every fake sample with a fake sample from a **different forgery
   domain** (`domains` ids; identity if fewer than two domains are present —
   then FSM is a no-op).
3. Compute channel-wise statistics over the token dimension of both orderings
   (`F_sort` = original fakes, `F̃_sort` = domain-shuffled fakes):
   `μ, σ ∈ (F, 1, D)`.
4. Sample a mixing weight per sample: `δ ~ Beta(β, β)` with `β = 0.1`.
5. Mix the statistics and reconstruct (paper Eqs. 7–9):

   ```
   γ_mix = δ·σ(F_sort) + (1−δ)·σ(F̃_sort)          (Eq. 7)
   η_mix = δ·μ(F_sort) + (1−δ)·μ(F̃_sort)          (Eq. 8)
   F'_mix = γ_mix · (F_sort − μ(F_sort)) / σ(F_sort) + η_mix   (Eq. 9)
   ```

6. Scatter the mixed fake features back into their original batch positions.

**Placement and gating.** FSM sits *after* backbone feature extraction and
*before* both classification heads (`LOGERModel.forward`). It is active only
when the module is in `train()` mode **and** `apply_fsm=True` — which is only
the case inside `training_step`. Validation, test and predict always call the
model with `apply_fsm=False`, and `model.eval()` makes it an identity mapping
regardless, so FSM can never leak into inference.

---

## 4. Branches and fusion

### Global branch

`backbone.pool(tokens)` produces the image-level embedding (MAP head for
SigLIP 2, `[CLS]` otherwise). `ImageClassifier`
(`fc(D→256) → ReLU → fc(256→2)`, dropout `model.head_dropout`) outputs two
logits; the branch score is the logit difference `d_global = l_fake − l_real`.
The penultimate 256-d feature is exposed as `scl_features` for optional
feature-level supervision.

### Local branch

Every patch token goes through the shared-weight `PatchClassifier`
(`fc(D→256) → ReLU → fc(256→2)`), giving per-patch logit differences
`d_i = l_i^fake − l_i^real`, shape `(B, N)`. Image-level aggregation uses
**MIL top-k pooling** (`mil_topk_pool`):

```
k = max(1, ⌊ model.topk_ratio · N ⌋)          # paper: ratio = 0.1
d_local = (1/k) · Σ_{i ∈ S_k} d_i             # S_k = indices of the k largest d_i
```

Rationale (multiple-instance assumption): a manipulated face contains at least
a few strongly-forged patches; averaging only the top-k patch scores makes the
image score sensitive to localised manipulations while ignoring background.

### Logit fusion

`d = α_g·d_global + α_l·d_local` with uniform weights `α = 1/2`
(`model.fusion_weights: null`); custom weights can be set as a two-element
list. Fusion happens in logit space (not probabilities), matching the paper's
ensemble formulation `d̄(x) = Σ_m α_m d_m(x)`, `p = σ(d̄)` (Eq. 3) — applied
here to the two branches of a single backbone, whereas the paper fuses M = 5
independently trained models (see §10).

### Multi-resolution inference

`LOGERModel.forward_multi_resolution(x, resolutions)` resizes the batch to
each resolution (bicubic), runs the full model with FSM off, and averages the
fused logits. Controlled by `model.eval_resolutions` (e.g. `[224, 336]`);
lists with fewer than two entries fall back to single-resolution evaluation.
Used by validation/test/predict via `LOGERLightningModule._eval_logits`.

---

## 5. Objective

`src/losses/loger.py`, config `configs/loss/loger.yaml`. Each term is an
independent `nn.Module` with its own weight:

```
L = w_global · BCE(d_global, y)                       global branch     (1.0)
  + w_fused  · BCE(d, y)                              fused ensemble    (1.0)
  + w_ce     · BCE(d_local, y)                        local L_CE        (1.0)
  + w_auc    · L_AUC(d_local, y)                      local L_AUC       (0.5)
  + w_mil    · L_MIL({d_i}, y)                        local L_MIL       (0.5)
  + w_reg    · L_reg({d_i})                           local L_reg       (1e-4)
```

matching the paper's local objective `L_local = L_CE + 0.5·L_AUC + 0.5·L_MIL
+ L_reg` (the paper gives no closed forms for the last three; the standard
formulations below are used):

- **`L_CE`** — binary cross-entropy with logits on the MIL-pooled local score.
- **`L_AUC`** (`AUCSurrogateLoss`) — pairwise squared-hinge AUC surrogate: for
  every (fake `i`, real `j`) pair in the batch,
  `mean_{i,j} max(0, margin − (s_i − s_j))²` with `margin = loss.auc_margin`
  (1.0). Directly optimises the ranking that AUC measures; returns 0 for
  single-class batches.
- **`L_MIL`** (`MILLoss`) — patch-level multiple-instance supervision: real
  images contain no forged patches (BCE of **all** `d_i` toward 0); fake
  images contain at least `k` forged patches (BCE of the **top-k** `d_i`
  toward 1, same `k` as the pooling).
- **`L_reg`** (`PatchLogitRegularization`) — `mean(d_i²)`, an L2 penalty that
  keeps patch logits bounded under the hinge/MIL pressure.

`loss.pos_weight` optionally re-weights the positive class in all BCE terms.
All component values are logged per step (`train/global_bce`, `train/local_auc`,
…).

---

## 6. Training process

Implementation: `src/lightning/loger_module.py` (`LOGERLightningModule`),
entry point `train_loger.py` (or `train.py`, which dispatches on
`model.name`). PyTorch Lightning 2.x drives the loop; nothing in the training
code changes between single-GPU and DDP.

### Data pipeline

`src/data/dataset.py` + `datamodule.py` + `transforms.py`:

1. **Record discovery** — the FF++-style folder tree is scanned per split;
   labels (`real→0`, else `1`), forgery-domain ids (per manipulation folder,
   used by FSM) and splits (`train/val/test`, with aliases) are inferred. A
   manifest-CSV mode covers arbitrary datasets.
2. **Class balancing** — real frames are replicated ×`data.real_oversample`
   (default 4, following OSDFD) in the training split.
3. **Augmentation** (train split only, `data.augmentation.*`) — each transform
   fires with its own independent probability: horizontal flip, random
   resize, JPEG recompression, Gaussian blur, motion blur (OpenCV `filter2D`),
   Gaussian noise, brightness, contrast, colour jitter, random crop, random
   erasing. Val/test use plain resize + normalise.
4. **Normalisation** — per-backbone stats (`data.normalization_mean/std`),
   derived automatically from `backbone.name` by the LOGER entry points.
5. Batches are dicts: `pixel_values (B,3,H,W)`, `label (B,)`, `domain (B,)`,
   `path`.

### Optimisation

| Aspect | Value / mechanism | Config |
|---|---|---|
| Optimizer | **AdamW** over `requires_grad` params only | `optimizer.{name,lr=3e-5,beta1,beta2,weight_decay=1e-2}` |
| Schedule | cosine annealing (per step), optional **linear warmup** wrapped via `SequentialLR` | `scheduler.{name,t_max=30000,eta_min,warmup_steps,interval}` |
| Gradient clipping | handled by the Trainer | `trainer.gradient_clip_val=1.0` |
| Mixed precision | AMP fp16 by default; bf16 or full fp32 selectable | `trainer.precision=16-mixed \| bf16-mixed \| 32-true` |
| Duration | step-driven | `trainer.max_steps=30000`, `val_check_interval=1000` |
| Multi-GPU | `trainer.strategy=ddp trainer.devices=N`; eval metrics `all_gather` predictions across ranks before computing AUC/EER | `trainer.*` |
| Determinism | global seeding + deterministic mode | `seed`, `deterministic` |

### Train step

```
training_step:
  out  = model(x, is_fake=label.bool(), domains=domain, apply_fsm=True)   # FSM ON
  loss = LOGERLoss(out.logits, out.global_logits, out.local_logits,
                   out.patch_diffs, label)
  log: total + every component, lr
on_before_optimizer_step: log total L2 gradient norm (pre-clipping)
on_train_epoch_end:       log epoch time + peak GPU memory
```

### Validation / test

FSM off. Scores `σ(d)` are buffered per epoch, gathered across DDP ranks, and
the full metric suite is computed: **accuracy, AUC (primary), F1, precision,
recall, AP, EER, FPR, FNR** (`src/training/metrics.py`). Test additionally
logs confusion-matrix / ROC / PR figures to both loggers.

### Checkpointing & resume

- `ModelCheckpoint` writes `checkpoints/best.ckpt` (max `val/auc`) and
  `checkpoints/last.ckpt` (`configs/callbacks/loger.yaml`).
- **Auto-resume**: when `resume.enabled=true` and no `ckpt_path` is given, the
  newest `**/checkpoints/last.ckpt` under `resume.dir` (the experiment root,
  e.g. `outputs/loger_fsm_siglip2`) is used — scoped so checkpoints from other
  experiments are never picked up. Explicit `ckpt_path=...` always wins.
- The full Hydra config is stored in every checkpoint
  (`save_hyperparameters`), which lets `predict.py` / `test.py` recover the
  model family (`model.name`), normalisation stats and eval resolutions from
  the checkpoint alone.

### Logging

TensorBoard and Weights & Biases run **simultaneously**
(`configs/logger/default.yaml`). Logged: all loss components (train/val), AUC,
accuracy, precision, recall, F1, AP, EER, FPR, FNR, learning rate, gradient
norm, epoch time, peak GPU memory, trainable/total parameter counts, and
evaluation figures.

---

## 7. Inference

`predict_loger.py` / `src/inference/loger_predictor.py` (or the generic
`predict.py`, which auto-detects OSDFD vs LOGER checkpoints):

1. Load the LightningModule from the checkpoint; `eval()` mode (FSM inactive).
2. Rebuild the eval transform with the training-time normalisation stats and
   image size from the stored config.
3. Optional face detection/cropping (`--face-crop`).
4. Score = `σ(d)` from the fused logit, using multi-resolution averaging when
   `model.eval_resolutions` has more than one entry.

---

## 8. OSDFD pipeline (original, unchanged)

The repository's original re-implementation is preserved and shares the data,
metrics and logging stack:

```
SigLIP 2 (frozen) + LoRA (attn q/k/v) + CDC adapter (FFN)
  → patch tokens → FSM (train-only) → MAP pooling
  → optional global+local fusion → MLP head
L = BCE + λ · Single-Center Loss        (λ = 1, margin 0.01)
```

Modules: `src/models/osdfd.py`, `cdc_adapter.py`, `peft_inject.py`,
`src/losses/single_center_loss.py`, `src/lightning/module.py`. Entry points:
`train.py` / `test.py` / `predict.py` with `--config-name config`.

---

## 9. Config → component map

| Config group | Controls | Module |
|---|---|---|
| `backbone` | VFM family, checkpoint, image size, freeze, pretrained | `src/models/backbones.py` |
| `model` (loger) | top-k ratio, head width/dropout, fusion weights, eval resolutions | `src/models/loger.py` |
| `peft` | mode (full/lora/frozen), LoRA r/α/dropout | `backbones.inject_lora`, `lora.py` |
| `fsm` | enabled, probability, beta_alpha | `src/models/fsm.py` |
| `loss` (loger) | all six loss weights, AUC margin, pos_weight | `src/losses/loger.py` |
| `optimizer` | AdamW lr/betas/weight decay | `loger_module.configure_optimizers` |
| `scheduler` | cosine/step/none, warmup steps, interval | idem |
| `data` | root/manifest, batch, workers, oversampling, augmentation, normalisation | `src/data/*` |
| `trainer` | precision, devices, strategy, clipping, steps, val cadence | `src/utils/lightning_setup.py` |
| `logger` / `callbacks` | TB+W&B, checkpointing, early stopping | idem |
| `resume` | auto-resume toggle + experiment root | `train_loger.py` / `train.py` |

---

## 10. TODOs / known limitations

### Divergences from the LOGER paper (deliberate)

- **Single model vs ensemble**: the paper trains **five independent models**
  (M1/M2: DINOv3-Huge global, M3: MetaCLIP2-Huge global, M4/M5: DINOv3-Large
  local) and fuses their logits at inference only; this repo trains one
  backbone with a global and a local head, supervised jointly (global BCE +
  fused BCE + local objective). The fused-logit BCE has no counterpart in the
  paper (each paper model is trained separately).
- **PEFT vs full fine-tuning**: the paper uses full fine-tuning and explicitly
  rejects LoRA ("partial adaptation tends to underfit forgery-relevant
  cues"); `peft.mode=full` is the default here too (`lora`/`frozen` remain as
  cheaper ablations, not the paper setting).
- **Focal Loss**: the paper's global models use Focal Loss (M3 switches
  CE→Focal after 20% of training); all image-level terms here are BCE.
- **Discriminative learning rates**: the paper uses separate backbone/head
  learning rates; a single LR covers all trainable params here.
- **Horizontal-flip TTA** (paper: M1/M2/M4/M5) is not implemented.
- **Multi-resolution**: the paper trains members at one resolution and infers
  at another (e.g. M4: train 224, infer 384); `eval_resolutions` here instead
  averages one model over several inference resolutions.
- **Sampler**: paper uses `WeightedRandomSampler`; here real frames are
  replicated ×4 (OSDFD convention).
- **L_MIL** here additionally supervises *all* patches of real images toward
  0; the paper describes MIL supervision on the top-k patch scores only.

### Divergences from the OSDFD paper (deliberate)

- **Backbone**: SigLIP 2 replaces the paper's ImageNet-21K ViT-B / CLIP ViT-L
  (the purpose of this re-implementation).
- **CDC θ**: the paper's Eq. 3 is the pure central-difference form (θ=1); the
  default here is the CDCN blend `peft.cdc.theta=0.7`.
- LoRA α/scaling and initialisation, the adapter bottleneck width (64) and the
  GELU inside the adapter are unspecified in the paper; standard choices are
  used.

### Other

- **DINOv3**: the backbone code path is implemented and guarded, but requires
  `transformers>=4.56` (the `loger` env ships 4.53.2). Upgrading transformers
  is untested against the pinned Lightning/torch stack — validate the OSDFD
  and LOGER smoke tests after upgrading.
- **Local-loss formulations**: the LOGER paper gives no closed forms for
  `L_AUC`, `L_MIL` and `L_reg`; the implementations in `src/losses/loger.py`
  (pairwise squared hinge, top-k patch BCE, patch-logit L2) are standard
  choices. Their weights (`loss.auc_weight=0.5`, `loss.mil_weight=0.5`,
  `loss.reg_weight=1e-4`) may need tuning to reproduce paper numbers.
- **Multi-resolution inference** defaults to a single resolution
  (`model.eval_resolutions: [224]`). Enabling e.g. `[224, 336]` multiplies
  eval memory/compute per batch — check VRAM headroom (RTX 3060, 12 GB) or
  lower `data.batch_size` for evaluation.
- **Fusion weights** are fixed (uniform ½/½ by default, or a config-set pair);
  learned or validation-calibrated per-branch weights are not implemented.
- **W&B** runs online by default (`logger.wandb.offline=true` to disable);
  the first online run requires `wandb login`.
- **Smoke artifacts**: throwaway runs live under `outputs/smoke_*` and can be
  deleted; they are not referenced by the auto-resume logic of the three
  experiment configs.
- **No full-length training run** has been executed yet — the pipeline is
  verified end-to-end (training steps, validation, checkpointing, resume,
  evaluation, inference) but not trained to convergence on FF++.
