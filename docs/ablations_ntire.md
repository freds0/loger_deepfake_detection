# Backbone ablation protocol — NTIRE 2026 (P2.4)

Fixed protocol to compare backbone families on the NTIRE dataset before
committing to one for the full ~30k-step run. All variants use the **same**
subset, step budget, seed and EMA/threshold-calibration machinery, so the only
thing that varies is the backbone (and, for the LoRA variant, the PEFT mode).

## Why these four

`siglip2_base` is the current default (vision-language pretraining, used by
the LOGER paper). DINOv2/DINOv3 are self-supervised (no text-alignment bias),
which tends to help on low-level synthesis artifacts rather than semantic
content — plausibly a better fit for whole-image AI-generation detection than
face-swap-oriented priors. `siglip2_large` under full fine-tuning would not
fit the same step/time budget as the base models, so it runs under LoRA
instead (`peft=loger_lora`) to keep the comparison time-bounded; that is a
confound (both backbone *and* PEFT mode change), noted in the results, not
hidden.

## Fixed protocol

- Subset: `data.max_records_per_split={train: 100000, val: 20000}`
- Steps: `trainer.max_steps=8000`
- Seed: `seed=0`
- Augmentation: on (inherited from `loger_fsm_ntire.yaml`'s default block —
  hflip/random_resize/jpeg/blur/noise/brightness/contrast/color_jitter/crop)
- EMA: on (inherited from `configs/callbacks/loger.yaml`, decay 0.999)
- FSM: off (inherited — NTIRE has no per-generator domain labels, see
  `configs/loger_fsm_ntire.yaml`)

Each command below was verified to compose correctly via Hydra (`hydra.compose`,
not just written by hand) before being listed here.

### 1. `siglip2_base` (reference, full fine-tune)

```bash
python train_loger.py --config-name loger_fsm_ntire \
  data.root=<NTIRE_ROOT> \
  data.max_records_per_split='{train: 100000, val: 20000}' \
  trainer.max_steps=8000 seed=0 \
  backbone=siglip2_base
```

### 2. `dinov2_base` (full fine-tune)

```bash
python train_loger.py --config-name loger_fsm_ntire \
  data.root=<NTIRE_ROOT> \
  data.max_records_per_split='{train: 100000, val: 20000}' \
  trainer.max_steps=8000 seed=0 \
  backbone=dinov2_base \
  data.normalization_mean=[0.485,0.456,0.406] data.normalization_std=[0.229,0.224,0.225]
```

### 3. `dinov3_base` (full fine-tune)

Requires `transformers>=4.56` (`DINOv3ViTModel`) — upgrade only in the
ablation environment; the pinned `transformers==4.53.2` in `requirements.txt`
is unaffected, do not bump it repo-wide for this.

```bash
python train_loger.py --config-name loger_fsm_ntire \
  data.root=<NTIRE_ROOT> \
  data.max_records_per_split='{train: 100000, val: 20000}' \
  trainer.max_steps=8000 seed=0 \
  backbone=dinov3_base \
  data.normalization_mean=[0.485,0.456,0.406] data.normalization_std=[0.229,0.224,0.225]
```

### 4. `siglip2_large` + LoRA (confound: backbone AND peft mode both change)

```bash
python train_loger.py --config-name loger_fsm_ntire \
  data.root=<NTIRE_ROOT> \
  data.max_records_per_split='{train: 100000, val: 20000}' \
  trainer.max_steps=8000 seed=0 \
  backbone=siglip2_large peft=loger_lora
```

## Adoption criterion

Pick the variant with the best EMA `val/auc` (the EMA-swapped weights are
what `val/auc` measures once `configs/callbacks/loger.yaml`'s `ema.enabled` is
on — see P1.2). If the top two variants are within 0.2 AUC points, prefer the
one with lower inference cost (steps/s, `gpu/mem_alloc_GB`) over the raw
metric.

## Results

Fill in after each run (`val/auc`, `val/acc`, `val/eer` logged by
`_finalise_eval`; `train/epoch_time_s` and `gpu/mem_alloc_GB` logged every
epoch — see `on_train_epoch_end`).

| Backbone | PEFT | val/auc (EMA) | val/acc | val/eer | steps/s | gpu/mem_alloc_GB |
|---|---|---|---|---|---|---|
| siglip2_base | full | | | | | |
| dinov2_base | full | | | | | |
| dinov3_base | full | | | | | |
| siglip2_large | lora | | | | | |

**Decision:** _(fill in once the table is complete — winning backbone and why)_

---

# PLAN v0.2 architecture ablations

The variations proposed in `PLAN_v0.2.md` are now materialised as dedicated
Hydra configs (`configs/ntire_v2_*.yaml`), so each run is a single
`--config-name`, no long override strings. They all compose
`ntire_v2_base.yaml`, which fixes the shared regime (8k steps, 100k/20k subset,
seed 0, resume off, EMA on, augmentation on). Baseline = `ntire_v2_base` with no
overrides.

Run any variant with:

```bash
python train_loger.py --config-name ntire_v2_<id> \
  data.root=<NTIRE_ROOT>
```

Adoption rule (see PLAN_v0.2 §1.2): Δ `val/auc` (EMA) ≥ **+0.10 pp** over the
baseline; ties broken toward the simpler / cheaper variant. Fill the tables
after each run.

## Group 1 — config-only (no code changes)

| Config | Variation | val/auc (EMA) | val/acc | val/eer | steps/s | gpu/mem_GB |
|---|---|---|---|---|---|---|
| ntire_v2_base | baseline (topk 0.1, fixed fusion) | | | | | |
| ntire_v2_topk05 | MIL ratio 0.05 | | | | | |
| ntire_v2_topk20 | MIL ratio 0.20 | | | | | |
| ntire_v2_topk40 | MIL ratio 0.40 | | | | | |
| ntire_v2_topk100 | MIL ratio 1.0 (mean pool) | | | | | |
| ntire_v2_learnfusion | learnable fusion weights | | | | | |
| ntire_v2_nofused | drop fused-logit BCE | | | | | |
| ntire_v2_head512 | head hidden 512 | | | | | |
| ntire_v2_head1024 | head hidden 1024 | | | | | |
| ntire_v2_dinov2 | backbone DINOv2-base | | | | | |
| ntire_v2_dinov3 | backbone DINOv3-base | | | | | |
| ntire_v2_siglip2large | SigLIP2-large + LoRA | | | | | |

For `ntire_v2_learnfusion`, also record the converged
`fusion/w_global` / `fusion/w_local` (logged each epoch): a strong tilt toward
the global branch corroborates the top-k / LSE findings.

## Group 1b — test-time only (no training; reuse the baseline checkpoint)

V6: multi-resolution + hflip TTA are evaluation switches over the baseline
`best.ckpt`, not new runs:

```bash
python test_loger.py --config-name ntire_v2_base ckpt_path=<baseline_best.ckpt> \
  'model.eval_resolutions=[224,336]'
python test_loger.py --config-name ntire_v2_base ckpt_path=<baseline_best.ckpt> \
  'model.eval_resolutions=[224,336,448]' model.tta_hflip=true
```

| Eval mode | val/auc | val/acc | val/eer |
|---|---|---|---|
| [224] (baseline) | | | |
| [224,336] | | | |
| [224,336,448] | | | |
| [224,336,448] + hflip | | | |

## Group 2 — code-backed variants (flags default off)

| Config | Variation | val/auc (EMA) | val/acc | val/eer | steps/s | gpu/mem_GB |
|---|---|---|---|---|---|---|
| ntire_v2_focal | Focal Loss on global+fused | | | | | |
| ntire_v2_headlr10 | head LR 10x backbone | | | | | |
| ntire_v2_lse_t05 | LSE pooling, tau 0.5 | | | | | |
| ntire_v2_lse_t10 | LSE pooling, tau 1.0 | | | | | |
| ntire_v2_scl | + Single-Center Loss (λ=1) | | | | | |
| ntire_v2_fsm_nodomain | domain-free FSM (reactivated) | | | | | |

## Group 3 — mini-ensemble (V12)

Train two specialists, then logit-average their `predictions.csv`:

```bash
python train_loger.py --config-name ntire_v2_spec_global data.root=<NTIRE_ROOT>
python train_loger.py --config-name ntire_v2_spec_local  data.root=<NTIRE_ROOT>
python test_loger.py --config-name ntire_v2_spec_global ckpt_path=<global_best.ckpt>
python test_loger.py --config-name ntire_v2_spec_local  ckpt_path=<local_best.ckpt>
python scripts/fuse_predictions.py <global_out>/predictions.csv <local_out>/predictions.csv
```

| Model | val/auc | val/acc | val/eer |
|---|---|---|---|
| global specialist | | | |
| local specialist | | | |
| logit-average ensemble | | | |

Adopt the ensemble only if Δ `val/auc` ≥ +0.15 pp over the best single model
(it costs N× inference).

## Combining winners

The winners of at most one variation per eixo (pooling, objective, LR,
backbone, FSM) are combined into `loger_fsm_ntire_v3` and re-validated at 8k
before the full 30k-step run — see PLAN_v0.2 §5.
