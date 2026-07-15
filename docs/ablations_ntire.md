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
