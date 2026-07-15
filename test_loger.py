"""LOGER + FSM evaluation entry point."""

from __future__ import annotations

import os

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from src.data.datamodule import ForgeryDataModule
from src.lightning.loger_module import LOGERLightningModule
from src.models.backbones import normalization_stats
from src.training.metrics import compute_metrics
from src.utils.ema import load_ema_weights
from src.utils.lightning_setup import build_loggers
from src.utils.seed import seed_everything


def _prepare_cfg(cfg: DictConfig) -> None:
    with open_dict(cfg):
        cfg.output_dir = HydraConfig.get().runtime.output_dir
        mean, std = normalization_stats(cfg.backbone.name)
        cfg.data.normalization_mean = list(mean)
        cfg.data.normalization_std = list(std)


@hydra.main(version_base=None, config_path="configs", config_name="loger_fsm_siglip2")
def main(cfg: DictConfig) -> None:
    _prepare_cfg(cfg)
    seed_everything(cfg.seed)
    if cfg.ckpt_path is None:
        raise ValueError("Set ckpt_path=/path/to/best.ckpt or last.ckpt to evaluate.")

    datamodule = ForgeryDataModule(**OmegaConf.to_container(cfg.data, resolve=True))
    model = LOGERLightningModule.load_from_checkpoint(cfg.ckpt_path, map_location="cpu")

    # `use_ema` is not part of the config schema (no yaml sets it); override
    # with `+use_ema=false` on the CLI to evaluate the raw (non-EMA) weights.
    if bool(OmegaConf.select(cfg, "use_ema", default=True)):
        used_ema = load_ema_weights(cfg.ckpt_path, model.model)
        print(f"[EMA] {'using EMA weights' if used_ema else 'no EMA state in checkpoint -- using raw weights'}")

    import lightning as L

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=1,
        precision=cfg.trainer.precision,
        logger=build_loggers(cfg),
        limit_test_batches=getattr(cfg.trainer, "limit_test_batches", None),
        limit_predict_batches=getattr(cfg.trainer, "limit_predict_batches", None),
        default_root_dir=cfg.output_dir,
    )
    trainer.test(model, datamodule=datamodule)

    predictions = trainer.predict(model, datamodule=datamodule)
    paths, probs, preds, labels = [], [], [], []
    for batch in predictions:
        if batch["path"] is not None:
            paths.extend(list(batch["path"]))
        probs.append(batch["prob"].numpy())
        preds.append(batch["pred"].numpy())
        if "label" in batch:
            labels.append(batch["label"].numpy())

    probs = np.concatenate(probs)
    preds = np.concatenate(preds)
    df = {"prob": probs, "pred": preds}
    if paths:
        df = {"path": paths, **df}
    if labels:
        y_true = np.concatenate(labels)
        df["label"] = y_true
        metrics = compute_metrics(y_true, probs)
        print("\n=== Evaluation metrics ===")
        for k, v in metrics.items():
            print(f"{k:>9}: {v:.4f}")

    out_csv = os.path.join(cfg.output_dir, "predictions.csv")
    os.makedirs(cfg.output_dir, exist_ok=True)
    pd.DataFrame(df).to_csv(out_csv, index=False)
    print(f"\nPredictions written to {out_csv}")


if __name__ == "__main__":
    main()
