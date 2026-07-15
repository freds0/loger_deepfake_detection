"""Factory helpers to build Lightning loggers, callbacks and the Trainer.

Keeps ``train.py`` / ``test.py`` thin and centralises the mapping from the Hydra
config to Lightning objects (TensorBoard + W&B loggers, checkpointing, early
stopping, LR / progress monitoring, and Trainer flags).
"""

from __future__ import annotations

import os

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import Logger, TensorBoardLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from .ema import EMACallback


def build_loggers(cfg: DictConfig) -> list[Logger]:
    """Build TensorBoard and/or W&B loggers per the ``logger`` config."""
    loggers: list[Logger] = []
    lc = cfg.logger

    if lc.tensorboard.enabled:
        os.makedirs(lc.tensorboard.save_dir, exist_ok=True)
        loggers.append(
            TensorBoardLogger(save_dir=lc.tensorboard.save_dir, name=lc.tensorboard.name)
        )

    if lc.wandb.enabled:
        loggers.append(
            WandbLogger(
                project=lc.wandb.project,
                name=lc.wandb.name,
                offline=lc.wandb.offline,
                log_model=lc.wandb.log_model,
                save_dir=lc.wandb.save_dir,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        )
    return loggers


def build_callbacks(cfg: DictConfig) -> list[L.Callback]:
    """Build checkpoint / early-stopping / monitor callbacks."""
    cc = cfg.callbacks
    callbacks: list[L.Callback] = []

    ckpt = cc.checkpoint
    callbacks.append(
        ModelCheckpoint(
            dirpath=os.path.join(cfg.output_dir, "checkpoints"),
            monitor=ckpt.monitor,
            mode=ckpt.mode,
            save_top_k=ckpt.save_top_k,
            save_last=ckpt.save_last,
            filename=ckpt.filename,
            auto_insert_metric_name=ckpt.auto_insert_metric_name,
        )
    )

    if cc.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=cc.early_stopping.monitor,
                mode=cc.early_stopping.mode,
                patience=cc.early_stopping.patience,
            )
        )

    has_logger = cfg.logger.tensorboard.enabled or cfg.logger.wandb.enabled
    if cc.lr_monitor.enabled and has_logger:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    if cc.ema.enabled:
        callbacks.append(EMACallback(decay=cc.ema.decay))

    return callbacks


def build_trainer(cfg: DictConfig, loggers: list[Logger], callbacks: list[L.Callback]) -> L.Trainer:
    """Build the Lightning Trainer from the ``trainer`` config."""
    tc = cfg.trainer
    return L.Trainer(
        accelerator=tc.accelerator,
        devices=tc.devices,
        strategy=tc.strategy,
        precision=tc.precision,
        max_steps=tc.max_steps,
        max_epochs=tc.max_epochs,
        accumulate_grad_batches=tc.accumulate_grad_batches,
        gradient_clip_val=tc.gradient_clip_val,
        val_check_interval=tc.val_check_interval,
        check_val_every_n_epoch=tc.check_val_every_n_epoch,
        log_every_n_steps=tc.log_every_n_steps,
        num_sanity_val_steps=tc.num_sanity_val_steps,
        limit_train_batches=getattr(tc, "limit_train_batches", None),
        limit_val_batches=getattr(tc, "limit_val_batches", None),
        limit_test_batches=getattr(tc, "limit_test_batches", None),
        limit_predict_batches=getattr(tc, "limit_predict_batches", None),
        deterministic=cfg.deterministic,
        # cudnn autotune: only safe when determinism isn't required (the two
        # are mutually exclusive in PyTorch/Lightning).
        benchmark=not cfg.deterministic,
        logger=loggers,
        callbacks=callbacks,
        default_root_dir=cfg.output_dir,
    )
