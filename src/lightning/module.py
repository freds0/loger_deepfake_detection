"""LightningModule for OSDFD training / validation / testing / prediction.

Implements the full training pipeline of the OSDFD paper on a SigLIP 2 backbone
using PyTorch Lightning 2.x (no custom training loop). The objective is
``L = L_BCE + lambda * L_SCL`` (Eq. 10); the Forgery Style Mixture module is
active only during ``training_step`` and disabled for validation / test /
predict (paper Sec. III-C).
"""

from __future__ import annotations

import time

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig

from ..losses.combined import OSDFDLoss
from ..models.osdfd import OSDFDModel
from ..models.peft_inject import CDCConfig, LoRAConfig
from ..training.metrics import compute_metrics, log_figures


def build_model(cfg: DictConfig) -> OSDFDModel:
    """Construct an :class:`OSDFDModel` from the ``model`` config group."""
    lora_cfg = cfg.peft.lora
    cdc_cfg = cfg.peft.cdc
    lora = (
        LoRAConfig(
            r=lora_cfg.r,
            alpha=lora_cfg.alpha,
            dropout=lora_cfg.dropout,
            targets=tuple(lora_cfg.targets),
        )
        if lora_cfg.enabled
        else None
    )
    cdc = (
        CDCConfig(
            bottleneck=cdc_cfg.bottleneck,
            kernel_size=cdc_cfg.kernel_size,
            theta=cdc_cfg.theta,
            activation=cdc_cfg.activation,
        )
        if cdc_cfg.enabled
        else None
    )
    return OSDFDModel(
        model_name=cfg.backbone.model_name,
        lora=lora,
        cdc=cdc,
        fsm_prob=cfg.fsm.prob if cfg.fsm.enabled else 0.0,
        fsm_alpha=cfg.fsm.alpha,
        feature_fusion=cfg.model.feature_fusion,
        head_hidden_dim=cfg.model.head_hidden_dim,
        head_dropout=cfg.model.head_dropout,
        freeze_backbone=cfg.backbone.freeze,
        train_norm=cfg.model.train_norm,
    )


class OSDFDLightningModule(L.LightningModule):
    """LightningModule wrapping the OSDFD model and objective."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.model = build_model(cfg)
        self.loss_fn = OSDFDLoss(
            scl_weight=cfg.loss.scl_weight,
            scl_margin=cfg.loss.scl_margin,
            pos_weight=cfg.loss.pos_weight,
        )

        # Per-epoch score/label buffers for eval metrics.
        self._val_scores: list[np.ndarray] = []
        self._val_labels: list[np.ndarray] = []
        self._test_scores: list[np.ndarray] = []
        self._test_labels: list[np.ndarray] = []
        self._epoch_start = 0.0

    # ------------------------------------------------------------------ core
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Inference forward returning fake probabilities (FSM disabled)."""
        out = self.model(pixel_values, apply_fsm=False)
        return torch.sigmoid(out.logits)

    def on_fit_start(self) -> None:
        n_train = self.model.num_trainable_parameters()
        n_total = self.model.num_total_parameters()
        self.print(
            f"[OSDFD] Trainable params: {n_train/1e6:.3f}M / {n_total/1e6:.1f}M "
            f"({100*n_train/n_total:.2f}%)"
        )
        # self.log() is disallowed in on_fit_start; log straight to the loggers.
        metrics = {"params/trainable_M": n_train / 1e6, "params/total_M": n_total / 1e6}
        for logger in self.loggers:
            logger.log_metrics(metrics, step=0)

    def on_train_epoch_start(self) -> None:
        self._epoch_start = time.time()

    # --------------------------------------------------------------- training
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        out = self.model(
            batch["pixel_values"],
            is_fake=batch["label"].bool(),
            domains=batch["domain"],
            apply_fsm=True,
        )
        loss, parts = self.loss_fn(out.logits, out.scl_features, batch["label"])
        bs = batch["label"].size(0)
        self.log("train/loss", parts["total"], prog_bar=True, batch_size=bs)
        self.log("train/bce", parts["bce"], batch_size=bs)
        self.log("train/scl", parts["scl"], batch_size=bs)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True, batch_size=bs)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train/epoch_time_s", time.time() - self._epoch_start)
        if torch.cuda.is_available():
            self.log("gpu/mem_alloc_GB", torch.cuda.max_memory_allocated() / 1e9)
            torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------- evaluation
    def _eval_step(self, batch: dict, scores: list, labels: list) -> torch.Tensor:
        out = self.model(batch["pixel_values"], apply_fsm=False)
        loss, parts = self.loss_fn(out.logits, out.scl_features, batch["label"])
        scores.append(torch.sigmoid(out.logits).detach().float().cpu().numpy())
        labels.append(batch["label"].detach().cpu().numpy())
        return parts["total"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss = self._eval_step(batch, self._val_scores, self._val_labels)
        self.log("val/loss", loss, prog_bar=True, batch_size=batch["label"].size(0))

    def on_validation_epoch_end(self) -> None:
        self._finalise_eval(self._val_scores, self._val_labels, "val")
        self._val_scores.clear()
        self._val_labels.clear()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self._eval_step(batch, self._test_scores, self._test_labels)

    def on_test_epoch_end(self) -> None:
        self._finalise_eval(self._test_scores, self._test_labels, "test", figures=True)
        self._test_scores.clear()
        self._test_labels.clear()

    def _finalise_eval(self, scores: list, labels: list, prefix: str, figures: bool = False) -> None:
        if not scores:
            return
        y_score = np.concatenate(scores)
        y_true = np.concatenate(labels)
        # Under DDP each rank only sees its shard; gather all predictions so
        # AUC/EER are computed over the full split (DistributedSampler pads
        # ranks to equal length, so all_gather shapes match).
        if self.trainer is not None and self.trainer.world_size > 1:
            y_score = self.all_gather(
                torch.from_numpy(y_score).to(self.device)
            ).flatten().cpu().numpy()
            y_true = self.all_gather(
                torch.from_numpy(y_true).to(self.device)
            ).flatten().cpu().numpy()
        metrics = compute_metrics(y_true, y_score)
        # AUC is the primary validation metric (paper Sec. IV-A).
        self.log_dict(
            {f"{prefix}/{k}": v for k, v in metrics.items()},
            prog_bar=(prefix == "val"),
        )
        if figures:
            # `self.logger` only returns the first logger; send the figures to
            # every attached logger (TensorBoard *and* W&B).
            for logger in self.loggers:
                log_figures(logger, y_true, y_score, self.global_step, prefix)

    # -------------------------------------------------------------- inference
    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> dict:
        out = self.model(batch["pixel_values"], apply_fsm=False)
        probs = torch.sigmoid(out.logits)
        result = {
            "path": batch.get("path"),
            "logit": out.logits.detach().cpu(),
            "prob": probs.detach().cpu(),
            "pred": (probs >= 0.5).long().detach().cpu(),
        }
        if "label" in batch:
            result["label"] = batch["label"].detach().cpu()
        return result

    # ------------------------------------------------------------- optimizers
    def configure_optimizers(self):
        oc = self.cfg.optimizer
        params = list(self.model.trainable_parameters())
        optimizer = torch.optim.Adam(
            params,
            lr=oc.lr,
            betas=(oc.beta1, oc.beta2),
            weight_decay=oc.weight_decay,
        )

        sc = self.cfg.scheduler
        if sc.name == "none":
            return optimizer
        if sc.name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=sc.t_max, eta_min=sc.eta_min
            )
        elif sc.name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=sc.step_size, gamma=sc.gamma
            )
        else:
            raise ValueError(f"Unknown scheduler: {sc.name}")
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": sc.interval}}
