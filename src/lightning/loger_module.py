"""LightningModule for LOGER + FSM training / validation / testing.

Mirrors :class:`src.lightning.module.OSDFDLightningModule` but wraps the
dual-branch :class:`~src.models.loger.LOGERModel` and the
:class:`~src.losses.loger.LOGERLoss` objective. The Forgery Style Mixture
module runs only in ``training_step``; validation / test / predict always
disable it. Adds AdamW with linear warmup, gradient-norm logging and optional
multi-resolution evaluation (LOGER global-branch strategy).
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from ..losses.loger import LOGERLoss
from ..models.loger import LOGERModel
from ..training.metrics import best_accuracy_threshold, compute_metrics, log_figures


def _select(cfg: DictConfig, key: str, default=None):
    value = OmegaConf.select(cfg, key)
    return default if value is None else value


def build_loger_model(cfg: DictConfig) -> LOGERModel:
    """Construct a :class:`LOGERModel` from the Hydra config groups."""
    fusion = cfg.model.fusion_weights
    fsm_prob = _select(cfg, "fsm.probability", _select(cfg, "fsm.prob", 0.5))
    fsm_alpha = _select(cfg, "fsm.beta_alpha", _select(cfg, "fsm.alpha", 0.1))
    backbone_overrides = _select(cfg, "backbone.config_overrides", None)
    return LOGERModel(
        backbone_name=_select(cfg, "backbone.name", "siglip2-base"),
        model_name=cfg.backbone.model_name,
        peft_mode=_select(cfg, "peft.mode", "full"),
        lora_r=cfg.peft.lora.r,
        lora_alpha=cfg.peft.lora.alpha,
        lora_dropout=cfg.peft.lora.dropout,
        fsm_prob=fsm_prob if cfg.fsm.enabled else 0.0,
        fsm_alpha=fsm_alpha,
        topk_ratio=cfg.model.topk_ratio,
        head_hidden_dim=cfg.model.head_hidden_dim,
        head_dropout=cfg.model.head_dropout,
        fusion_weights=tuple(fusion) if fusion is not None else None,
        learnable_fusion=_select(cfg, "model.learnable_fusion", False),
        pretrained=_select(cfg, "backbone.pretrained", True),
        backbone_config_overrides=backbone_overrides,
        gradient_checkpointing=_select(cfg, "backbone.gradient_checkpointing", False),
    )


class LOGERLightningModule(L.LightningModule):
    """LightningModule wrapping the LOGER model and objective."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.model = build_loger_model(cfg)
        lc = cfg.loss
        self.loss_fn = LOGERLoss(
            global_weight=lc.global_weight,
            fused_weight=lc.fused_weight,
            ce_weight=lc.ce_weight,
            auc_weight=lc.auc_weight,
            mil_weight=lc.mil_weight,
            reg_weight=lc.reg_weight,
            auc_margin=lc.auc_margin,
            topk_ratio=cfg.model.topk_ratio,
            pos_weight=lc.pos_weight,
        )
        # Multi-resolution evaluation (null/empty = single resolution).
        self.eval_resolutions = list(cfg.model.eval_resolutions or [])
        # Horizontal-flip TTA at predict time only (see predict_step).
        self.tta_hflip = bool(_select(cfg, "model.tta_hflip", False))

        # Per-epoch score/label/path buffers for eval metrics.
        self._val_scores: list[np.ndarray] = []
        self._val_labels: list[np.ndarray] = []
        self._val_paths: list[list[str]] = []
        self._test_scores: list[np.ndarray] = []
        self._test_labels: list[np.ndarray] = []
        self._test_paths: list[list[str]] = []
        self._epoch_start = 0.0

        # Accuracy-maximising decision threshold, calibrated on val each
        # epoch (see _finalise_eval) and used by predict_step instead of a
        # fixed 0.5. Persisted across checkpoints (on_save/on_load_checkpoint).
        self.best_threshold: float = 0.5

    # ------------------------------------------------------------------ core
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Inference forward returning fake probabilities (FSM disabled)."""
        return torch.sigmoid(self._eval_logits(pixel_values))

    def _eval_logits(
        self,
        pixel_values: torch.Tensor,
        precomputed: tuple[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Fused logits with FSM off, multi-resolution if configured.

        Args:
            precomputed: Optional ``(resolution, logits)`` already computed at
                the input's native resolution (see ``_eval_step``), so
                ``forward_multi_resolution`` doesn't run that resolution twice.
        """
        if len(self.eval_resolutions) > 1:
            return self.model.forward_multi_resolution(
                pixel_values, self.eval_resolutions, precomputed=precomputed
            )
        return self.model(pixel_values, apply_fsm=False).logits

    def on_fit_start(self) -> None:
        n_train = self.model.num_trainable_parameters()
        n_total = self.model.num_total_parameters()
        self.print(
            f"[LOGER] Trainable params: {n_train/1e6:.3f}M / {n_total/1e6:.1f}M "
            f"({100*n_train/n_total:.2f}%) | peft={self.model.peft_mode}"
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
        loss, parts = self.loss_fn(
            out.logits, out.global_logits, out.local_logits, out.patch_diffs, batch["label"]
        )
        bs = batch["label"].size(0)
        self.log("train/loss", parts["total"], prog_bar=True, batch_size=bs)
        for name, value in parts.items():
            if name != "total":
                self.log(f"train/{name}", value, batch_size=bs)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True, batch_size=bs)
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        # Total L2 gradient norm over trainable parameters (pre-clipping).
        grads = [p.grad.norm(2) for p in self.model.parameters() if p.grad is not None]
        if grads:
            self.log("train/grad_norm", torch.stack(grads).norm(2))

    def on_train_epoch_end(self) -> None:
        self.log("train/epoch_time_s", time.time() - self._epoch_start)
        if torch.cuda.is_available():
            self.log("gpu/mem_alloc_GB", torch.cuda.max_memory_allocated() / 1e9)
            torch.cuda.reset_peak_memory_stats()
        if self.model.learnable_fusion:
            w = torch.softmax(self.model.fusion_logits.detach(), dim=0)
            self.log("fusion/w_global", w[0])
            self.log("fusion/w_local", w[1])

    # ------------------------------------------------------------- evaluation
    def _eval_step(
        self, batch: dict, scores: list, labels: list, paths: list | None = None
    ) -> torch.Tensor:
        out = self.model(batch["pixel_values"], apply_fsm=False)
        loss, _ = self.loss_fn(
            out.logits, out.global_logits, out.local_logits, out.patch_diffs, batch["label"]
        )
        logits = (
            self._eval_logits(
                batch["pixel_values"],
                precomputed=(batch["pixel_values"].shape[-1], out.logits),
            )
            if len(self.eval_resolutions) > 1
            else out.logits
        )
        scores.append(torch.sigmoid(logits).detach().float().cpu().numpy())
        labels.append(batch["label"].detach().cpu().numpy())
        if paths is not None:
            paths.append(batch.get("path") or [])
        return loss.detach()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss = self._eval_step(batch, self._val_scores, self._val_labels, self._val_paths)
        self.log("val/loss", loss, prog_bar=True, batch_size=batch["label"].size(0))

    def on_validation_epoch_end(self) -> None:
        self._finalise_eval(self._val_scores, self._val_labels, "val", paths=self._val_paths)
        self._val_scores.clear()
        self._val_labels.clear()
        self._val_paths.clear()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self._eval_step(batch, self._test_scores, self._test_labels, self._test_paths)

    def on_test_epoch_end(self) -> None:
        self._finalise_eval(
            self._test_scores, self._test_labels, "test", figures=True, paths=self._test_paths
        )
        self._test_scores.clear()
        self._test_labels.clear()
        self._test_paths.clear()

    def _log_per_group_metrics(
        self, y_true: np.ndarray, y_score: np.ndarray, paths: list[str], prefix: str
    ) -> None:
        """Break AUC/acc down by the parent-of-parent directory of each path.

        For the NTIRE shard layout (``<root>/shard_N/images/<name>``) this is
        the shard; for the FF++ folder layout
        (``<root>/<split>/<manipulation>/...``) it's the manipulation type.
        Either way it surfaces subsets the model does poorly on that the
        pooled metric hides. Skips groups too small or single-class to give a
        meaningful AUC (min 50 samples, both classes present).
        """
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, path in enumerate(paths):
            groups[Path(path).parent.parent.name].append(idx)
        for group, idxs in sorted(groups.items()):
            if len(idxs) < 50:
                continue
            group_true = y_true[idxs]
            if len(np.unique(group_true)) < 2:
                continue
            group_metrics = compute_metrics(group_true, y_score[idxs])
            self.log(f"{prefix}/auc/{group}", group_metrics["auc"])
            self.log(f"{prefix}/acc/{group}", group_metrics["acc"])

    def _finalise_eval(
        self,
        scores: list,
        labels: list,
        prefix: str,
        figures: bool = False,
        paths: list[list[str]] | None = None,
    ) -> None:
        if not scores:
            return
        y_score = np.concatenate(scores)
        y_true = np.concatenate(labels)
        is_distributed = self.trainer is not None and self.trainer.world_size > 1
        # Under DDP each rank only sees its shard; gather all predictions so
        # AUC/EER are computed over the full split (DistributedSampler pads
        # ranks to equal length, so all_gather shapes match).
        if is_distributed:
            y_score = self.all_gather(
                torch.from_numpy(y_score).to(self.device)
            ).flatten().cpu().numpy()
            y_true = self.all_gather(
                torch.from_numpy(y_true).to(self.device)
            ).flatten().cpu().numpy()
        metrics = compute_metrics(y_true, y_score)
        # AUC is the primary validation metric.
        self.log_dict(
            {f"{prefix}/{k}": v for k, v in metrics.items()},
            prog_bar=(prefix == "val"),
        )
        if prefix == "val" and len(np.unique(y_true)) > 1:
            threshold, best_acc = best_accuracy_threshold(y_true, y_score)
            self.best_threshold = threshold
            self.log("val/best_threshold", threshold)
            self.log("val/best_acc", best_acc)
        # Per-group breakdown needs `paths` aligned index-for-index with
        # y_true/y_score; all_gather above reshuffles that alignment (and
        # gathering the path strings themselves isn't worth the complexity),
        # so this is skipped entirely under DDP rather than logged skewed.
        if paths and not is_distributed:
            flat_paths = [p for batch_paths in paths for p in batch_paths]
            if len(flat_paths) == len(y_true):
                self._log_per_group_metrics(y_true, y_score, flat_paths, prefix)
        if figures:
            for logger in self.loggers:
                log_figures(logger, y_true, y_score, self.global_step, prefix)

    # -------------------------------------------------------------- inference
    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> dict:
        x = batch["pixel_values"]
        logits = self._eval_logits(x)
        if self.tta_hflip:
            flipped_logits = self._eval_logits(torch.flip(x, dims=[-1]))
            logits = 0.5 * (logits + flipped_logits)
        probs = torch.sigmoid(logits)
        threshold = getattr(self, "best_threshold", 0.5)
        result = {
            "path": batch.get("path"),
            "logit": logits.detach().cpu(),
            "prob": probs.detach().cpu(),
            "pred": (probs >= threshold).long().detach().cpu(),
        }
        if "label" in batch:
            result["label"] = batch["label"].detach().cpu()
        return result

    # ------------------------------------------------------- checkpoint state
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["best_threshold"] = self.best_threshold

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.best_threshold = checkpoint.get("best_threshold", 0.5)

    # ------------------------------------------------------------- optimizers
    def configure_optimizers(self):
        oc = self.cfg.optimizer
        params = list(self.model.trainable_parameters())
        if oc.name == "adamw":
            optimizer = torch.optim.AdamW(
                params, lr=oc.lr, betas=(oc.beta1, oc.beta2), weight_decay=oc.weight_decay
            )
        elif oc.name == "adam":
            optimizer = torch.optim.Adam(
                params, lr=oc.lr, betas=(oc.beta1, oc.beta2), weight_decay=oc.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {oc.name}")

        sc = self.cfg.scheduler
        warmup_steps = int(getattr(sc, "warmup_steps", 0) or 0)

        if sc.name == "none" and warmup_steps == 0:
            return optimizer
        if sc.name == "cosine":
            main = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=sc.t_max, eta_min=sc.eta_min
            )
            interval = sc.interval
        elif sc.name == "step":
            main = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=sc.step_size, gamma=sc.gamma
            )
            interval = sc.interval
        elif sc.name == "none":
            main = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)
            interval = "step"
        else:
            raise ValueError(f"Unknown scheduler: {sc.name}")

        if warmup_steps > 0:
            # Linear warmup, then the main schedule. Warmup counts optimizer
            # steps, so force step-wise scheduling for the combined schedule.
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup, main], milestones=[warmup_steps]
            )
            interval = "step"
        else:
            scheduler = main
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": interval},
        }
