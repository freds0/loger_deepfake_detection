"""LOGER training objective: global/fused BCE + local (CE + AUC + MIL + reg).

The LOGER paper trains the local branch with

    L_local = L_CE + 0.5 * L_AUC + 0.5 * L_MIL + L_reg

on top of the standard image-level supervision of the global branch. The paper
does not spell out closed forms for the AUC / MIL / regularisation terms, so
this module implements standard formulations, each as an independent
``nn.Module`` with a config-controlled weight:

  * ``L_CE``  -- BCE-with-logits on the MIL-pooled local logit ``d_local``.
  * ``L_AUC`` -- pairwise squared-hinge AUC surrogate on ``d_local``: every
    (fake, real) pair in the batch should be separated by a margin.
  * ``L_MIL`` -- patch-level multiple-instance supervision on the ``d_i``:
    real images contain no forged patches (all ``d_i`` pushed real), fake
    images contain at least ``k`` forged patches (top-k ``d_i`` pushed fake).
  * ``L_reg`` -- L2 penalty on the per-patch logits, preventing the patch
    scores from exploding under the hinge/MIL pressure.

The global branch and the fused ensemble logit are supervised with plain BCE.
All weights are exposed through the ``loss`` config group.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AUCSurrogateLoss(nn.Module):
    """Pairwise squared-hinge surrogate of the AUC on image-level logits.

    For every (fake, real) pair ``(i, j)`` in the batch the fake score should
    exceed the real score by ``margin``:

        L = mean_{i,j} max(0, margin - (s_i - s_j))^2

    Args:
        margin: Required logit separation between fake and real samples.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Args: image-level logits ``(B,)`` and binary labels ``(B,)``."""
        fake = logits[labels.bool()]
        real = logits[~labels.bool()]
        if fake.numel() == 0 or real.numel() == 0:
            return logits.new_zeros(())
        diff = fake.unsqueeze(1) - real.unsqueeze(0)  # (F, R) pairwise s_i - s_j
        return F.relu(self.margin - diff).pow(2).mean()


class MILLoss(nn.Module):
    """Patch-level multiple-instance supervision on per-patch logits.

    MIL assumption: a real image contains only real patches; a fake image
    contains at least ``k = floor(ratio * N)`` forged patches (the manipulated
    face region). Hence:

      * real images: BCE of *all* patch logits toward 0,
      * fake images: BCE of the *top-k* patch logits toward 1.

    Args:
        topk_ratio: Fraction of patches assumed forged in fake images (must
            match the model's MIL pooling ratio).
    """

    def __init__(self, topk_ratio: float = 0.1) -> None:
        super().__init__()
        self.topk_ratio = topk_ratio

    def forward(self, patch_diffs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Args: per-patch logits ``(B, N)`` and binary labels ``(B,)``."""
        fake_mask = labels.bool()
        terms = []
        if (~fake_mask).any():
            real_patches = patch_diffs[~fake_mask]
            terms.append(F.binary_cross_entropy_with_logits(
                real_patches, torch.zeros_like(real_patches)))
        if fake_mask.any():
            n = patch_diffs.size(1)
            k = max(1, math.floor(self.topk_ratio * n))
            topk = patch_diffs[fake_mask].topk(k, dim=1).values
            terms.append(F.binary_cross_entropy_with_logits(
                topk, torch.ones_like(topk)))
        if not terms:  # pragma: no cover - empty batch
            return patch_diffs.new_zeros(())
        return torch.stack(terms).mean()


class PatchLogitRegularization(nn.Module):
    """L2 penalty on per-patch logits: ``L = mean(d_i^2)``."""

    def forward(self, patch_diffs: torch.Tensor) -> torch.Tensor:
        return patch_diffs.pow(2).mean()


class LOGERLoss(nn.Module):
    """Full LOGER objective with independently weighted components.

    ``L = w_global * BCE(d_global) + w_fused * BCE(d)``
    ``  + w_ce * BCE(d_local) + w_auc * L_AUC + w_mil * L_MIL + w_reg * L_reg``

    Args:
        global_weight: Weight on the global-branch BCE.
        fused_weight: Weight on the fused-logit BCE.
        ce_weight: Weight on the local-branch image-level BCE (``L_CE``).
        auc_weight: Weight on the AUC surrogate (paper: 0.5).
        mil_weight: Weight on the patch-level MIL loss (paper: 0.5).
        reg_weight: Weight on the patch-logit L2 regulariser.
        auc_margin: Margin of the AUC surrogate.
        topk_ratio: MIL top-k ratio (must match the model).
        pos_weight: Optional positive-class weight for all BCE terms.
    """

    def __init__(
        self,
        global_weight: float = 1.0,
        fused_weight: float = 1.0,
        ce_weight: float = 1.0,
        auc_weight: float = 0.5,
        mil_weight: float = 0.5,
        reg_weight: float = 1e-4,
        auc_margin: float = 1.0,
        topk_ratio: float = 0.1,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.global_weight = global_weight
        self.fused_weight = fused_weight
        self.ce_weight = ce_weight
        self.auc_weight = auc_weight
        self.mil_weight = mil_weight
        self.reg_weight = reg_weight

        pw = torch.tensor(pos_weight) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.auc = AUCSurrogateLoss(margin=auc_margin)
        self.mil = MILLoss(topk_ratio=topk_ratio)
        self.reg = PatchLogitRegularization()

    def forward(
        self,
        fused_logits: torch.Tensor,
        global_logits: torch.Tensor,
        local_logits: torch.Tensor,
        patch_diffs: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the total loss and its detached components.

        Args:
            fused_logits: Fused ensemble logits ``(B,)``.
            global_logits: Global-branch logits ``(B,)``.
            local_logits: MIL-pooled local logits ``(B,)``.
            patch_diffs: Per-patch logits ``(B, N)``.
            labels: Binary labels ``(B,)`` (1 = fake, 0 = real).

        Returns:
            ``(total_loss, parts)`` with detached scalars for logging.
        """
        labels_f = labels.to(fused_logits.dtype)

        bce_global = self.bce(global_logits, labels_f)
        bce_fused = self.bce(fused_logits, labels_f)
        ce_local = self.bce(local_logits, labels_f)
        auc = self.auc(local_logits, labels)
        mil = self.mil(patch_diffs, labels)
        reg = self.reg(patch_diffs)

        total = (
            self.global_weight * bce_global
            + self.fused_weight * bce_fused
            + self.ce_weight * ce_local
            + self.auc_weight * auc
            + self.mil_weight * mil
            + self.reg_weight * reg
        )
        parts = {
            "global_bce": bce_global.detach(),
            "fused_bce": bce_fused.detach(),
            "local_ce": ce_local.detach(),
            "local_auc": auc.detach(),
            "local_mil": mil.detach(),
            "local_reg": reg.detach(),
            "total": total.detach(),
        }
        return total, parts
