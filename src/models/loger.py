"""LOGER model: local-global ensemble deepfake detector with FSM.

Implements the LOGER architecture ("LOGER: Local-Global Ensemble for Robust
Deepfake Detection in the Wild", arXiv:2604.03558) on a swappable Vision
Foundation Model, with the Forgery Style Mixture module (OSDFD paper,
Sec. III-C) inserted between backbone feature extraction and the
classification heads:

    pixel_values
        -> VFM backbone (siglip2 / dinov2 / dinov3 / clip)   (token sequence)
        -> Forgery Style Mixture     (train only, fake samples, unchanged)
        -> global branch: pooled embedding -> MLP head -> d_global
        -> local branch:  patch tokens -> patch classifier (2 logits/patch)
                          -> d_i = l_i^fake - l_i^real
                          -> MIL top-k pooling, k = floor(0.1 * N) -> d_local
        -> logit fusion:  d = sum_m alpha_m * d_m  (uniform alpha by default)

The backbone itself is never modified: FSM operates on the token sequence it
returns, so both the pooled global feature and the patch tokens seen by the
local branch reflect the mixed forgery statistics.

PEFT modes (config ``peft.mode``): ``full`` fine-tuning, ``lora`` (frozen
backbone + LoRA on attention q/k/v) or ``frozen`` (heads only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_backbone, inject_lora
from .fsm import ForgeryStyleMixture

PEFT_MODES = ("full", "lora", "frozen")


@dataclass
class LOGEROutput:
    """Structured forward output (all logits are fake-minus-real differences)."""

    logits: torch.Tensor         # (B,) fused logit d = mean(d_global, d_local)
    global_logits: torch.Tensor  # (B,) global-branch logit d_global
    local_logits: torch.Tensor   # (B,) MIL top-k pooled local logit d_local
    patch_diffs: torch.Tensor    # (B, N) per-patch logit differences d_i
    scl_features: torch.Tensor   # (B, H) global-head penultimate features
    pooled: torch.Tensor         # (B, D) pooled global embedding


class ImageClassifier(nn.Module):
    """Image-level two-logit classifier of the LOGER global branch."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.act(self.fc1(self.dropout(x)))
        logits = self.fc2(feat)
        return logits[..., 1] - logits[..., 0], feat


class PatchClassifier(nn.Module):
    """Patch-level classifier of the LOGER local branch.

    Maps every patch token to two logits ``(l_real, l_fake)``; the per-patch
    forgery evidence is their difference ``d_i = l_i^fake - l_i^real``.

    Args:
        in_dim: Patch-token dimension.
        hidden_dim: Width of the hidden layer.
        dropout: Dropout before the hidden layer.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Return per-patch logit differences ``d_i`` of shape ``(B, N)``.

        Args:
            patch_tokens: Patch tokens ``(B, N, D)``.
        """
        logits = self.fc2(self.act(self.fc1(self.dropout(patch_tokens))))  # (B, N, 2)
        return logits[..., 1] - logits[..., 0]  # d_i = l_fake - l_real


def mil_topk_pool(patch_diffs: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    """MIL top-k pooling of per-patch logit differences (LOGER local branch).

    ``d_img = (1/k) * sum_{i in S_k} d_i`` with ``k = floor(ratio * N)``
    (clamped to at least 1), where ``S_k`` indexes the k largest ``d_i``.

    Args:
        patch_diffs: Per-patch logit differences ``(B, N)``.
        ratio: Fraction of patches to keep (paper: 0.1).

    Returns:
        Image-level logits ``(B,)``.
    """
    n = patch_diffs.size(1)
    k = max(1, math.floor(ratio * n))
    topk = patch_diffs.topk(k, dim=1).values
    return topk.mean(dim=1)


class LOGERModel(nn.Module):
    """LOGER dual-branch detector with train-only Forgery Style Mixture.

    Args:
        backbone_name: Backbone family (``siglip2`` / ``dinov2`` / ``dinov3`` /
            ``clip``), optionally suffixed (e.g. ``siglip2-base``).
        model_name: HuggingFace checkpoint id (``None`` = family default).
        peft_mode: ``full`` | ``lora`` | ``frozen``.
        lora_r / lora_alpha / lora_dropout: LoRA hyper-parameters (lora mode).
        fsm_prob: FSM activation probability (``0`` disables FSM).
        fsm_alpha: ``Beta(alpha, alpha)`` parameter for FSM mixing.
        topk_ratio: MIL top-k ratio (paper: 0.1 -> k = floor(0.1 * N)).
        head_hidden_dim: Hidden width of both branch heads.
        head_dropout: Dropout in both branch heads.
        fusion_weights: Branch weights ``(alpha_global, alpha_local)``;
            ``None`` = uniform ``alpha_m = 1/M`` (paper default).
        pretrained: Load pre-trained backbone weights.
        backbone_config_overrides: Tiny-model config for offline tests.
    """

    def __init__(
        self,
        backbone_name: str = "siglip2",
        model_name: str | None = None,
        peft_mode: str = "full",
        lora_r: int = 8,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        fsm_prob: float = 0.5,
        fsm_alpha: float = 0.1,
        topk_ratio: float = 0.1,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.0,
        fusion_weights: tuple[float, float] | None = None,
        pretrained: bool = True,
        backbone_config_overrides: dict | None = None,
    ) -> None:
        super().__init__()
        if peft_mode not in PEFT_MODES:
            raise ValueError(f"Unknown peft_mode: {peft_mode} (choose from {PEFT_MODES})")
        self.peft_mode = peft_mode
        self.topk_ratio = topk_ratio

        self.backbone = build_backbone(
            backbone_name,
            model_name=model_name,
            freeze=(peft_mode != "full"),
            pretrained=pretrained,
            config_overrides=backbone_config_overrides,
        )
        if peft_mode == "lora":
            inject_lora(self.backbone, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

        self.fsm = ForgeryStyleMixture(prob=fsm_prob, alpha=fsm_alpha)

        d = self.backbone.hidden_size
        self.global_head = ImageClassifier(d, hidden_dim=head_hidden_dim, dropout=head_dropout)
        self.local_head = PatchClassifier(d, hidden_dim=head_hidden_dim, dropout=head_dropout)

        if fusion_weights is not None:
            w = torch.tensor(list(fusion_weights), dtype=torch.float32)
        else:
            w = torch.full((2,), 0.5)  # uniform alpha_m = 1/M, M = 2 branches
        self.register_buffer("fusion_weights", w)

    def forward(
        self,
        pixel_values: torch.Tensor,
        is_fake: torch.Tensor | None = None,
        domains: torch.Tensor | None = None,
        apply_fsm: bool = True,
    ) -> LOGEROutput:
        """Forward pass.

        Args:
            pixel_values: Normalised images ``(B, 3, H, W)``.
            is_fake: Boolean mask ``(B,)`` used by FSM (required if ``apply_fsm``).
            domains: Forgery-domain ids ``(B,)`` used by FSM.
            apply_fsm: Whether to run FSM (train mode only; no-op otherwise).
        """
        tokens = self.backbone(pixel_values)  # (B, T, D)

        # FSM: after backbone feature extraction, before every classifier head.
        if apply_fsm and is_fake is not None and domains is not None:
            tokens = self.fsm(tokens, is_fake=is_fake, domains=domains)

        # Global branch: pooled embedding -> MLP head -> d_global.
        pooled = self.backbone.pool(tokens)
        global_logits, scl_features = self.global_head(pooled)

        # Local branch: patch classifier -> d_i -> MIL top-k -> d_local.
        patch_diffs = self.local_head(self.backbone.patches(tokens))
        local_logits = mil_topk_pool(patch_diffs, self.topk_ratio)

        # Logit-space fusion: d = alpha_g * d_global + alpha_l * d_local.
        w = self.fusion_weights
        logits = w[0] * global_logits + w[1] * local_logits

        return LOGEROutput(
            logits=logits,
            global_logits=global_logits,
            local_logits=local_logits,
            patch_diffs=patch_diffs,
            scl_features=scl_features,
            pooled=pooled,
        )

    @torch.no_grad()
    def forward_multi_resolution(
        self,
        pixel_values: torch.Tensor,
        resolutions: list[int],
    ) -> torch.Tensor:
        """Multi-resolution inference (LOGER global-branch strategy).

        Resizes the batch to each resolution, runs the full model (FSM off)
        and averages the fused logits.

        Args:
            pixel_values: Normalised images ``(B, 3, H, W)``.
            resolutions: Square input resolutions to average over.

        Returns:
            Averaged fused logits ``(B,)``.
        """
        logits = []
        for res in resolutions:
            x = pixel_values
            if x.shape[-1] != res:
                x = F.interpolate(x, size=(res, res), mode="bicubic", align_corners=False)
            logits.append(self(x, apply_fsm=False).logits)
        return torch.stack(logits).mean(dim=0)

    def trainable_parameters(self):
        """Iterator over parameters with ``requires_grad=True``."""
        return (p for p in self.parameters() if p.requires_grad)

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def num_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
