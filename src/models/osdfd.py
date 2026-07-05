"""OSDFD model: SigLIP 2 backbone + forgery-aware PEFT + FSM + head.

Assembles the full Open-Set Deepfake Detection model of the OSDFD paper on top
of a SigLIP 2 visual encoder (Sec. III-A, Fig. 4a):

    pixel_values
        -> SigLIP 2 encoder (frozen) + LoRA + CDC adapters   (patch tokens)
        -> Forgery Style Mixture   (train only, fake samples)
        -> MAP pooling head         (global embedding)
        -> optional global+local feature fusion
        -> MLP classifier           (logit, SCL feature)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .backbone import Siglip2Backbone
from .fsm import ForgeryStyleMixture
from .head import ClassifierHead
from .peft_inject import CDCConfig, LoRAConfig, inject_peft, mark_trainable


@dataclass
class OSDFDOutput:
    """Structured forward output."""

    logits: torch.Tensor          # (B,) raw forgery logits
    scl_features: torch.Tensor    # (B, hidden) penultimate features for SCL
    pooled: torch.Tensor          # (B, D) MAP-pooled global embedding


class OSDFDModel(nn.Module):
    """Full OSDFD detector.

    Args:
        model_name: SigLIP 2 checkpoint id.
        lora: LoRA config (``None`` disables LoRA).
        cdc: CDC-adapter config (``None`` disables the adapter).
        fsm_prob: FSM activation probability (``0`` disables FSM).
        fsm_alpha: ``Beta(alpha, alpha)`` parameter for FSM mixing.
        feature_fusion: ``"global"`` (MAP pooled only) or ``"global_local"``
            (concatenate MAP pooled with mean-pooled patch tokens).
        head_hidden_dim: Width of the classifier's penultimate (SCL) layer.
        head_dropout: Dropout in the classifier head.
        freeze_backbone: Freeze SigLIP 2 weights (paper default: True).
        train_norm: Also train LayerNorm affine params (optional).
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        lora: LoRAConfig | None = None,
        cdc: CDCConfig | None = None,
        fsm_prob: float = 0.5,
        fsm_alpha: float = 0.1,
        feature_fusion: str = "global",
        head_hidden_dim: int = 256,
        head_dropout: float = 0.0,
        freeze_backbone: bool = True,
        train_norm: bool = False,
        pretrained: bool = True,
        backbone_config_overrides: dict | None = None,
    ) -> None:
        super().__init__()
        if feature_fusion not in ("global", "global_local"):
            raise ValueError(f"Unknown feature_fusion: {feature_fusion}")

        self.feature_fusion = feature_fusion

        self.backbone = Siglip2Backbone(
            model_name=model_name,
            freeze=freeze_backbone,
            pretrained=pretrained,
            config_overrides=backbone_config_overrides,
        )
        inject_peft(self.backbone, lora=lora, cdc=cdc)

        self.fsm = ForgeryStyleMixture(prob=fsm_prob, alpha=fsm_alpha)

        d = self.backbone.hidden_size
        in_dim = d * 2 if feature_fusion == "global_local" else d
        self.head = ClassifierHead(in_dim, hidden_dim=head_hidden_dim, dropout=head_dropout)

        if freeze_backbone:
            mark_trainable(self, train_norm=train_norm)

    def forward(
        self,
        pixel_values: torch.Tensor,
        is_fake: torch.Tensor | None = None,
        domains: torch.Tensor | None = None,
        apply_fsm: bool = True,
    ) -> OSDFDOutput:
        """Forward pass.

        Args:
            pixel_values: Normalised images ``(B, 3, H, W)``.
            is_fake: Boolean mask ``(B,)`` used by FSM (required if ``apply_fsm``).
            domains: Forgery-domain ids ``(B,)`` used by FSM.
            apply_fsm: Whether to run FSM (only has an effect in ``train`` mode
                and when ``is_fake`` / ``domains`` are provided).

        Returns:
            :class:`OSDFDOutput`.
        """
        patch_tokens = self.backbone(pixel_values)  # (B, N, D)

        if apply_fsm and is_fake is not None and domains is not None:
            patch_tokens = self.fsm(patch_tokens, is_fake=is_fake, domains=domains)

        pooled = self.backbone.pool(patch_tokens)  # (B, D)

        if self.feature_fusion == "global_local":
            local = patch_tokens.mean(dim=1)         # (B, D) local descriptor
            feat = torch.cat([pooled, local], dim=-1)
        else:
            feat = pooled

        logits, scl_features = self.head(feat)
        return OSDFDOutput(logits=logits, scl_features=scl_features, pooled=pooled)

    def trainable_parameters(self):
        """Iterator over parameters with ``requires_grad=True``."""
        return (p for p in self.parameters() if p.requires_grad)

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def num_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
