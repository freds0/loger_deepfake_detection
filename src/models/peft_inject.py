"""Inject forgery-aware PEFT modules into a SigLIP 2 encoder.

Realises the "Forgery-aware PEFT" design of Sec. III-B / Fig. 4 of the OSDFD
paper on top of the SigLIP 2 vision transformer:

  * LoRA is injected into the query / key / value projections of every
    self-attention block (Eqs. 4-5).
  * A CDC adapter is added in parallel to every FFN (Eqs. 1-2), turning the
    block output into ``MLP(h) + Adapter(h)``.

Only these injected modules are trainable; the SigLIP 2 weights stay frozen.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from .backbone import Siglip2Backbone
from .cdc_adapter import AdapterMLP, CDCAdapter
from .lora import LoRALinear


@dataclass
class LoRAConfig:
    """LoRA hyper-parameters (paper: rank r defaults to 8 for ViT-B, d=768)."""

    r: int = 8
    alpha: float = 8.0
    dropout: float = 0.0
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj")


@dataclass
class CDCConfig:
    """CDC adapter hyper-parameters (Fig. 4c / Fig. 5)."""

    bottleneck: int = 64
    kernel_size: int = 3
    theta: float = 0.7
    activation: bool = True


def inject_peft(
    backbone: Siglip2Backbone,
    lora: LoRAConfig | None = None,
    cdc: CDCConfig | None = None,
) -> Siglip2Backbone:
    """Insert LoRA and/or CDC-adapter modules into every encoder block.

    Args:
        backbone: A (frozen) :class:`Siglip2Backbone`.
        lora: LoRA configuration, or ``None`` to skip LoRA injection.
        cdc: CDC-adapter configuration, or ``None`` to skip adapter injection.

    Returns:
        The same backbone, modified in place.
    """
    dim = backbone.hidden_size

    for layer in backbone.encoder_layers:
        if lora is not None and lora.r > 0:
            attn = layer.self_attn
            for name in lora.targets:
                base = getattr(attn, name)
                setattr(
                    attn,
                    name,
                    LoRALinear(base, r=lora.r, alpha=lora.alpha, dropout=lora.dropout),
                )

        if cdc is not None:
            adapter = CDCAdapter(
                dim=dim,
                bottleneck=cdc.bottleneck,
                kernel_size=cdc.kernel_size,
                theta=cdc.theta,
                activation=cdc.activation,
            )
            layer.mlp = AdapterMLP(layer.mlp, adapter)

    return backbone


def mark_trainable(model: nn.Module, train_norm: bool = False) -> None:
    """Ensure only PEFT modules, the head and (optionally) norms are trainable.

    Enforces the paper's training regime: only LoRA, the CDC adapter, the
    classification head and (optionally) normalisation layers are optimised.
    The module constructors already set the correct grad state (LoRA keeps its
    frozen ``base``; the adapter/head are trainable); this only re-asserts the
    PEFT branches and handles the optional ``train_norm`` flag.

    Args:
        model: The full OSDFD model.
        train_norm: If True, LayerNorm affine parameters are also trainable.
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            # Only the low-rank branch is trainable; ``base`` stays frozen.
            if module.lora_down is not None:
                module.lora_down.weight.requires_grad_(True)
                module.lora_up.weight.requires_grad_(True)
        elif isinstance(module, CDCAdapter):
            for p in module.parameters():
                p.requires_grad_(True)

    if train_norm:
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                for p in module.parameters():
                    p.requires_grad_(True)
