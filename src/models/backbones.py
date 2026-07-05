"""Modular vision-backbone abstraction for the LOGER + FSM pipeline.

LOGER (arXiv:2604.03558) builds its global and local branches on top of
heterogeneous Vision Foundation Models. This module exposes a single interface
so the backbone is swappable through config (``backbone.name``):

  * ``siglip2``  -- SigLIP 2 vision tower (reuses :class:`Siglip2Backbone`).
  * ``dinov2``   -- DINOv2 (``facebook/dinov2-*``).
  * ``dinov3``   -- DINOv3 (requires a transformers version that ships it).
  * ``clip``     -- CLIP vision tower (``openai/clip-vit-*``).

Every backbone implements the same contract:

    tokens = backbone(pixel_values)      # (B, T, D) full token sequence
    pooled = backbone.pool(tokens)       # (B, D)    global image embedding
    patches = backbone.patches(tokens)   # (B, N, D) patch tokens only

The Forgery Style Mixture module operates on the *full* token sequence
(returned by ``forward``) so that both the pooled global feature and the patch
tokens seen by the local branch reflect the mixed forgery statistics -- i.e.
FSM sits after backbone feature extraction and before every classification
head, and the backbone itself is never modified.

``pool``/``patches`` are pure post-processing of the token sequence:

  * SigLIP 2 has no ``[CLS]`` token; ``pool`` applies the pre-trained MAP
    (attention pooling) head and ``patches`` is the identity.
  * DINOv2 / DINOv3 / CLIP carry a ``[CLS]`` token (DINOv3 also register
    tokens); ``pool`` selects it and ``patches`` drops the non-patch prefix.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import Siglip2Backbone
from .lora import LoRALinear

# Per-family normalisation statistics (used by the data transforms).
SIGLIP_STATS = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
IMAGENET_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP_STATS = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

# Linear-projection name suffixes wrapped by generic LoRA injection.
_LORA_TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj",           # SigLIP / CLIP / DINOv3 attention
    "attention.query", "attention.key", "attention.value",  # DINOv2 attention
)


class SiglipLOGERBackbone(Siglip2Backbone):
    """SigLIP 2 with the LOGER token-sequence contract (no ``[CLS]`` token)."""

    num_prefix_tokens = 0

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return post-layernorm tokens ``(B, N, D)``, any square resolution.

        Extends :meth:`Siglip2Backbone.forward` with positional-embedding
        interpolation so the LOGER global branch can run multi-resolution
        inference on a fixed-resolution SigLIP 2 checkpoint.
        """
        vm = self.vision_model
        interpolate = pixel_values.shape[-1] != vm.config.image_size
        hidden_states = vm.embeddings(pixel_values, interpolate_pos_encoding=interpolate)
        encoder_outputs = vm.encoder(inputs_embeds=hidden_states)
        return vm.post_layernorm(encoder_outputs.last_hidden_state)

    def patches(self, tokens: torch.Tensor) -> torch.Tensor:
        """All tokens are patch tokens for SigLIP 2."""
        return tokens

    # ``pool`` (MAP head) is inherited unchanged.


class Dinov2Backbone(nn.Module):
    """DINOv2 encoder exposing the ``[CLS]`` token and patch tokens.

    Args:
        model_name: HuggingFace checkpoint id (e.g. ``facebook/dinov2-base``).
        freeze: Freeze all backbone parameters.
        pretrained: Load pre-trained weights; if False build a random tiny
            model from ``config_overrides`` (offline tests).
        config_overrides: ``Dinov2Config`` kwargs for the non-pretrained path.
    """

    num_prefix_tokens = 1  # [CLS]

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        freeze: bool = True,
        pretrained: bool = True,
        config_overrides: dict | None = None,
    ) -> None:
        super().__init__()
        if pretrained:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(model_name)
        else:
            from transformers import Dinov2Config, Dinov2Model

            self.model = Dinov2Model(Dinov2Config(**(config_overrides or {})))
        self.hidden_size = self.model.config.hidden_size

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    @property
    def encoder_layers(self) -> nn.ModuleList:
        return self.model.encoder.layer

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return the full layer-normed token sequence ``(B, 1+N, D)``.

        ``Dinov2Model`` applies its final LayerNorm to ``last_hidden_state``
        itself, and it interpolates positional encodings automatically, so any
        input resolution divisible by the patch size works (multi-resolution
        inference).
        """
        out = self.model(pixel_values=pixel_values)
        return out.last_hidden_state

    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """Global embedding = ``[CLS]`` token."""
        return tokens[:, 0]

    def patches(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens[:, self.num_prefix_tokens:]


class Dinov3Backbone(Dinov2Backbone):
    """DINOv3 encoder (``facebook/dinov3-*``), if the installed transformers
    version ships it. DINOv3 adds register tokens between ``[CLS]`` and the
    patch tokens; they are dropped from ``patches``.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        freeze: bool = True,
        pretrained: bool = True,
        config_overrides: dict | None = None,
    ) -> None:
        try:
            from transformers import DINOv3ViTModel  # noqa: F401
        except ImportError as e:  # pragma: no cover - depends on transformers
            raise ImportError(
                "DINOv3 requires transformers>=4.56 (DINOv3ViTModel not found in "
                "the installed version). Upgrade transformers or pick another "
                "backbone (siglip2 / dinov2 / clip)."
            ) from e
        nn.Module.__init__(self)
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name) if pretrained else None
        if self.model is None:  # pragma: no cover
            raise ValueError("Dinov3Backbone supports pretrained=True only.")
        self.hidden_size = self.model.config.hidden_size
        self.num_prefix_tokens = 1 + getattr(self.model.config, "num_register_tokens", 0)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    @property
    def encoder_layers(self) -> nn.ModuleList:
        return self.model.layer

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=pixel_values)
        return out.last_hidden_state


class CLIPBackbone(nn.Module):
    """CLIP vision tower exposing the ``[CLS]`` token and patch tokens."""

    num_prefix_tokens = 1  # [CLS]

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        freeze: bool = True,
        pretrained: bool = True,
        config_overrides: dict | None = None,
    ) -> None:
        super().__init__()
        if pretrained:
            from transformers import CLIPVisionModel

            self.model = CLIPVisionModel.from_pretrained(model_name)
        else:
            from transformers import CLIPVisionConfig, CLIPVisionModel

            self.model = CLIPVisionModel(CLIPVisionConfig(**(config_overrides or {})))
        self.hidden_size = self.model.config.hidden_size
        self._native_size = self.model.config.image_size

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    @property
    def encoder_layers(self) -> nn.ModuleList:
        return self.model.vision_model.encoder.layers

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return the final-layernormed token sequence ``(B, 1+N, D)``."""
        interpolate = pixel_values.shape[-1] != self._native_size
        out = self.model(pixel_values=pixel_values, interpolate_pos_encoding=interpolate)
        return self.model.vision_model.post_layernorm(out.last_hidden_state)

    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens[:, 0]

    def patches(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens[:, self.num_prefix_tokens:]


_BACKBONES = {
    "siglip2": SiglipLOGERBackbone,
    "dinov2": Dinov2Backbone,
    "dinov3": Dinov3Backbone,
    "clip": CLIPBackbone,
}

_NORMALIZATION = {
    "siglip2": SIGLIP_STATS,
    "dinov2": IMAGENET_STATS,
    "dinov3": IMAGENET_STATS,
    "clip": CLIP_STATS,
}


def normalization_stats(name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return ``(mean, std)`` preprocessing statistics for a backbone family."""
    family = name.split("-")[0]
    if family not in _NORMALIZATION:
        raise ValueError(f"Unknown backbone family: {name}")
    return _NORMALIZATION[family]


def build_backbone(
    name: str,
    model_name: str | None = None,
    freeze: bool = True,
    pretrained: bool = True,
    config_overrides: dict | None = None,
) -> nn.Module:
    """Instantiate a backbone by family name (``siglip2-base`` -> ``siglip2``).

    Args:
        name: Backbone family, optionally suffixed (e.g. ``siglip2-base``).
        model_name: HuggingFace checkpoint id; ``None`` uses the class default.
        freeze: Freeze backbone weights (PEFT modes re-enable selectively).
        pretrained: Load pre-trained weights.
        config_overrides: Tiny-model config for offline tests.
    """
    family = name.split("-")[0]
    if family not in _BACKBONES:
        raise ValueError(f"Unknown backbone: {name} (choose from {sorted(_BACKBONES)})")
    kwargs: dict = dict(freeze=freeze, pretrained=pretrained, config_overrides=config_overrides)
    if model_name:
        kwargs["model_name"] = model_name
    return _BACKBONES[family](**kwargs)


def inject_lora(backbone: nn.Module, r: int = 8, alpha: float = 8.0, dropout: float = 0.0) -> int:
    """Wrap attention q/k/v projections of any supported backbone with LoRA.

    Walks the encoder blocks and replaces linear layers whose qualified name
    ends with a known attention-projection suffix by :class:`LoRALinear`.

    Returns:
        The number of injected LoRA layers.
    """
    injected = 0
    for layer in backbone.encoder_layers:
        for qual_name, module in list(layer.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not any(qual_name.endswith(sfx) for sfx in _LORA_TARGET_SUFFIXES):
                continue
            parent = layer
            *path, attr = qual_name.split(".")
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, attr, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
            injected += 1
    if injected == 0:
        raise RuntimeError("LoRA injection found no attention projections to wrap.")
    return injected
