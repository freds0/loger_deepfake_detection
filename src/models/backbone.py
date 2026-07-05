"""SigLIP 2 visual encoder wrapper.

Loads a SigLIP 2 vision tower (default ``google/siglip2-base-patch16-224``) and
exposes the two representations the OSDFD framework needs:

  * ``patch_tokens`` -- the last-layer, un-pooled patch-token sequence
    ``(B, N, D)`` (used by the CDC adapter, the Forgery Style Mixture module and
    optional local feature fusion).
  * ``pooled`` -- the MAP (Multihead Attention Pooling) head embedding
    ``(B, D)`` (the global feature, paper Sec. III-A).

SigLIP 2 follows the standard ViT architecture with learned positional
embeddings and pools with a MAP head (SigLIP 2 paper, Sec. 2.1: "Vision and
text representations are pooled using a MAP head (attention pooling)"). Unlike
CLIP there is **no** ``[CLS]`` token, so every token corresponds to an image
patch -- convenient for the CDC adapter's token->grid reshape.

The backbone is frozen by default (paper: "we freeze plain ViT backbones ...
and solely update the lightweight Adapter and LoRA layers"). PEFT modules are
injected by :func:`src.models.peft_inject.inject_peft`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class Siglip2Backbone(nn.Module):
    """Frozen SigLIP 2 vision encoder exposing patch tokens and MAP pooling.

    Args:
        model_name: HuggingFace checkpoint id of the SigLIP 2 model.
        freeze: If True, all backbone parameters start frozen (default). PEFT
            modules injected afterwards re-enable gradients only for themselves.
        pretrained: If True, load pre-trained weights from ``model_name``. If
            False, build a randomly-initialised vision tower from
            ``config_overrides`` (used for fast, offline unit tests).
        config_overrides: ``SiglipVisionConfig`` kwargs for the non-pretrained
            path (e.g. a tiny model for testing).
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        freeze: bool = True,
        pretrained: bool = True,
        config_overrides: dict | None = None,
    ) -> None:
        super().__init__()
        if pretrained:
            # Fixed-resolution SigLIP 2 checkpoints load via the SigLIP v1
            # classes (backward compatible); AutoModel returns the full model.
            full = AutoModel.from_pretrained(model_name)
            self.vision_model = full.vision_model
        else:
            from transformers import SiglipVisionConfig, SiglipVisionModel

            cfg = SiglipVisionConfig(**(config_overrides or {}))
            self.vision_model = SiglipVisionModel(cfg).vision_model
        self.hidden_size = self.vision_model.config.hidden_size

        if freeze:
            for p in self.vision_model.parameters():
                p.requires_grad_(False)

    @property
    def encoder_layers(self) -> nn.ModuleList:
        """The transformer encoder blocks (for PEFT injection)."""
        return self.vision_model.encoder.layers

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the (PEFT-adapted) encoder and return the patch tokens.

        The MAP head is *not* applied here so that the Forgery Style Mixture
        module can operate on the un-pooled patch tokens before pooling
        (paper Fig. 4a: transformer blocks -> FSM -> MLP head). Use
        :meth:`pool` to obtain the global embedding from (optionally FSM-mixed)
        tokens.

        Args:
            pixel_values: Normalised images ``(B, 3, H, W)``.

        Returns:
            ``patch_tokens`` of shape ``(B, N, D)``.
        """
        vm = self.vision_model
        hidden_states = vm.embeddings(pixel_values)
        encoder_outputs = vm.encoder(inputs_embeds=hidden_states)
        last_hidden_state = encoder_outputs.last_hidden_state
        patch_tokens = vm.post_layernorm(last_hidden_state)
        return patch_tokens

    def pool(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Apply the MAP pooling head to (optionally FSM-mixed) patch tokens.

        Args:
            patch_tokens: ``(B, N, D)`` post-layernorm patch tokens.

        Returns:
            Pooled global embedding ``(B, D)``.
        """
        return self.vision_model.head(patch_tokens)
