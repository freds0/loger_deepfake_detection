"""Low-Rank Adaptation (LoRA) layer.

Implements the LoRA module described in Sec. III-B of the OSDFD paper
("Open-Set Deepfake Detection: A Parameter-Efficient Adaptation Method with
Forgery Style Mixture"), Fig. 4 (d) and Eqs. (4)-(5):

    h_out = W_{q/k/v} h_in + LoRA(h_in)
    LoRA(h_in) = h_in W_down W_up,   W_down in R^{d x r}, W_up in R^{r x d}, r << d

The pre-trained projection weight ``W_{q/k/v}`` is kept frozen; only the two
low-rank matrices ``W_down`` and ``W_up`` are optimised. LoRA is injected into
the query / key / value projections of every self-attention block (paper: "the
LoRA layer, injected into the self-attention blocks").

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
(arXiv:2106.09685), cited as [63] in the OSDFD paper.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap a frozen ``nn.Linear`` with a trainable low-rank residual branch.

    The forward pass returns ``base(x) + scaling * up(down(x))`` where ``base``
    is the original (frozen) projection. Only ``down`` and ``up`` receive
    gradients.

    Args:
        base_linear: The pre-trained linear layer to adapt (kept frozen).
        r: LoRA rank (``r << d``). ``r = 0`` disables the low-rank branch.
        alpha: LoRA scaling factor; the residual is scaled by ``alpha / r``.
        dropout: Dropout applied to the input of the low-rank branch.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_linear)}")

        self.base = base_linear
        # Freeze the pre-trained projection (Eq. 4: W_{q/k/v} is fixed).
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        if r > 0:
            # W_down in R^{d x r}, W_up in R^{r x d} (Eq. 5).
            self.lora_down = nn.Linear(in_features, r, bias=False)
            self.lora_up = nn.Linear(r, out_features, bias=False)
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self._reset_lora_parameters()
        else:  # pragma: no cover - r=0 is an ablation-only degenerate case
            self.lora_down = None
            self.lora_up = None
            self.dropout = nn.Identity()

    def _reset_lora_parameters(self) -> None:
        # Standard LoRA init: Kaiming-uniform for down, zeros for up so the
        # adapted model initially equals the pre-trained model.
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r > 0:
            out = out + self.scaling * self.lora_up(self.lora_down(self.dropout(x)))
        return out

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"r={self.r}, alpha={self.alpha}"
