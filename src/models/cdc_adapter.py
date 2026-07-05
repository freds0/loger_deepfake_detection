"""Central Difference Convolution (CDC) adapter.

Implements the CDC adapter inserted into the transformer FFN blocks, described
in Sec. III-B of the OSDFD paper, Fig. 4 (c), Fig. 5 and Eqs. (1)-(3):

    h_out          = MLP(h_in) + Adapter(h_in)                          (Eq. 1)
    Adapter(h_in)  = Conv1x1_up( CDC( Conv1x1_down(h_in) ) )            (Eq. 2)
    x_out          = sum_{i in Omega} w_i (x_i^in - x_c)                (Eq. 3)

The adapter reshapes the sequence of patch tokens back into a 2-D feature map,
reduces the channel dimension with a 1x1 convolution, applies the CDC operator
(a local high-frequency / anomaly extractor), restores the channels with a
second 1x1 convolution and flattens the result to the original token shape.

The CDC operator (Eq. 3) computes, inside each local sliding window, the
weighted sum of differences between peripheral pixels ``x_i`` and the central
pixel ``x_c``. Following the generalised formulation of Su et al. (CDCN,
cited as [68]), we blend the central-difference term with a vanilla
convolution via ``theta``:

    y = vanilla_conv(x) - theta * x_c * sum(w)

``theta = 1`` recovers the pure central-difference form of Eq. (3); the CDCN
default ``theta = 0.7`` is used here as a well-tested value and is configurable.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CDC2d(nn.Module):
    """Central Difference Convolution over a 2-D feature map (Eq. 3, Fig. 5).

    Args:
        channels: Number of input == output channels (depth-preserving).
        kernel_size: Size of the local window ``Omega`` (default 3 -> 3x3).
        theta: Blend between central-difference (1.0) and vanilla conv (0.0).
    """

    def __init__(self, channels: int, kernel_size: int = 3, theta: float = 0.7) -> None:
        super().__init__()
        self.theta = theta
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_normal = self.conv(x)
        if self.theta == 0.0:
            return out_normal
        # Central-difference term: sum_i w_i * x_c = x_c * sum_i(w_i).
        # A 1x1 conv with per-(out,in)-channel weight = sum of the spatial
        # kernel weights realises x_c * sum(w) at every location.
        kernel_diff = self.conv.weight.sum(dim=(2, 3))  # (C_out, C_in)
        kernel_diff = kernel_diff[:, :, None, None]
        out_diff = F.conv2d(x, kernel_diff, stride=self.conv.stride, padding=0)
        return out_normal - self.theta * out_diff


class CDCAdapter(nn.Module):
    """FFN-parallel adapter with a CDC bottleneck (Eqs. 1-2, Fig. 4c).

    Operates on patch-token sequences of shape ``(B, N, D)`` where ``N`` is a
    perfect square (the number of patch tokens). SigLIP 2 has no ``[CLS]``
    token, so every token maps to a spatial location and ``N = H * W``.

    Args:
        dim: Token embedding dimension ``D`` (768 for SigLIP2-base).
        bottleneck: Reduced channel dimension inside the adapter.
        kernel_size: CDC window size.
        theta: CDC blend factor (see :class:`CDC2d`).
        activation: Whether to apply a GELU between CDC and the up-projection.
    """

    def __init__(
        self,
        dim: int,
        bottleneck: int = 64,
        kernel_size: int = 3,
        theta: float = 0.7,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        # 1x1 down-projection: reduces channels to limit trainable parameters
        # (paper: "reduce the dimension ... using a 1x1 convolutional layer").
        self.conv_down = nn.Conv2d(dim, bottleneck, kernel_size=1)
        self.cdc = CDC2d(bottleneck, kernel_size=kernel_size, theta=theta)
        self.act = nn.GELU() if activation else nn.Identity()
        # 1x1 up-projection: restores the original channel count.
        self.conv_up = nn.Conv2d(bottleneck, dim, kernel_size=1)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Near-zero up-projection so the adapter starts as an identity residual
        # (the FFN output is unchanged at initialisation).
        nn.init.kaiming_uniform_(self.conv_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.conv_down.bias)
        nn.init.zeros_(self.conv_up.weight)
        nn.init.zeros_(self.conv_up.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, n, d = h.shape
        side = int(round(math.sqrt(n)))
        if side * side != n:
            raise ValueError(
                f"CDCAdapter expects a square number of patch tokens, got N={n}. "
                "SigLIP 2 fixed-resolution models produce (image/patch)^2 tokens."
            )
        # (B, N, D) -> (B, D, H, W): expand tokens into a 2-D feature map.
        x = h.transpose(1, 2).reshape(b, d, side, side)
        x = self.conv_down(x)
        x = self.cdc(x)
        x = self.act(x)
        x = self.conv_up(x)
        # (B, D, H, W) -> (B, N, D): flatten back to the original token shape.
        return x.reshape(b, d, n).transpose(1, 2)


class AdapterMLP(nn.Module):
    """Wrap a frozen transformer FFN with a parallel CDC adapter (Eq. 1).

    Replaces a SigLIP 2 encoder layer's ``mlp`` submodule so that the block
    output becomes ``MLP(h_in) + Adapter(h_in)``. The original MLP weights are
    frozen; only the adapter is trainable.
    """

    def __init__(self, base_mlp: nn.Module, adapter: CDCAdapter) -> None:
        super().__init__()
        self.base_mlp = base_mlp
        for p in self.base_mlp.parameters():
            p.requires_grad_(False)
        self.adapter = adapter

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.base_mlp(h) + self.adapter(h)
