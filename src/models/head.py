"""Classification head for OSDFD.

A small MLP that maps the (optionally fused global + local) SigLIP 2 embedding
to a single forgery logit. Following the OSDFD paper (Sec. III-D), the feature
"after the second-to-last fully-connected layer" is exposed separately so it can
be used by the Single-Center Loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """MLP classification head returning ``(logit, scl_feature)``.

    Args:
        in_dim: Dimension of the input embedding.
        hidden_dim: Width of the penultimate (SCL) representation.
        dropout: Dropout applied before the penultimate layer.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Second-to-last FC layer: its output is the SCL feature (paper Sec. III-D).
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        # Last FC layer: produces the raw forgery logit for BCEWithLogitsLoss.
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logit, scl_feature)``.

        Args:
            x: Input embedding ``(B, in_dim)``.

        Returns:
            ``logit`` of shape ``(B,)`` and ``scl_feature`` of shape
            ``(B, hidden_dim)`` (the penultimate representation used by SCL).
        """
        scl_feature = self.act(self.fc1(self.dropout(x)))
        logit = self.fc2(scl_feature).squeeze(-1)
        return logit, scl_feature
