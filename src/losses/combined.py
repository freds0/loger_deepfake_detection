"""Combined OSDFD objective: BCE + lambda * SCL.

Implements the overall objective of Sec. III-D, Eq. (10):

    L = L_BCE + lambda * L_SCL

where ``L_BCE`` is the binary cross-entropy over forgery logits and ``L_SCL`` is
the :class:`SingleCenterLoss`. The paper sets ``lambda = 1`` by default.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .single_center_loss import SingleCenterLoss


class OSDFDLoss(nn.Module):
    """BCE-with-logits combined with the Single-Center Loss.

    Args:
        scl_weight: Loss weight ``lambda`` on the SCL term (paper default: 1.0).
        scl_margin: Margin of the Single-Center Loss (paper default: 0.01).
        pos_weight: Optional positive-class weight for BCE (class imbalance).
    """

    def __init__(
        self,
        scl_weight: float = 1.0,
        scl_margin: float = 0.01,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.scl_weight = scl_weight
        pw = torch.tensor(pos_weight) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.scl = SingleCenterLoss(margin=scl_margin)

    def forward(
        self,
        logits: torch.Tensor,
        scl_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the total loss and its components.

        Args:
            logits: Forgery logits ``(B,)``.
            scl_features: Penultimate features ``(B, F)``.
            labels: Binary labels ``(B,)`` (1 = fake, 0 = real).

        Returns:
            ``(total_loss, parts)`` where ``parts`` holds the detached ``bce``,
            ``scl`` and ``total`` scalars for logging.
        """
        labels_f = labels.to(logits.dtype)
        bce = self.bce(logits, labels_f)
        scl = self.scl(scl_features, labels)
        total = bce + self.scl_weight * scl
        parts = {
            "bce": bce.detach(),
            "scl": scl.detach(),
            "total": total.detach(),
        }
        return total, parts
