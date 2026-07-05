"""Single-Center Loss (SCL).

Implements the Single-Center Loss of Sec. III-D of the OSDFD paper,
Eqs. (11)-(13):

    C       = (1 / N_R) * sum_i f_R^i                                   (13)
    Dist_R  = (1 / N_R) * sum_i || f_R^i - C ||_2                       (12)
    Dist_F  = (1 / N_F) * sum_j || f_F^j - C ||_2                       (12)
    L_SCL   = Dist_R + max(Dist_R - Dist_F + margin, 0)                 (11)

The real center ``C`` is computed as the batch mean of the *real* features
(Eq. 13) rather than a learned parameter. SCL compacts the real-feature
distribution while pushing fake features away from the real center, yielding a
clearer real/fake decision boundary. It is computed on the penultimate
("second-to-last fully-connected layer") features.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SingleCenterLoss(nn.Module):
    """Single-Center Loss (Eqs. 11-13).

    Args:
        margin: How much farther fake features must lie from the real center
            than the real features (paper default: 0.01).
    """

    def __init__(self, margin: float = 0.01) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the SCL for a batch.

        Args:
            features: Penultimate features ``(B, F)``.
            labels: Binary labels ``(B,)`` with 1 = fake, 0 = real.

        Returns:
            Scalar loss. Returns ``0`` if the batch has no real samples (the
            center is undefined) so training does not crash on degenerate
            batches.
        """
        labels = labels.to(features.dtype)
        real_mask = labels < 0.5
        fake_mask = ~real_mask

        if real_mask.sum() == 0:
            return features.new_zeros(())

        real_feats = features[real_mask]
        center = real_feats.mean(dim=0, keepdim=True)             # C (Eq. 13)

        dist_r = torch.norm(real_feats - center, dim=1).mean()    # Dist_R (Eq. 12)

        if fake_mask.sum() == 0:
            # Without fakes only the compactness term is defined.
            return dist_r

        fake_feats = features[fake_mask]
        dist_f = torch.norm(fake_feats - center, dim=1).mean()    # Dist_F (Eq. 12)

        hinge = torch.clamp(dist_r - dist_f + self.margin, min=0.0)
        return dist_r + hinge                                     # L_SCL (Eq. 11)
