"""RCS-Loss: combined classification (cross-entropy) + regression (MAE/MSE) for gaze angle prediction.

This implementation follows Abdelrahman *et al.* 2024 ("Fine-grained gaze estimation based on the combination of regression and classification losses").

• A discrete set of equally-sized bins cover the full angular range.
• The network outputs `num_bins` logits; soft-max yields probabilities `p_i`.
• An expected angle \thetâ is recovered via the *soft arg-max*:
      \thetâ = w * Σ p_i (i - (B-1)/2)
  where `w` is the bin width and `i=0…B-1`.
• Loss = CE(logits, bin_target) + α * regression(\thetâ, θ_gt).

Notes
-----
- Binning helper works with arbitrary ranges by specifying `bin_width` and `num_bins`; angles outside the range are clamped.
- Supports either MAE (default) or MSE as the regression metric.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class RCSLoss(nn.Module):
    """Regression-Classification Separation loss for a single angle (yaw or pitch)."""

    def __init__(
        self,
        num_bins: int,
        bin_width: float = 3.0,
        alpha: float = 1.0,
        regression: Literal["mae", "mse"] = "mae",
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.bin_width = bin_width
        self.alpha = alpha
        self.regression = regression.lower()
        if self.regression not in {"mae", "mse"}:
            raise ValueError("regression must be 'mae' or 'mse'")

        self.ce_loss = nn.CrossEntropyLoss()

        # Pre-compute bin offset tensor used for expected angle calculation
        # offset = (i - (num_bins - 1)/2) for i in [0, B-1]
        offsets = torch.arange(num_bins).float() - (num_bins - 1) / 2.0
        self.register_buffer("_offsets", offsets)

    def _angle_to_bin(self, angle: torch.Tensor) -> torch.Tensor:
        """Convert continuous angle (degrees) to integer bin indices."""
        half_range = (self.num_bins * self.bin_width) / 2.0
        idx = torch.floor((angle + half_range) / self.bin_width)
        idx = torch.clamp(idx, 0, self.num_bins - 1).long()

        return idx

    def forward(self, logits: torch.Tensor, angle_target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss."""
        if logits.shape[1] != self.num_bins:
            raise ValueError(
                f"logits second dim ({logits.shape[1]}) != num_bins ({self.num_bins})")

        angle_target = angle_target.squeeze()
        bin_target = self._angle_to_bin(angle_target)

        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)

        # Classification term (cross-entropy)
        cls_loss = self.ce_loss(probs, bin_target)

        # Regression term
        offsets = self._offsets.to(logits.device)
        # Expected angle in bin units then scale by bin width
        expected = self.bin_width * torch.sum(probs * offsets, dim=1)

        if self.regression == "mae":
            reg_loss = F.l1_loss(expected, angle_target)
        else:
            reg_loss = F.mse_loss(expected, angle_target)

        total_loss = cls_loss + self.alpha * reg_loss

        return total_loss 
    