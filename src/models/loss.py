r"""Hybrid Focal + Dice loss for imbalanced binary segmentation (spec Task 5.1).

.. math::
    \\mathcal{L}_{total} = \\mathcal{L}_{Focal} + w\\,\\mathcal{L}_{Dice}

* **Focal loss** down-weights easy background pixels so the vast, easy negative
  class does not swamp the gradient:
  :math:`\\mathcal{L}_{Focal} = -\\alpha_t (1 - p_t)^\\gamma \\log(p_t)`
  with :math:`\\alpha = 0.25,\\ \\gamma = 2.0`.
* **Dice loss** optimises region overlap directly, which stabilises training
  when the positive class is tiny:
  :math:`\\mathcal{L}_{Dice} = 1 - \\frac{2 \\sum p y + \\epsilon}{\\sum p + \\sum y + \\epsilon}`.

All modules take **logits** (pre-sigmoid) of shape ``(N, 1, H, W)`` and float
targets in ``{0, 1}`` of the same shape.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FocalLoss(nn.Module):
    """Binary focal loss on logits."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        """Initialise.

        Args:
            alpha: Weight of the positive class in ``[0, 1]``.
            gamma: Focusing parameter; 0 recovers weighted BCE.
            reduction: ``"mean"``, ``"sum"`` or ``"none"``.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute the focal loss.

        Args:
            logits: Pre-sigmoid predictions, any shape.
            target: Same-shape float tensor with values in ``{0, 1}``.

        Returns:
            Scalar (or per-element, if ``reduction="none"``) loss.
        """
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1.0 - p) * (1.0 - target)          # prob of the true class
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """Soft Dice loss on logits (computed per-batch)."""

    def __init__(self, smooth: float = 1.0) -> None:
        """Initialise.

        Args:
            smooth: Laplace smoothing added to numerator and denominator; also
                makes the loss well-defined when a batch has no positive pixels.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute ``1 - Dice`` over the whole batch."""
        p = torch.sigmoid(logits)
        p = p.reshape(p.shape[0], -1)
        y = target.reshape(target.shape[0], -1)
        intersection = (p * y).sum(dim=1)
        union = p.sum(dim=1) + y.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalDiceLoss(nn.Module):
    """Sum of focal and Dice losses (spec Task 5.1).

    ``forward`` returns the combined scalar. The individual components from the
    most recent call are stashed on ``self.last_components`` for logging.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        dice_weight: float = 1.0,
        dice_smooth: float = 1.0,
    ) -> None:
        """Initialise.

        Args:
            alpha: Focal positive-class weight.
            gamma: Focal focusing parameter.
            dice_weight: Multiplier on the Dice term.
            dice_smooth: Dice Laplace smoothing.
        """
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss(smooth=dice_smooth)
        self.dice_weight = dice_weight
        self.last_components: dict[str, float] = {}

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Return ``focal + dice_weight * dice``."""
        focal = self.focal(logits, target)
        dice = self.dice(logits, target)
        total = focal + self.dice_weight * dice
        self.last_components = {
            "focal": float(focal.detach()),
            "dice": float(dice.detach()),
            "total": float(total.detach()),
        }
        return total
