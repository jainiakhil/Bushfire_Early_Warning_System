"""Segmentation metrics: IoU and Precision-Recall AUC (spec Task 5.3).

Kept dependency-light and threshold-explicit so the numbers logged to MLflow are
easy to reason about. ``torchmetrics`` is used for PR-AUC because a correct
streaming implementation is fiddly.
"""

from __future__ import annotations

import torch
from torch import Tensor


def iou_score(logits: Tensor, target: Tensor, *, threshold: float = 0.5, eps: float = 1e-7) -> float:
    """Intersection-over-Union for the positive (fire) class.

    Args:
        logits: Pre-sigmoid predictions ``(N, 1, H, W)``.
        target: Binary targets, same shape.
        threshold: Probability cut for the predicted mask.
        eps: Guards division when both masks are empty (score -> ~1).

    Returns:
        Mean IoU over the batch as a Python float.
    """
    p = (torch.sigmoid(logits) >= threshold).float()
    y = (target >= 0.5).float()
    dims = tuple(range(1, p.ndim))
    inter = (p * y).sum(dim=dims)
    union = p.sum(dim=dims) + y.sum(dim=dims) - inter
    return float(((inter + eps) / (union + eps)).mean())


class MetricTracker:
    """Accumulates predictions across a validation epoch for IoU + PR-AUC.

    PR-AUC (a.k.a. average precision) is the area under the precision-recall
    curve and is far more informative than ROC-AUC when positives are rare.
    """

    def __init__(self, *, iou_threshold: float = 0.5, max_points: int = 2_000_000) -> None:
        """Initialise.

        Args:
            iou_threshold: Threshold used for the IoU summary.
            max_points: Cap on stored (prob, label) pairs for the PR curve, to
                bound memory on large validation sets (reservoir-style trim).
        """
        self.iou_threshold = iou_threshold
        self.max_points = max_points
        self._probs: list[Tensor] = []
        self._labels: list[Tensor] = []
        self._iou_sum = 0.0
        self._n_batches = 0

    def update(self, logits: Tensor, target: Tensor) -> None:
        """Add one batch of predictions."""
        self._iou_sum += iou_score(logits, target, threshold=self.iou_threshold)
        self._n_batches += 1
        probs = torch.sigmoid(logits).detach().flatten().cpu()
        labels = (target.detach().flatten().cpu() >= 0.5).long()
        # Subsample to keep memory bounded.
        if probs.numel() > 50_000:
            idx = torch.randperm(probs.numel())[:50_000]
            probs, labels = probs[idx], labels[idx]
        self._probs.append(probs)
        self._labels.append(labels)

    def compute(self) -> dict[str, float]:
        """Return ``{"iou": ..., "pr_auc": ...}`` for the accumulated data."""
        iou = self._iou_sum / max(self._n_batches, 1)
        result = {"iou": iou, "pr_auc": 0.0}
        if not self._probs:
            return result
        probs = torch.cat(self._probs)
        labels = torch.cat(self._labels)
        if labels.unique().numel() < 2:
            return result
        try:
            from torchmetrics.functional import average_precision

            ap = average_precision(probs, labels, task="binary")
            result["pr_auc"] = float(ap) if ap is not None else 0.0
        except Exception:  # pragma: no cover - fallback
            from sklearn.metrics import average_precision_score

            result["pr_auc"] = float(average_precision_score(labels.numpy(), probs.numpy()))
        return result

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._probs.clear()
        self._labels.clear()
        self._iou_sum = 0.0
        self._n_batches = 0
