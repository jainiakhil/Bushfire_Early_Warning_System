"""Tests for the Focal, Dice and hybrid Focal-Dice losses."""

from __future__ import annotations

import torch

from src.models.loss import DiceLoss, FocalDiceLoss, FocalLoss


def _logit(p: float) -> float:
    t = torch.tensor(p)
    return float(torch.log(t / (1 - t)))


def test_dice_loss_zero_for_perfect_prediction():
    target = torch.zeros(2, 1, 16, 16)
    target[:, :, 4:8, 4:8] = 1.0
    logits = torch.where(target > 0, torch.tensor(12.0), torch.tensor(-12.0))
    assert DiceLoss()(logits, target) < 1e-2


def test_dice_loss_high_for_inverted_prediction():
    target = torch.zeros(2, 1, 16, 16)
    target[:, :, 4:8, 4:8] = 1.0
    logits = torch.where(target > 0, torch.tensor(-12.0), torch.tensor(12.0))
    assert DiceLoss()(logits, target) > 0.9


def test_focal_loss_less_than_bce_for_easy_examples():
    # gamma>0 must down-weight confident-correct predictions relative to BCE.
    logits = torch.full((100,), 4.0)
    target = torch.ones(100)
    focal = FocalLoss(alpha=0.5, gamma=2.0)(logits, target)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    assert focal < bce


def test_focal_dice_is_sum_of_components():
    torch.manual_seed(0)
    logits = torch.randn(3, 1, 8, 8, requires_grad=True)
    target = (torch.rand(3, 1, 8, 8) > 0.7).float()
    crit = FocalDiceLoss(alpha=0.25, gamma=2.0, dice_weight=1.0)
    total = crit(logits, target)
    comp = crit.last_components
    assert abs(comp["focal"] + comp["dice"] - float(total)) < 1e-5
    total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
