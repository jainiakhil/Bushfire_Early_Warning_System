"""Training loop with MLflow tracking, cosine LR, AMP and early stopping.

Spec Task 5.3:

* MLflow logs hyperparameters, the cosine-annealing learning-rate schedule,
  epoch losses, IoU and Precision-Recall AUC.
* Early stopping on validation IoU.

Gradient accumulation (``training.grad_accum_steps``) lets a 6 GB GPU train at an
effective batch size of 16 while only holding 8 samples in memory.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config import Config
from src.data.dataset import build_dataloaders
from src.models.loss import FocalDiceLoss
from src.models.metrics import MetricTracker
from src.models.unet import build_model, export_torchscript
from src.utils.logging import get_logger
from src.utils.runtime import resolve_device, seed_everything

logger = get_logger(__name__)


@dataclass
class TrainResult:
    """Outcome of a training run."""

    best_iou: float
    best_epoch: int
    checkpoint_path: Path
    history: list[dict[str, float]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def _git_sha() -> str | None:
    """Best-effort current commit hash for provenance."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


class EarlyStopping:
    """Stop when the monitored metric has not improved for ``patience`` epochs."""

    def __init__(self, patience: int = 8, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.count = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Record a new metric value; return True if it was an improvement."""
        if value > self.best + self.min_delta:
            self.best = value
            self.count = 0
            return True
        self.count += 1
        if self.count >= self.patience:
            self.should_stop = True
        return False


def _run_epoch(
    model: nn.Module,
    loader: Any,
    criterion: FocalDiceLoss,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    grad_accum: int = 1,
    desc: str = "",
) -> dict[str, float]:
    """Run one train (optimizer given) or eval (optimizer None) epoch."""
    is_train = optimizer is not None
    model.train(is_train)
    tracker = MetricTracker()
    total_loss = 0.0
    n = 0
    use_amp = scaler is not None and device.type == "cuda"

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    n_batches = len(loader)
    for step, (x, y) in enumerate(tqdm(loader, desc=desc, leave=False)):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train), torch.autocast(
            device_type=device.type, enabled=use_amp
        ):
            logits = model(x)
            loss = criterion(logits, y)

        if optimizer is not None:
            loss_scaled = loss / grad_accum
            if scaler is not None:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()
            if (step + 1) % grad_accum == 0 or (step + 1) == n_batches:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            tracker.update(logits, y)

        total_loss += float(loss.detach()) * x.size(0)
        n += x.size(0)

    out = {"loss": total_loss / max(n, 1)}
    if not is_train:
        out.update(tracker.compute())
    return out


def train(cfg: Config, feature_manifest: dict[str, Any], *, fold: int = 0) -> TrainResult:
    """Train the segmentation model for one spatial-CV fold.

    Args:
        cfg: Pipeline config.
        feature_manifest: Manifest of feature-stack rasters + FIRMS paths.
        fold: Fold index to hold out for validation.

    Returns:
        A :class:`TrainResult` with the best checkpoint path and metric history.
    """
    seed_everything(cfg.training.seed)
    device = resolve_device(cfg.training.device)
    logger.info("Training fold %d on %s", fold, device)

    train_loader, val_loader, info = build_dataloaders(cfg, feature_manifest, fold=fold)
    model = build_model(cfg).to(device)
    criterion = FocalDiceLoss(
        alpha=cfg.training.focal_alpha,
        gamma=cfg.training.focal_gamma,
        dice_weight=cfg.training.dice_weight,
    )
    optimizer = AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.training.epochs, 1))
    scaler = (
        torch.amp.GradScaler("cuda")
        if (cfg.training.precision == "16-mixed" and device.type == "cuda")
        else None
    )
    stopper = EarlyStopping(patience=cfg.training.early_stopping_patience)

    ckpt_dir = cfg.data.output_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"fold{fold}_best.pt"

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # tolerate file:// if configured
    mlflow.set_tracking_uri(cfg.mlflow.resolved_tracking_uri())
    mlflow.set_experiment(cfg.mlflow.experiment)
    history: list[dict[str, float]] = []
    best = TrainResult(best_iou=-1.0, best_epoch=-1, checkpoint_path=ckpt_path)

    with mlflow.start_run(run_name=f"fold{fold}-{int(time.time())}"):
        mlflow.log_params(
            {
                "encoder": cfg.model.encoder,
                "encoder_weights": cfg.model.encoder_weights,
                "in_channels": cfg.model.in_channels,
                "batch_size": cfg.training.batch_size,
                "grad_accum_steps": cfg.training.grad_accum_steps,
                "effective_batch": cfg.training.effective_batch_size,
                "learning_rate": cfg.training.learning_rate,
                "epochs": cfg.training.epochs,
                "focal_alpha": cfg.training.focal_alpha,
                "focal_gamma": cfg.training.focal_gamma,
                "fold": fold,
                "n_train_tiles": info["n_train"],
                "n_val_tiles": info["n_val"],
                "device": device.type,
            }
        )

        for epoch in range(cfg.training.epochs):
            t0 = time.time()
            tr = _run_epoch(
                model, train_loader, criterion, device,
                optimizer=optimizer, scaler=scaler,
                grad_accum=cfg.training.grad_accum_steps, desc=f"train e{epoch}",
            )
            va = _run_epoch(model, val_loader, criterion, device, desc=f"val e{epoch}")
            scheduler.step()

            row: dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": float(tr["loss"]),
                "val_loss": float(va["loss"]),
                "val_iou": float(va["iou"]),
                "val_pr_auc": float(va["pr_auc"]),
                "lr": float(scheduler.get_last_lr()[0]),
                "epoch_seconds": time.time() - t0,
            }
            history.append(row)
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
            logger.info(
                "epoch %d | train_loss %.4f | val_loss %.4f | val_IoU %.4f | val_PR-AUC %.4f | lr %.2e",
                epoch, tr["loss"], va["loss"], va["iou"], va["pr_auc"], row["lr"],
            )

            improved = stopper.step(va["iou"])
            if improved:
                best = TrainResult(
                    best_iou=va["iou"], best_epoch=epoch, checkpoint_path=ckpt_path,
                    history=history, metrics={"val_iou": va["iou"], "val_pr_auc": va["pr_auc"], "val_loss": va["loss"]},
                )
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": cfg.model_dump(),
                        "epoch": epoch,
                        "metrics": best.metrics,
                        "norm_stats": info["norm_stats"],
                    },
                    ckpt_path,
                )
                logger.info("  -> new best (val IoU %.4f); saved %s", va["iou"], ckpt_path.name)

            if stopper.should_stop:
                logger.info("Early stopping at epoch %d (no val-IoU gain for %d epochs)",
                            epoch, cfg.training.early_stopping_patience)
                break

        mlflow.log_metrics({"best_val_iou": best.best_iou, "best_epoch": best.best_epoch})
        if ckpt_path.is_file():
            mlflow.log_artifact(str(ckpt_path))

    best.history = history
    return best


def train_and_export(cfg: Config, feature_manifest: dict[str, Any], *, fold: int = 0) -> tuple[TrainResult, Path]:
    """Train one fold then export the best checkpoint to TorchScript."""
    result = train(cfg, feature_manifest, fold=fold)
    if not result.checkpoint_path.is_file():
        raise RuntimeError("training produced no checkpoint")

    ckpt = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    ts_path = cfg.data.output_path / "model.torchscript"
    export_torchscript(
        model, cfg, ts_path,
        metrics=result.metrics,
        norm_stats=ckpt.get("norm_stats"),
        git_sha=_git_sha(),
    )
    return result, ts_path
