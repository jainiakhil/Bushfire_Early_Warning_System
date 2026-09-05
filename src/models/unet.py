"""U-Net segmentation model and TorchScript export (spec Task 5.2).

The architecture is a U-Net with a **ResNet-34 encoder pre-trained on ImageNet**,
provided by `segmentation-models-pytorch`. `smp` automatically adapts the first
convolution to accept 6 input channels (it repeats/rescales the pretrained RGB
kernels), so no manual surgery is needed.

`smp` has no built-in bottleneck dropout, so :class:`SegModel` wraps the `smp`
model and injects :class:`torch.nn.Dropout2d` (spatial dropout, ``p=0.2``) on the
deepest encoder feature map before it enters the decoder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import segmentation_models_pytorch as smp
import torch
from torch import Tensor, nn

from src.config import Config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SegModel(nn.Module):
    """ResNet-U-Net with spatial dropout in the bottleneck.

    Wrapping (rather than subclassing) the `smp` model keeps this robust to
    `smp` internal changes: we only rely on the documented
    ``encoder`` / ``decoder`` / ``segmentation_head`` attributes.
    """

    def __init__(self, cfg: Config) -> None:
        """Build the model from ``cfg.model``."""
        super().__init__()
        m = cfg.model
        self.net: smp.Unet = smp.Unet(
            encoder_name=m.encoder,
            encoder_weights=m.encoder_weights,
            in_channels=m.in_channels,
            classes=m.classes,
            decoder_use_norm="batchnorm",
        )
        self.dropout = nn.Dropout2d(p=m.bottleneck_dropout)
        self.in_channels = m.in_channels
        self.classes = m.classes

    def forward(self, x: Tensor) -> Tensor:
        """Return segmentation **logits** of shape ``(N, classes, H, W)``."""
        features = list(self.net.encoder(x))
        # features run shallow -> deep; apply spatial dropout to the deepest map.
        features[-1] = self.dropout(features[-1])
        decoder_output = self.net.decoder(features)
        masks: Tensor = self.net.segmentation_head(decoder_output)
        return masks


def build_model(cfg: Config) -> SegModel:
    """Instantiate the segmentation model.

    Args:
        cfg: Pipeline config.

    Returns:
        A :class:`SegModel` (uninitialised optimiser state; caller moves it to
        the target device).
    """
    model = SegModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Built %s-U-Net: encoder=%s weights=%s in_ch=%d params=%.1fM",
        cfg.model.arch,
        cfg.model.encoder,
        cfg.model.encoder_weights,
        cfg.model.in_channels,
        n_params / 1e6,
    )
    return model


@torch.no_grad()
def export_torchscript(
    model: nn.Module,
    cfg: Config,
    out_path: str | Path,
    *,
    metrics: dict[str, float] | None = None,
    norm_stats: dict[str, list[float]] | None = None,
    git_sha: str | None = None,
) -> Path:
    """Trace the model to TorchScript and write a sidecar model card.

    Args:
        model: A trained model (any device; moved to CPU + eval for export).
        cfg: Pipeline config.
        out_path: Destination ``.torchscript`` path.
        metrics: Optional training metrics to record in the model card.
        norm_stats: Per-channel normalisation used at train time (inference must
            reuse these).
        git_sha: Optional commit hash for provenance.

    Returns:
        Path to the written TorchScript file. A ``<stem>.modelcard.json`` is
        written alongside it.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval().cpu()
    dummy = torch.randn(1, cfg.model.in_channels, cfg.data.tile_size, cfg.data.tile_size)

    try:
        scripted = torch.jit.trace(model, dummy, strict=False)
        # Parity check: traced output must match eager within tolerance.
        if not torch.allclose(model(dummy), scripted(dummy), atol=1e-4):
            raise RuntimeError("traced output diverged from eager; falling back to script")
        mode = "trace"
    except Exception as exc:  # pragma: no cover - architecture dependent
        logger.warning("torch.jit.trace failed (%s); using torch.jit.script", exc)
        scripted = torch.jit.script(model)
        mode = "script"

    scripted.save(str(out_path))

    card: dict[str, Any] = {
        "arch": cfg.model.arch,
        "encoder": cfg.model.encoder,
        "encoder_weights": cfg.model.encoder_weights,
        "in_channels": cfg.model.in_channels,
        "classes": cfg.model.classes,
        "tile_size": cfg.data.tile_size,
        "feature_bands": cfg.features.selected_bands,
        "confidence_threshold": cfg.api.confidence_threshold,
        "export_mode": mode,
        "metrics": metrics or {},
        "norm_stats": norm_stats or {},
        "git_sha": git_sha,
        "torch_version": torch.__version__,
    }
    card_path = out_path.with_suffix(".modelcard.json")
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    logger.info("Exported TorchScript (%s) -> %s", mode, out_path)
    return out_path
