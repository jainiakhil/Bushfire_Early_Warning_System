"""Inference: tiled TorchScript prediction + vectorisation.

Shared by the FastAPI service, the Streamlit dashboard and the tests so there is
exactly one code path from "a 6-band GeoTIFF" to "a GeoJSON FeatureCollection".

Large scenes are processed tile-by-tile (256x256, 32-px overlap) and the
overlapping predictions are averaged when stitched back together, which removes
visible seams.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch

from src.config import Config
from src.utils.geo import iter_tile_windows, raster_to_geojson
from src.utils.logging import get_logger
from src.utils.runtime import resolve_device

logger = get_logger(__name__)


@lru_cache(maxsize=2)
def load_model(model_path: str, device_str: str = "cpu") -> torch.jit.ScriptModule:
    """Load and cache a TorchScript model.

    Args:
        model_path: Path to a ``.torchscript`` file.
        device_str: ``"cpu"`` or ``"cuda"``.

    Returns:
        The loaded, ``eval()``-mode :class:`torch.jit.ScriptModule`.
    """
    device = torch.device(device_str)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    logger.info("Loaded TorchScript model %s on %s", model_path, device_str)
    return model


def load_model_card(model_path: str | Path) -> dict[str, Any]:
    """Load the ``<stem>.modelcard.json`` sidecar, or return ``{}``."""
    card_path = Path(model_path).with_suffix(".modelcard.json")
    if card_path.is_file():
        return json.loads(card_path.read_text(encoding="utf-8"))
    return {}


def _standardise(arr: np.ndarray, norm_stats: dict[str, list[float]] | None) -> np.ndarray:
    """Apply the training-time per-channel standardisation (or a no-op)."""
    if not norm_stats:
        return arr
    mean = np.asarray(norm_stats["mean"], dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(norm_stats["std"], dtype=np.float32).reshape(-1, 1, 1)
    return (arr - mean) / std


@torch.no_grad()
def predict_geotiff(
    source: str | Path,
    model: torch.jit.ScriptModule,
    cfg: Config,
    *,
    norm_stats: dict[str, list[float]] | None = None,
    device_str: str = "cpu",
    batch_tiles: int = 8,
) -> tuple[np.ndarray, Any, Any]:
    """Run tiled inference over a multi-band GeoTIFF.

    Args:
        source: Path to a GeoTIFF with at least ``cfg.model.in_channels`` bands.
        model: A loaded TorchScript model.
        cfg: Pipeline config.
        norm_stats: Per-channel normalisation (from the model card / checkpoint).
        device_str: Device to run on.
        batch_tiles: How many tiles to push through the model at once.

    Returns:
        ``(prob_mask, transform, crs)`` - *prob_mask* is a full-scene float32
        array in ``[0, 1]``; *transform* and *crs* locate it.
    """
    device = torch.device(device_str)
    model = model.to(device)
    n_ch = cfg.model.in_channels
    ts = cfg.data.tile_size

    with rasterio.open(source) as ds:
        if ds.count < n_ch:
            raise ValueError(f"expected >= {n_ch} bands, got {ds.count}")
        transform, crs = ds.transform, ds.crs
        H, W = ds.height, ds.width
        windows = iter_tile_windows(W, H, ts, cfg.data.tile_overlap)

        prob_sum = np.zeros((H, W), dtype=np.float32)
        weight = np.zeros((H, W), dtype=np.float32)
        # A cosine taper gives overlapping tiles a smooth blend.
        taper = _cosine_window(ts)

        buf: list[tuple[Any, np.ndarray]] = []

        def _flush() -> None:
            if not buf:
                return
            batch = np.stack([b for _, b in buf])
            x = torch.from_numpy(batch).to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)
            for (win, _), p in zip(buf, probs, strict=False):
                r0, c0 = int(win.row_off), int(win.col_off)
                r1, c1 = r0 + int(win.height), c0 + int(win.width)
                t = taper[: r1 - r0, : c1 - c0]
                prob_sum[r0:r1, c0:c1] += p[: r1 - r0, : c1 - c0] * t
                weight[r0:r1, c0:c1] += t
            buf.clear()

        for win in windows:
            tile = ds.read(
                list(range(1, n_ch + 1)),
                window=win,
                boundless=True,
                fill_value=np.nan,
                out_dtype="float32",
            )
            tile = np.nan_to_num(tile, nan=0.0, posinf=0.0, neginf=0.0)
            tile = _standardise(tile, norm_stats)
            # Pad partial edge tiles to the full tile size.
            if tile.shape[1:] != (ts, ts):
                padded = np.zeros((n_ch, ts, ts), dtype=np.float32)
                padded[:, : tile.shape[1], : tile.shape[2]] = tile
                tile = padded
            buf.append((win, tile))
            if len(buf) >= batch_tiles:
                _flush()
        _flush()

    prob_mask = np.where(weight > 0, prob_sum / np.maximum(weight, 1e-6), 0.0).astype(np.float32)
    logger.info("Inference done: %dx%d, mean prob %.3f", W, H, float(prob_mask.mean()))
    return prob_mask, transform, crs


def _cosine_window(size: int) -> np.ndarray:
    """A 2-D separable Hann-like taper in ``[~0, 1]`` for seam-free blending."""
    w = np.hanning(size + 2)[1:-1]
    w = np.clip(w, 1e-3, None)
    return np.outer(w, w).astype(np.float32)


def detect(
    source: str | Path,
    cfg: Config,
    *,
    model_path: str | Path | None = None,
    device_str: str | None = None,
) -> dict[str, Any]:
    """End-to-end: GeoTIFF -> probability raster -> GeoJSON fire polygons.

    Args:
        source: Input multi-band GeoTIFF.
        cfg: Pipeline config.
        model_path: TorchScript model path. Defaults to
            ``cfg.data.output_path / "model.torchscript"``.
        device_str: Override device selection.

    Returns:
        A GeoJSON ``FeatureCollection`` (see :func:`src.utils.geo.raster_to_geojson`).
    """
    model_path = Path(model_path or (cfg.data.output_path / "model.torchscript"))
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train and export one first "
            "(scripts/run_training.py)."
        )
    device_str = device_str or resolve_device(cfg.training.device).type
    card = load_model_card(model_path)
    norm_stats = card.get("norm_stats") or None

    model = load_model(str(model_path), device_str)
    prob_mask, transform, crs = predict_geotiff(
        source, model, cfg, norm_stats=norm_stats, device_str=device_str
    )
    return raster_to_geojson(
        prob_mask,
        transform,
        crs,
        threshold=cfg.api.confidence_threshold,
        simplify_tolerance=cfg.api.simplify_tolerance,
        min_area_ha=cfg.api.min_polygon_area_ha,
        out_crs=cfg.data.crs_output,
    )
