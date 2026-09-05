"""Radiometric calibration and speckle filtering (spec Module 2, Task 2.2).

Steps:

1. **Calibration to decibels** - convert intensity/DN to
   :math:`\\sigma^0_{dB} = 10 \\log_{10}(DN^2 + \\epsilon)`.
   (Synthetic scenes are already written in dB; the helper is idempotent for
   values that are already in a sane dB range.)
2. **Refined Lee speckle filter** - an edge-preserving adaptive filter applied
   over a 5x5 window in *linear power*, then converted back to dB.
3. **Reprojection** - warp to the configured working CRS (a local UTM zone by
   default) and write a float32, tiled, compressed GeoTIFF.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.ndimage import uniform_filter

from src.config import Config
from src.utils.geo import resolve_working_crs
from src.utils.logging import get_logger

logger = get_logger(__name__)


def to_db(dn: np.ndarray, *, eps: float = 1e-10) -> np.ndarray:
    """Convert digital numbers / linear intensity to backscatter decibels.

    Implements :math:`\\sigma^0_{dB} = 10 \\log_{10}(DN^2 + \\epsilon)`.

    If the input already looks like decibels (values mostly in ``[-50, 10]``),
    it is returned unchanged so the function is safe to call twice.

    Args:
        dn: Array of digital numbers or linear intensity (non-negative), or an
            array already in dB.
        eps: Small constant guarding the logarithm.

    Returns:
        Array in decibels, same shape as *dn*.
    """
    finite = dn[np.isfinite(dn)]
    if finite.size and finite.min() < -1.0 and finite.max() < 30.0:
        # Already in dB (has negative values, no huge magnitudes).
        return dn.astype(np.float32)
    return (10.0 * np.log10(np.square(dn.astype(np.float64)) + eps)).astype(np.float32)


def db_to_linear(db: np.ndarray) -> np.ndarray:
    """Decibels -> linear power."""
    return np.power(10.0, db.astype(np.float64) / 10.0)


def linear_to_db(lin: np.ndarray, *, eps: float = 1e-10) -> np.ndarray:
    """Linear power -> decibels."""
    return (10.0 * np.log10(np.maximum(lin, eps))).astype(np.float32)


def refined_lee(image_db: np.ndarray, *, window: int = 5, enl: float = 4.0) -> np.ndarray:
    """Edge-preserving Refined Lee speckle filter.

    The Lee filter shrinks each pixel towards its local mean by a weight that
    depends on the local coefficient of variation relative to the speckle
    (noise) coefficient of variation :math:`C_v = 1/\\sqrt{L}`:

    .. math::
        \\hat R = \\bar I + W (I - \\bar I), \\quad
        W = \\frac{\\mathrm{Var}(I) - \\sigma_v^2 \\bar I^2}{\\mathrm{Var}(I)}

    The *refined* variant estimates the local statistics from the sub-window
    (one of 8 directional half-windows) that best aligns with a local edge,
    which preserves high-contrast structural boundaries - important for keeping
    fire fronts crisp.

    Args:
        image_db: 2-D backscatter image in decibels.
        window: Odd moving-window size (default 5).
        enl: Equivalent number of looks; sets :math:`\\sigma_v^2 = 1/L`.

    Returns:
        Filtered image in decibels, same shape.
    """
    if image_db.ndim != 2:
        raise ValueError("refined_lee expects a 2-D image")
    if window % 2 == 0:
        raise ValueError("window must be odd")

    lin = db_to_linear(image_db)
    sigma_v2 = 1.0 / float(max(enl, 1e-3))

    # Full-window local mean / variance.
    mean = uniform_filter(lin, size=window, mode="reflect")
    mean_sq = uniform_filter(lin * lin, size=window, mode="reflect")
    var = np.maximum(mean_sq - mean * mean, 0.0)

    # --- edge-aligned sub-window refinement -----------------------------
    # Gradient magnitude via a coarse 3x3 mean difference in 4 directions.
    g = _directional_gradients(mean)
    edge_dir = np.argmax(np.abs(g), axis=0)  # 0..3 -> 0deg,45deg,90deg,135deg

    # For strong edges, recompute mean/var from the half-window on the brighter
    # side of the edge; elsewhere keep the isotropic statistics.
    edge_strength = np.max(np.abs(g), axis=0)
    strong = edge_strength > (0.5 * np.sqrt(var + 1e-12))

    ref_mean, ref_var = _subwindow_stats(lin, window, edge_dir)
    local_mean = np.where(strong, ref_mean, mean)
    local_var = np.where(strong, ref_var, var)

    # Lee weight, clipped to [0, 1].
    denom = np.where(local_var > 0, local_var, 1.0)
    weight = (local_var - sigma_v2 * local_mean * local_mean) / denom
    weight = np.clip(weight, 0.0, 1.0)

    filtered_lin = local_mean + weight * (lin - local_mean)
    filtered_lin = np.maximum(filtered_lin, 1e-12)
    return linear_to_db(filtered_lin)


def _directional_gradients(a: np.ndarray) -> np.ndarray:
    """Return a ``(4, H, W)`` stack of directional first differences."""
    up = np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0)          # vertical (0 deg edge)
    left = np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1)        # horizontal (90 deg)
    diag1 = np.roll(np.roll(a, -1, 0), -1, 1) - np.roll(np.roll(a, 1, 0), 1, 1)   # 45
    diag2 = np.roll(np.roll(a, -1, 0), 1, 1) - np.roll(np.roll(a, 1, 0), -1, 1)   # 135
    return np.stack([up, left, diag1, diag2])


def _subwindow_stats(
    lin: np.ndarray, window: int, edge_dir: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Local mean/variance from the half-window aligned with the local edge.

    A lightweight approximation: for each of the 4 edge orientations we average
    over a shifted full-window so the estimate is drawn predominantly from one
    side of the edge. This keeps the implementation vectorised and dependency
    free while still sharpening boundaries relative to the isotropic filter.
    """
    shifts = {
        0: (window // 2, 0),   # sample from below a vertical edge
        1: (0, window // 2),   # sample from the right of a horizontal edge
        2: (window // 2, window // 2),
        3: (window // 2, -window // 2),
    }
    mean_full = uniform_filter(lin, size=window, mode="reflect")
    var_full = np.maximum(
        uniform_filter(lin * lin, size=window, mode="reflect") - mean_full**2, 0.0
    )
    out_mean = mean_full.copy()
    out_var = var_full.copy()
    for d, (sy, sx) in shifts.items():
        m = edge_dir == d
        if not m.any():
            continue
        shifted_mean = np.roll(np.roll(mean_full, sy, axis=0), sx, axis=1)
        shifted_var = np.roll(np.roll(var_full, sy, axis=0), sx, axis=1)
        out_mean[m] = shifted_mean[m]
        out_var[m] = shifted_var[m]
    return out_mean, out_var


def calibrate_and_filter(
    scene_path: str | Path,
    out_path: str | Path,
    cfg: Config,
    *,
    dst_crs: str | None = None,
) -> Path:
    """Calibrate, speckle-filter and reproject a 2-band (VV, VH) scene.

    Args:
        scene_path: Input GeoTIFF (2 bands: VV, VH; DN or dB).
        out_path: Output GeoTIFF path.
        cfg: Pipeline config.
        dst_crs: Override the working CRS (else resolved from ``cfg.data``).

    Returns:
        Path to the written GeoTIFF (float32, 2 bands, tiled, deflate).
    """
    scene_path, out_path = Path(scene_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(scene_path) as src:
        data = src.read().astype(np.float32)  # (2, H, W)
        src_profile = src.profile
        src_bounds = src.bounds
        src_crs = src.crs

        # Resolve destination CRS.
        if dst_crs is None:
            from rasterio.warp import transform_bounds

            lon_min, lat_min, lon_max, lat_max = transform_bounds(
                src_crs, CRS.from_epsg(4326), *src_bounds
            )
            dst_crs = resolve_working_crs(
                cfg.data.crs_working, (lon_min, lat_min, lon_max, lat_max)
            )
        dst_crs_obj = CRS.from_user_input(dst_crs)

        # Calibrate + speckle filter each band (in source grid).
        filtered = np.empty_like(data)
        for b in range(data.shape[0]):
            db = to_db(data[b])
            filtered[b] = refined_lee(
                db, window=cfg.features.refined_lee_window, enl=cfg.features.equivalent_looks
            )

        # Reproject to the working CRS.
        transform, width, height = calculate_default_transform(
            src_crs, dst_crs_obj, src.width, src.height, *src_bounds,
            resolution=cfg.data.spatial_resolution,
        )
        dst = np.full((data.shape[0], height, width), np.nan, dtype=np.float32)
        for b in range(data.shape[0]):
            reproject(
                source=filtered[b],
                destination=dst[b],
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=dst_crs_obj,
                resampling=Resampling.bilinear,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )

    profile = {
        **src_profile,
        "driver": "GTiff",
        "dtype": "float32",
        "count": data.shape[0],
        "height": height,
        "width": width,
        "crs": dst_crs_obj,
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(out_path, "w", **profile) as dst_ds:
        dst_ds.write(dst)
        dst_ds.set_band_description(1, "sigma0_vv_db")
        dst_ds.set_band_description(2, "sigma0_vh_db")
        dst_ds.update_tags(processing="calibrated+refined_lee", enl=cfg.features.equivalent_looks)

    logger.info("Calibrated %s -> %s (%s, %dx%d)", scene_path.name, out_path.name, dst_crs, width, height)
    return out_path


def preprocess_manifest(cfg: Config, manifest: dict[str, Any]) -> dict[str, Any]:
    """Run :func:`calibrate_and_filter` over every scene in a manifest.

    Args:
        cfg: Pipeline config.
        manifest: Manifest dict (from :func:`src.data.download.resolve_scenes`).

    Returns:
        A processed-manifest dict mapping scene id -> calibrated t0/t1 paths and
        the FIRMS path, also written to ``processed_dir/processed_manifest.json``.
    """
    out: dict[str, Any] = {"source": manifest.get("source"), "scenes": []}
    for scene in manifest["scenes"]:
        sid = scene["id"]
        t0_out = cfg.data.processed_path / f"{sid}_t0_cal.tif"
        t1_out = cfg.data.processed_path / f"{sid}_t1_cal.tif"
        dst_crs = scene.get("crs")
        calibrate_and_filter(scene["t0"], t0_out, cfg, dst_crs=dst_crs)
        calibrate_and_filter(scene["t1"], t1_out, cfg, dst_crs=dst_crs)
        out["scenes"].append(
            {
                "id": sid,
                "t0_cal": str(t0_out),
                "t1_cal": str(t1_out),
                "firms": scene["firms"],
                "crs": dst_crs,
                "bbox": scene.get("bbox"),
            }
        )
    path = cfg.data.processed_path / "processed_manifest.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("Wrote processed manifest (%d scenes) -> %s", len(out["scenes"]), path)
    return out
