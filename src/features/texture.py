"""Multi-channel feature-stack construction (spec Module 3, Task 3.1).

Given calibrated, speckle-filtered pre-event (t0) and co-event (t1) VV/VH
rasters, build the 6-channel stack the U-Net consumes:

===  =====================  ================================================
Idx  Band                   Meaning
===  =====================  ================================================
0    ``sigma0_vv``          co-event VV (dB) - surface roughness
1    ``sigma0_vh``          co-event VH (dB) - volume scattering
2    ``delta_vv``           VV(t1) - VV(t0) (dB) - temporal change
3    ``delta_vh``           VH(t1) - VH(t0) (dB) - canopy loss / moisture drop
4    ``pol_ratio``          VH(t1) / (VV(t1) + eps) - bare ground vs vegetation
5    ``glcm_contrast``      GLCM contrast texture, 7x7 window on VH(t1)
===  =====================  ================================================

The channel *order* is taken from ``cfg.features.selected_bands`` so the feature
selection step can drop or reorder channels without touching this code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from skimage.feature import graycomatrix, graycoprops

from src.config import Config
from src.data.preprocess import db_to_linear
from src.utils.logging import get_logger

logger = get_logger(__name__)

_EPS = 1e-6

# Every band this module knows how to compute. `selected_bands` must be a subset.
_KNOWN_BANDS = {"sigma0_vv", "sigma0_vh", "delta_vv", "delta_vh", "pol_ratio", "glcm_contrast"}


def glcm_contrast(
    image_db: np.ndarray, *, window: int = 7, levels: int = 32, clip: tuple[float, float] = (-30.0, 5.0)
) -> np.ndarray:
    r"""Sliding-window GLCM contrast texture.

    The Grey-Level Co-occurrence Matrix contrast
    :math:`\\sum_{i,j} (i-j)^2 p(i,j)` is high where neighbouring pixels differ
    strongly - e.g. the turbulent edge of an active fire front.

    The dB image is linearly quantised to *levels* grey levels over *clip*, then
    for every pixel a GLCM is built from its ``window x window`` neighbourhood
    (offsets of 1 px at 0 degrees and 90 degrees, averaged).

    Args:
        image_db: 2-D backscatter image (dB), typically VH(t1).
        window: Odd neighbourhood size.
        levels: Number of quantisation levels.
        clip: dB range mapped onto ``[0, levels-1]``.

    Returns:
        Float32 contrast map, same shape as *image_db*, edges reflected.
    """
    if window % 2 == 0:
        raise ValueError("window must be odd")

    lo, hi = clip
    q = np.clip((image_db - lo) / (hi - lo), 0.0, 1.0)
    q = (q * (levels - 1)).round().astype(np.uint8)

    pad = window // 2
    qp = np.pad(q, pad, mode="reflect")
    out = np.zeros_like(image_db, dtype=np.float32)

    # Iterating pixel-by-pixel is O(H*W*window^2); acceptable for tiles and cached
    # to disk per scene. For very large scenes this is the dominant cost.
    h, w = image_db.shape
    for r in range(h):
        row_slice = qp[r : r + window]
        for c in range(w):
            patch = row_slice[:, c : c + window]
            glcm = graycomatrix(
                patch,
                distances=[1],
                angles=[0.0, np.pi / 2],
                levels=levels,
                symmetric=True,
                normed=True,
            )
            out[r, c] = float(graycoprops(glcm, "contrast").mean())
    return out


def _read_two_band(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a 2-band (VV, VH) raster -> ``(vv, vh, profile)``."""
    with rasterio.open(path) as ds:
        arr = ds.read().astype(np.float32)
        profile = ds.profile
    return arr[0], arr[1], profile


def build_feature_stack(
    t0_cal: str | Path,
    t1_cal: str | Path,
    cfg: Config,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    """Compute the ordered feature stack for one scene.

    Args:
        t0_cal: Calibrated pre-event 2-band raster (VV, VH; dB).
        t1_cal: Calibrated co-event 2-band raster (VV, VH; dB).
        cfg: Pipeline config (``features.selected_bands`` drives channel order).

    Returns:
        ``(stack, profile, band_names)`` where *stack* is ``(C, H, W)`` float32
        and *band_names* matches ``cfg.features.selected_bands``.
    """
    bands = cfg.features.selected_bands
    unknown = set(bands) - _KNOWN_BANDS
    if unknown:
        raise ValueError(f"Unknown feature band(s): {sorted(unknown)}")

    vv0, vh0, _ = _read_two_band(t0_cal)
    vv1, vh1, profile = _read_two_band(t1_cal)

    if vv0.shape != vv1.shape:
        raise ValueError(f"t0 {vv0.shape} and t1 {vv1.shape} rasters are not aligned")

    computed: dict[str, np.ndarray] = {}
    if "sigma0_vv" in bands:
        computed["sigma0_vv"] = vv1
    if "sigma0_vh" in bands:
        computed["sigma0_vh"] = vh1
    if "delta_vv" in bands:
        computed["delta_vv"] = vv1 - vv0
    if "delta_vh" in bands:
        computed["delta_vh"] = vh1 - vh0
    if "pol_ratio" in bands:
        # Ratio of linear powers is the physically meaningful quantity.
        computed["pol_ratio"] = (db_to_linear(vh1) / (db_to_linear(vv1) + _EPS)).astype(np.float32)
    if "glcm_contrast" in bands:
        logger.info("Computing GLCM contrast (%dx%d window) - this is the slow step",
                    cfg.features.glcm_window, cfg.features.glcm_window)
        computed["glcm_contrast"] = glcm_contrast(
            vh1, window=cfg.features.glcm_window, levels=cfg.features.glcm_levels
        )

    stack = np.stack([computed[b] for b in bands]).astype(np.float32)
    # Replace non-finite values (from ratios / edges) with 0.
    stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)

    out_profile = {
        **profile,
        "count": len(bands),
        "dtype": "float32",
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    logger.info("Built %d-channel feature stack %s", len(bands), stack.shape)
    return stack, out_profile, bands


def write_feature_stack(
    stack: np.ndarray, profile: dict[str, Any], band_names: list[str], out_path: str | Path
) -> Path:
    """Write a feature stack to a band-described GeoTIFF."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as ds:
        ds.write(stack)
        for i, name in enumerate(band_names, start=1):
            ds.set_band_description(i, name)
        ds.update_tags(feature_bands=",".join(band_names))
    return out_path


def build_features_for_manifest(cfg: Config, processed_manifest: dict[str, Any]) -> dict[str, Any]:
    """Compute + cache feature stacks for every scene in a processed manifest.

    Returns a feature manifest (also written to
    ``processed_dir/feature_manifest.json``) with a ``feature_path`` per scene.
    """
    out: dict[str, Any] = {"source": processed_manifest.get("source"), "scenes": []}
    for scene in processed_manifest["scenes"]:
        sid = scene["id"]
        feat_path = cfg.data.processed_path / f"{sid}_features.tif"
        if not feat_path.is_file():
            stack, profile, names = build_feature_stack(scene["t0_cal"], scene["t1_cal"], cfg)
            write_feature_stack(stack, profile, names, feat_path)
        out["scenes"].append(
            {
                "id": sid,
                "feature_path": str(feat_path),
                "firms": scene["firms"],
                "crs": scene.get("crs"),
                "bbox": scene.get("bbox"),
            }
        )
    path = cfg.data.processed_path / "feature_manifest.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("Wrote feature manifest (%d scenes) -> %s", len(out["scenes"]), path)
    return out
