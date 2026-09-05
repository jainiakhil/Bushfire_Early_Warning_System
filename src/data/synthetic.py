"""Synthetic Sentinel-1 scene-pair generator (credential-free development data).

Downloading real Sentinel-1 GRD scenes and NASA FIRMS hotspots needs accounts,
API keys and gigabytes of transfer. To keep the pipeline, the test-suite and CI
fully runnable offline, this module fabricates *physically plausible* stand-ins:

* Two-band (VV, VH) GeoTIFFs for a pre-event ``t0`` and co-event ``t1`` scene,
  georeferenced in a real UTM zone at ~10 m resolution.
* Backscatter modelled in decibels as a smooth random field plus a few land-cover
  patches; multiplicative Gamma speckle is applied so the Refined Lee filter has
  real work to do.
* In ``t1``, elliptical "fire" regions where VH drops several dB (canopy loss /
  moisture) with roughened boundaries.
* A FIRMS-style GeoJSON of thermal hotspots sampled inside the fire ellipses
  (plus a few false positives), matching the field names of the real product.

The output layout and the ``manifest.json`` it writes are identical to what the
real :mod:`src.data.download` path produces, so nothing downstream can tell the
difference.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from scipy.ndimage import gaussian_filter

from src.config import Config
from src.utils.geo import utm_epsg_for_lonlat
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Nominal dB levels for a temperate vegetated landscape (Sentinel-1 IW).
_VV_MEAN_DB = -11.0
_VH_MEAN_DB = -17.5
_VV_STD_DB = 2.2
_VH_STD_DB = 2.8

# A scattering of plausible AOI centroids over fire-prone regions (lon, lat).
_AOI_CENTROIDS: list[tuple[float, float]] = [
    (149.13, -35.28),   # SE Australia (ACT / NSW)
    (151.21, -33.87),   # Greater Sydney bushland
    (-120.5, 38.5),     # Sierra Nevada, California
    (-8.4, 40.2),       # central Portugal
    (23.7, 38.9),       # Attica, Greece
]


def _db_to_linear(db: np.ndarray) -> np.ndarray:
    """Convert decibels to linear power."""
    return np.power(10.0, db / 10.0)


def _linear_to_db(lin: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Convert linear power to decibels."""
    return 10.0 * np.log10(np.maximum(lin, eps))


def _smooth_field(rng: np.random.Generator, size: int, mean: float, std: float, scale: float) -> np.ndarray:
    """A spatially correlated Gaussian random field (dB units)."""
    noise = rng.standard_normal((size, size)).astype(np.float32)
    field = gaussian_filter(noise, sigma=scale)
    field = (field - field.mean()) / (field.std() + 1e-6)
    return mean + std * field


def _add_landcover_patches(
    rng: np.random.Generator, base: np.ndarray, n_patches: int = 6
) -> np.ndarray:
    """Overlay a few brighter/darker blobs to mimic distinct land-cover classes."""
    size = base.shape[0]
    out = base.copy()
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(n_patches):
        cy, cx = rng.integers(0, size, size=2)
        r = rng.integers(size // 12, size // 5)
        offset = rng.uniform(-3.0, 3.0)
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) < r**2
        out[mask] += offset
    return out


def _apply_speckle(rng: np.random.Generator, linear: np.ndarray, looks: int) -> np.ndarray:
    """Apply multiplicative Gamma(L, 1/L) speckle to a linear-power image.

    Multi-look SAR intensity speckle follows a Gamma distribution with shape
    ``L`` (the equivalent number of looks) and unit mean. Lower ``looks`` = more
    speckle.
    """
    looks = max(1, int(looks))
    speckle = rng.gamma(shape=looks, scale=1.0 / looks, size=linear.shape).astype(np.float32)
    return linear * speckle


def _stamp_fires(
    rng: np.random.Generator,
    vv_db: np.ndarray,
    vh_db: np.ndarray,
    n_fires: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Burn elliptical fire scars into the co-event dB images.

    Returns the modified VV/VH arrays and a list of ellipse descriptors
    ``{cx, cy, ry, rx, angle}`` in pixel coordinates for hotspot sampling.
    """
    size = vv_db.shape[0]
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    ellipses: list[dict[str, float]] = []
    vv_out, vh_out = vv_db.copy(), vh_db.copy()

    for _ in range(n_fires):
        cy = rng.uniform(0.2 * size, 0.8 * size)
        cx = rng.uniform(0.2 * size, 0.8 * size)
        ry = rng.uniform(size * 0.05, size * 0.15)
        rx = rng.uniform(size * 0.05, size * 0.15)
        ang = rng.uniform(0, np.pi)

        cos_a, sin_a = np.cos(ang), np.sin(ang)
        xr = (xx - cx) * cos_a + (yy - cy) * sin_a
        yr = -(xx - cx) * sin_a + (yy - cy) * cos_a
        dist = (xr / rx) ** 2 + (yr / ry) ** 2

        core = dist < 1.0
        # Roughened boundary: noisy annulus 1.0 <= dist < 1.6
        boundary = (dist >= 1.0) & (dist < 1.6) & (rng.random((size, size)) < 0.5)

        vh_drop = rng.uniform(3.0, 6.0)
        vv_drop = rng.uniform(0.5, 2.0)
        vh_out[core] -= vh_drop
        vv_out[core] -= vv_drop
        vh_out[boundary] -= vh_drop * 0.5
        # Extra texture right at the front.
        vh_out[boundary] += rng.uniform(-1.5, 1.5, size=boundary.sum())

        ellipses.append({"cx": cx, "cy": cy, "rx": rx, "ry": ry, "angle": ang})

    return vv_out, vh_out, ellipses


def _hotspots_geojson(
    rng: np.random.Generator,
    ellipses: list[dict[str, float]],
    transform: Affine,
    crs: CRS,
    overpass: datetime,
) -> dict[str, Any]:
    """Build a FIRMS-style ``FeatureCollection`` of thermal hotspots.

    Points are sampled inside each fire ellipse (with jitter) plus a handful of
    false positives elsewhere. Property names mirror the real VIIRS product.
    """
    from rasterio.warp import transform as warp_transform

    feats: list[dict[str, Any]] = []

    def _emit(px: float, py: float, confidence: str, bright: float) -> None:
        x, y = transform * (px, py)
        lon, lat = warp_transform(crs, CRS.from_epsg(4326), [x], [y])
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon[0], lat[0]]},
                "properties": {
                    "latitude": round(lat[0], 5),
                    "longitude": round(lon[0], 5),
                    "bright_ti4": round(bright, 1),
                    "acq_date": overpass.strftime("%Y-%m-%d"),
                    "acq_time": overpass.strftime("%H%M"),
                    "confidence": confidence,
                    "satellite": "N",
                    "instrument": "VIIRS",
                    "frp": round(rng.uniform(2.0, 45.0), 1),
                },
            }
        )

    for ell in ellipses:
        n_pts = rng.integers(8, 25)
        for _ in range(n_pts):
            t = rng.uniform(0, 2 * np.pi)
            rad = np.sqrt(rng.uniform(0, 1))
            px = ell["cx"] + rad * ell["rx"] * np.cos(t)
            py = ell["cy"] + rad * ell["ry"] * np.sin(t)
            conf = rng.choice(["nominal", "high", "high", "low"])
            _emit(px, py, str(conf), rng.uniform(320.0, 367.0))

    # False positives (e.g. industrial heat, sun glint).
    size = 1  # placeholder; use transform extent instead
    for _ in range(rng.integers(1, 4)):
        px = rng.uniform(0, 1000)
        py = rng.uniform(0, 1000)
        _emit(px, py, "low", rng.uniform(300.0, 315.0))

    return {"type": "FeatureCollection", "features": feats}


def generate_sar_scene_pair(
    cfg: Config,
    scene_id: str,
    *,
    seed: int,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate one synthetic ``(t0, t1, hotspots)`` triple and write it to disk.

    Args:
        cfg: Pipeline config (uses ``cfg.synthetic`` and ``cfg.data``).
        scene_id: Identifier used in output filenames, e.g. ``"scene_000"``.
        seed: Per-scene RNG seed for reproducibility.
        out_dir: Target directory. Defaults to ``cfg.data.raw_path``.

    Returns:
        A manifest entry dict: ``{id, t0, t1, firms, bbox, crs, overpass}``.
    """
    scfg = cfg.synthetic
    size = scfg.image_size
    out_dir = out_dir or cfg.data.raw_path
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # --- geolocation -------------------------------------------------------
    lon0, lat0 = _AOI_CENTROIDS[seed % len(_AOI_CENTROIDS)]
    lon0 += rng.uniform(-0.05, 0.05)
    lat0 += rng.uniform(-0.05, 0.05)
    epsg = utm_epsg_for_lonlat(lon0, lat0)
    crs = CRS.from_string(epsg)

    from rasterio.warp import transform as warp_transform

    east, north = warp_transform(CRS.from_epsg(4326), crs, [lon0], [lat0])
    res = cfg.data.spatial_resolution
    # Place the AOI centroid at the image centre.
    x_min = east[0] - (size / 2) * res
    y_max = north[0] + (size / 2) * res
    transform = Affine(res, 0.0, x_min, 0.0, -res, y_max)

    # --- t0 backscatter (dB) --------------------------------------------
    vv0 = _add_landcover_patches(rng, _smooth_field(rng, size, _VV_MEAN_DB, _VV_STD_DB, scale=size / 40))
    vh0 = _add_landcover_patches(rng, _smooth_field(rng, size, _VH_MEAN_DB, _VH_STD_DB, scale=size / 40))

    # --- t1 backscatter: t0 plus small change plus fires -----------------
    vv1 = vv0 + _smooth_field(rng, size, 0.0, 0.6, scale=size / 30)
    vh1 = vh0 + _smooth_field(rng, size, 0.0, 0.8, scale=size / 30)
    vv1, vh1, ellipses = _stamp_fires(rng, vv1, vh1, scfg.n_fires)

    # --- speckle (apply in linear power, store dB) ----------------------
    def _speckled_db(db: np.ndarray) -> np.ndarray:
        return _linear_to_db(_apply_speckle(rng, _db_to_linear(db), scfg.noise_looks))

    bands = {
        "t0": np.stack([_speckled_db(vv0), _speckled_db(vh0)]).astype(np.float32),
        "t1": np.stack([_speckled_db(vv1), _speckled_db(vh1)]).astype(np.float32),
    }

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 2,
        "height": size,
        "width": size,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "predictor": 2,
        "nodata": None,
    }

    paths: dict[str, str] = {}
    for tag, arr in bands.items():
        p = out_dir / f"{scene_id}_{tag}.tif"
        with rasterio.open(p, "w", **profile) as ds:
            ds.write(arr)
            ds.set_band_description(1, "sigma0_vv_db")
            ds.set_band_description(2, "sigma0_vh_db")
        paths[tag] = str(p)

    overpass = datetime.now(timezone.utc) - timedelta(days=int(rng.integers(1, 20)))
    hotspots = _hotspots_geojson(rng, ellipses, transform, crs, overpass)
    firms_path = out_dir / f"{scene_id}_firms.geojson"
    firms_path.write_text(json.dumps(hotspots, indent=2), encoding="utf-8")

    # AOI bbox in lon/lat.
    xs = [transform.c, transform.c + size * res]
    ys = [transform.f - size * res, transform.f]
    lons, lats = warp_transform(crs, CRS.from_epsg(4326), xs, ys)
    bbox = [min(lons), min(lats), max(lons), max(lats)]

    logger.info("Generated synthetic scene %s (%s, %d fires)", scene_id, epsg, scfg.n_fires)
    return {
        "id": scene_id,
        "t0": paths["t0"],
        "t1": paths["t1"],
        "firms": str(firms_path),
        "bbox": [round(v, 5) for v in bbox],
        "crs": epsg,
        "overpass": overpass.isoformat(),
        "n_fires": scfg.n_fires,
    }


def generate_dataset(
    cfg: Config, *, n_scenes: int | None = None, out_dir: Path | None = None
) -> Path:
    """Generate a full synthetic dataset and write ``manifest.json``.

    Args:
        cfg: Pipeline config.
        n_scenes: Number of scene pairs. Defaults to ``cfg.synthetic.n_scenes``.
        out_dir: Output directory. Defaults to ``cfg.data.raw_path``.

    Returns:
        Path to the written ``manifest.json``.
    """
    n_scenes = n_scenes or cfg.synthetic.n_scenes
    out_dir = out_dir or cfg.data.raw_path
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = [
        generate_sar_scene_pair(cfg, f"scene_{i:03d}", seed=cfg.synthetic.seed + i, out_dir=out_dir)
        for i in range(n_scenes)
    ]
    manifest = {
        "source": "synthetic",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "spatial_resolution": cfg.data.spatial_resolution,
        "scenes": entries,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote manifest with %d scenes -> %s", len(entries), manifest_path)
    return manifest_path
