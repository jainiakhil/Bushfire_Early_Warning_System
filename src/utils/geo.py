"""Geospatial utilities: CRS selection, raster windowing, and vectorisation.

The headline function is :func:`raster_to_geojson`, which turns a model
probability raster into a clean GeoJSON ``FeatureCollection`` of fire-front
polygons with confidence and area metadata (spec Task 6.1).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.features import shapes as rio_shapes
from rasterio.warp import transform_geom
from rasterio.windows import Window
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from src.utils.logging import get_logger

logger = get_logger(__name__)

# 1 pixel at 10 m resolution = 100 m^2 = 0.01 ha. Overridable per call.
_M2_PER_HA = 10_000.0


def utm_epsg_for_lonlat(lon: float, lat: float) -> str:
    """Return the EPSG code of the UTM zone containing ``(lon, lat)``.

    Args:
        lon: Longitude in degrees (-180..180).
        lat: Latitude in degrees (-80..84 for valid UTM).

    Returns:
        An ``"EPSG:32XXX"`` string - ``326xx`` for the northern hemisphere,
        ``327xx`` for the southern.
    """
    zone = int(math.floor((lon + 180.0) / 6.0) % 60) + 1
    return f"EPSG:{('326' if lat >= 0 else '327')}{zone:02d}"


def resolve_working_crs(crs_working: str, bounds_lonlat: tuple[float, float, float, float]) -> str:
    """Resolve the configured working CRS, expanding the ``"auto-utm"`` sentinel.

    Args:
        crs_working: Value of ``data.crs_working`` from config.
        bounds_lonlat: ``(min_lon, min_lat, max_lon, max_lat)`` of the AOI, used
            only when *crs_working* is ``"auto-utm"``.

    Returns:
        A concrete CRS string such as ``"EPSG:32756"``.
    """
    if crs_working.lower() == "auto-utm":
        cx = 0.5 * (bounds_lonlat[0] + bounds_lonlat[2])
        cy = 0.5 * (bounds_lonlat[1] + bounds_lonlat[3])
        epsg = utm_epsg_for_lonlat(cx, cy)
        logger.debug("auto-utm resolved to %s for centroid (%.3f, %.3f)", epsg, cx, cy)
        return epsg
    return crs_working


def iter_tile_windows(
    width: int, height: int, tile_size: int, overlap: int
) -> list[Window]:
    """Enumerate overlapping read windows covering a raster.

    The last row/column of tiles is shifted inward so every tile is exactly
    ``tile_size`` on a side (no ragged edges), at the cost of extra overlap
    against the penultimate tile.

    Args:
        width: Raster width in pixels.
        height: Raster height in pixels.
        tile_size: Tile edge length in pixels.
        overlap: Overlap between adjacent tiles in pixels.

    Returns:
        A list of :class:`rasterio.windows.Window`.
    """
    if tile_size <= overlap:
        raise ValueError("tile_size must be greater than overlap")
    step = tile_size - overlap
    xs = list(range(0, max(1, width - overlap), step))
    ys = list(range(0, max(1, height - overlap), step))
    windows: list[Window] = []
    for y0 in ys:
        for x0 in xs:
            x = min(x0, max(0, width - tile_size))
            y = min(y0, max(0, height - tile_size))
            windows.append(Window(x, y, min(tile_size, width), min(tile_size, height)))
    # De-duplicate windows that collapsed onto each other on tiny rasters.
    seen: set[tuple[float, float, float, float]] = set()
    unique: list[Window] = []
    for w in windows:
        key = (w.col_off, w.row_off, w.width, w.height)
        if key not in seen:
            seen.add(key)
            unique.append(w)
    return unique


def _risk_level(confidence: float) -> str:
    """Bucket a mean confidence value into a coarse risk label."""
    if confidence >= 0.85:
        return "extreme"
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.65:
        return "moderate"
    return "low"


def raster_to_geojson(
    prob_mask: np.ndarray,
    transform: Affine,
    src_crs: str | CRS,
    *,
    threshold: float = 0.65,
    simplify_tolerance: float = 1e-4,
    min_area_ha: float = 0.5,
    out_crs: str | CRS = "EPSG:4326",
    pixel_area_m2: float | None = None,
    detected_at: datetime | None = None,
) -> dict[str, Any]:
    """Vectorise a probability raster into fire-front polygons.

    Pipeline (spec Task 6.1):

    1. Threshold ``prob_mask >= threshold`` to a binary mask.
    2. Extract polygons with :func:`rasterio.features.shapes` (in *src_crs*).
    3. Merge adjacent polygons with :func:`shapely.ops.unary_union`.
    4. Reproject each merged polygon to *out_crs*, then simplify with the
       Douglas-Peucker algorithm (tolerance is in *out_crs* units).
    5. Drop polygons smaller than *min_area_ha* and attach metadata
       (mean confidence, area in hectares, risk level).

    Args:
        prob_mask: 2-D array of per-pixel fire probabilities in ``[0, 1]``.
        transform: Affine geotransform mapping pixel -> *src_crs* coordinates.
        src_crs: CRS of *transform* (the working/projected CRS).
        threshold: Probability cut for the binary mask.
        simplify_tolerance: Douglas-Peucker tolerance in *out_crs* units.
        min_area_ha: Minimum polygon area to keep, in hectares.
        out_crs: CRS of the returned GeoJSON (default WGS84 lon/lat).
        pixel_area_m2: Area of one pixel in square metres. If ``None`` it is
            derived from *transform* (``|a * e|``).
        detected_at: Timestamp stamped into each feature. Defaults to now (UTC).

    Returns:
        A GeoJSON ``FeatureCollection`` dict. ``collection["properties"]`` holds
        run-level aggregates (``n_features``, ``total_area_ha``, ``threshold``).
    """
    if prob_mask.ndim != 2:
        raise ValueError(f"prob_mask must be 2-D, got shape {prob_mask.shape}")

    src_crs = CRS.from_user_input(src_crs)
    out_crs = CRS.from_user_input(out_crs)
    detected_at = detected_at or datetime.now(UTC)

    if pixel_area_m2 is None:
        pixel_area_m2 = abs(transform.a * transform.e)

    binary = (prob_mask >= threshold).astype(np.uint8)
    if not binary.any():
        logger.info("raster_to_geojson: no pixels above threshold %.2f", threshold)
        return _empty_collection(threshold)

    # (geometry, value) pairs; value==1 are the fire polygons.
    raw_polys = [
        (shape(geom), float(val))
        for geom, val in rio_shapes(binary, mask=binary.astype(bool), transform=transform)
        if val == 1
    ]
    if not raw_polys:
        return _empty_collection(threshold)

    merged = unary_union([g for g, _ in raw_polys])
    parts = list(merged.geoms) if merged.geom_type.startswith("Multi") else [merged]

    features: list[dict[str, Any]] = []
    total_area_ha = 0.0
    for poly in parts:
        # Confidence = mean probability over the pixels this polygon covers.
        area_m2 = poly.area  # src_crs is projected -> area is in m^2
        area_ha = area_m2 / _M2_PER_HA
        if area_ha < min_area_ha:
            continue
        mean_conf = _mean_probability_in_polygon(prob_mask, transform, poly, threshold)

        geom_out = transform_geom(src_crs, out_crs, mapping(poly))
        geom_simplified = shape(geom_out).simplify(simplify_tolerance, preserve_topology=True)

        features.append(
            {
                "type": "Feature",
                "geometry": mapping(geom_simplified),
                "properties": {
                    "mean_confidence": round(mean_conf, 4),
                    "area_ha": round(area_ha, 3),
                    "area_pixels": int(round(area_m2 / pixel_area_m2)),
                    "risk_level": _risk_level(mean_conf),
                    "detected_at": detected_at.isoformat(),
                },
            }
        )
        total_area_ha += area_ha

    logger.info(
        "raster_to_geojson: %d polygons, %.2f ha total (threshold=%.2f)",
        len(features),
        total_area_ha,
        threshold,
    )
    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "n_features": len(features),
            "total_area_ha": round(total_area_ha, 3),
            "threshold": threshold,
            "crs": str(out_crs),
            "generated_at": detected_at.isoformat(),
        },
    }


def _mean_probability_in_polygon(
    prob_mask: np.ndarray, transform: Affine, poly: Any, threshold: float
) -> float:
    """Mean probability over pixels whose centre falls inside *poly*.

    Uses a rasterised polygon mask so the cost is O(bbox) rather than O(vertices).
    """
    from rasterio.features import geometry_mask

    h, w = prob_mask.shape
    try:
        poly_mask = geometry_mask(
            [mapping(poly)], out_shape=(h, w), transform=transform, invert=True
        )
    except Exception:  # pragma: no cover - degenerate geometry
        return float(prob_mask[prob_mask >= threshold].mean())
    vals = prob_mask[poly_mask]
    if vals.size == 0:
        return float(threshold)
    return float(vals.mean())


def _empty_collection(threshold: float) -> dict[str, Any]:
    """Return a well-formed empty ``FeatureCollection``."""
    return {
        "type": "FeatureCollection",
        "features": [],
        "properties": {
            "n_features": 0,
            "total_area_ha": 0.0,
            "threshold": threshold,
            "generated_at": datetime.now(UTC).isoformat(),
        },
    }


def read_band_stack(path: str, band_indices: list[int] | None = None) -> tuple[np.ndarray, dict]:
    """Read a (subset of) raster bands into a ``(C, H, W)`` float32 array.

    Args:
        path: Path to a GDAL-readable raster.
        band_indices: 1-based band numbers to read. ``None`` reads all bands.

    Returns:
        ``(array, profile)`` where *profile* is the rasterio profile dict.
    """
    with rasterio.open(path) as ds:
        idx = band_indices or list(range(1, ds.count + 1))
        arr = ds.read(idx).astype(np.float32)
        profile = ds.profile
    return arr, profile
