"""Tests for vectorisation (raster_to_geojson) and tiling helpers."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import shape

from src.utils.geo import iter_tile_windows, raster_to_geojson, utm_epsg_for_lonlat


def test_utm_epsg_zones():
    assert utm_epsg_for_lonlat(151.2, -33.9) == "EPSG:32756"  # Sydney, south
    assert utm_epsg_for_lonlat(-120.5, 38.5) == "EPSG:32610"  # California, north


def test_tile_windows_cover_and_are_full_size():
    windows = iter_tile_windows(500, 300, tile_size=256, overlap=32)
    assert windows
    for w in windows:
        assert w.width <= 256 and w.height <= 256
    # Coverage: the union of windows spans the raster.
    max_x = max(w.col_off + w.width for w in windows)
    max_y = max(w.row_off + w.height for w in windows)
    assert max_x >= 500 - 1 and max_y >= 300 - 1


def test_raster_to_geojson_two_blobs(two_blob_probability):
    arr, transform = two_blob_probability
    fc = raster_to_geojson(
        arr, transform, "EPSG:32755", threshold=0.5, min_area_ha=0.1, out_crs="EPSG:4326"
    )
    assert fc["type"] == "FeatureCollection"
    assert fc["properties"]["n_features"] == 2
    for feat in fc["features"]:
        geom = shape(feat["geometry"])
        assert geom.is_valid
        assert -180 <= geom.centroid.x <= 180
        assert 0.0 <= feat["properties"]["mean_confidence"] <= 1.0
        assert feat["properties"]["area_ha"] > 0


def test_raster_to_geojson_empty_when_below_threshold(two_blob_probability):
    arr, transform = two_blob_probability
    fc = raster_to_geojson(arr, transform, "EPSG:32755", threshold=0.99)
    assert fc["features"] == []
    assert fc["properties"]["n_features"] == 0


def test_raster_to_geojson_area_is_reasonable(two_blob_probability):
    arr, transform = two_blob_probability
    fc = raster_to_geojson(arr, transform, "EPSG:32755", threshold=0.5, min_area_ha=0.1)
    # Blob 1: radius 20 px * 10 m = 200 m -> ~pi*200^2 = 12.6 ha. Allow slack for
    # rasterisation + simplification.
    areas = sorted(f["properties"]["area_ha"] for f in fc["features"])
    assert 5 < areas[-1] < 20


def test_raster_to_geojson_rejects_3d():
    with pytest.raises(ValueError):
        raster_to_geojson(np.zeros((2, 4, 4)), None, "EPSG:4326")
