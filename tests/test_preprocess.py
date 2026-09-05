"""Tests for radiometric calibration and the Refined Lee speckle filter."""

from __future__ import annotations

import numpy as np
import rasterio

from src.data.preprocess import calibrate_and_filter, db_to_linear, refined_lee, to_db


def test_to_db_matches_closed_form():
    dn = np.array([[1.0, 10.0], [100.0, 1000.0]], dtype=np.float64)
    expected = 10.0 * np.log10(dn**2 + 1e-10)
    np.testing.assert_allclose(to_db(dn), expected, rtol=1e-5)


def test_to_db_is_idempotent_on_db_input():
    db = np.array([[-12.0, -18.0], [-9.5, -20.1]], dtype=np.float32)
    np.testing.assert_allclose(to_db(db), db, rtol=1e-6)


def test_db_linear_roundtrip():
    db = np.linspace(-30, 5, 50).astype(np.float32)
    back = 10.0 * np.log10(db_to_linear(db))
    np.testing.assert_allclose(back, db, atol=1e-4)


def test_refined_lee_reduces_variance_in_flat_region():
    rng = np.random.default_rng(0)
    flat_linear = rng.gamma(shape=4.0, scale=1 / 4.0, size=(128, 128)).astype(np.float32)
    flat_db = 10.0 * np.log10(flat_linear)
    filtered = refined_lee(flat_db, window=5, enl=4.0)
    assert filtered.shape == flat_db.shape
    assert np.nanvar(filtered) < np.nanvar(flat_db) * 0.8


def test_refined_lee_preserves_a_step_edge():
    img = np.full((64, 64), -18.0, dtype=np.float32)
    img[:, 32:] = -8.0  # 10 dB step
    filtered = refined_lee(img, window=5, enl=4.0)
    # Edge contrast should survive (allow some smoothing).
    left = filtered[:, 20:28].mean()
    right = filtered[:, 36:44].mean()
    assert (right - left) > 7.0


def test_calibrate_and_filter_writes_georeferenced_raster(tiny_config, raw_manifest):
    scene = raw_manifest["scenes"][0]
    out = tiny_config.data.processed_path / "cal_test.tif"
    calibrate_and_filter(scene["t0"], out, tiny_config, dst_crs=scene["crs"])
    assert out.is_file()
    with rasterio.open(out) as ds:
        assert ds.count == 2
        assert ds.crs is not None
        assert ds.dtypes[0] == "float32"
        data = ds.read(1)
    assert np.isfinite(data).any()
