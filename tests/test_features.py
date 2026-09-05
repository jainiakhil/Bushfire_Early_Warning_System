"""Tests for the 6-channel feature stack and feature-selection diagnostics."""

from __future__ import annotations

import numpy as np

from src.features.selection import compute_vif, mutual_information
from src.features.texture import build_feature_stack, glcm_contrast


def test_feature_stack_shape_and_band_names(tiny_config, processed_manifest):
    scene = processed_manifest["scenes"][0]
    stack, profile, names = build_feature_stack(scene["t0_cal"], scene["t1_cal"], tiny_config)
    assert names == tiny_config.features.selected_bands
    assert stack.shape[0] == len(names)
    assert stack.dtype == np.float32
    assert np.isfinite(stack).all()
    assert profile["count"] == len(names)


def test_delta_channels_are_t1_minus_t0(tiny_config, processed_manifest):
    import rasterio

    scene = processed_manifest["scenes"][0]
    with rasterio.open(scene["t0_cal"]) as ds:
        vv0, vh0 = ds.read(1), ds.read(2)
    with rasterio.open(scene["t1_cal"]) as ds:
        vv1, vh1 = ds.read(1), ds.read(2)

    stack, _, names = build_feature_stack(scene["t0_cal"], scene["t1_cal"], tiny_config)
    d_vv = stack[names.index("delta_vv")]
    np.testing.assert_allclose(np.nan_to_num(d_vv), np.nan_to_num(vv1 - vv0), atol=1e-4)
    d_vh = stack[names.index("delta_vh")]
    np.testing.assert_allclose(np.nan_to_num(d_vh), np.nan_to_num(vh1 - vh0), atol=1e-4)


def test_glcm_contrast_higher_on_textured_patch():
    rng = np.random.default_rng(1)
    flat = np.full((48, 48), -15.0, dtype=np.float32)
    textured = flat.copy()
    textured[12:36, 12:36] += rng.uniform(-6, 6, size=(24, 24)).astype(np.float32)
    c = glcm_contrast(textured, window=5, levels=16)
    assert c[24, 24] > c[3, 3]


def test_mutual_information_ranks_informative_feature_first():
    rng = np.random.default_rng(2)
    h = w = 64
    label = (rng.random((h, w)) < 0.2).astype(np.int64)
    informative = label + rng.normal(0, 0.1, (h, w))
    noise = rng.normal(0, 1, (h, w))
    stack = np.stack([informative, noise]).astype(np.float32)
    mi = mutual_information(stack, label, ["informative", "noise"], n_samples=3000, seed=0)
    assert mi["informative"] > mi["noise"]


def test_vif_flags_a_duplicated_column():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(2000,)).astype(np.float32)
    b = rng.normal(size=(2000,)).astype(np.float32)
    dup = a.copy()
    stack = np.stack([a, b, dup]).reshape(3, 2000, 1)
    vif = compute_vif(stack, ["a", "b", "dup"])
    assert vif["a"] >= 10.0 and vif["dup"] >= 10.0
    assert vif["b"] < 10.0
