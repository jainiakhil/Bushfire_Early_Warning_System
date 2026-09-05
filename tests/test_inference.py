"""End-to-end inference tests: tiled prediction, the FastAPI app, drift endpoint."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient

from app.main import create_app
from src.models.infer import load_model, predict_geotiff


@pytest.fixture(scope="module")
def api_client(tiny_config, trained_model):
    """A TestClient whose app lifespan is bound to the tiny_config workspace."""
    _result, ts_path = trained_model
    import app.main as app_main

    original = app_main.load_config
    app_main.load_config = lambda *a, **k: tiny_config  # lifespan reads this
    try:
        app = create_app()
        with TestClient(app) as client:
            yield client, ts_path
    finally:
        app_main.load_config = original


def test_predict_geotiff_output_range(tiny_config, trained_model, feature_manifest):
    _result, ts_path = trained_model
    model = load_model(str(ts_path), "cpu")
    scene = feature_manifest["scenes"][0]
    prob, transform, crs = predict_geotiff(scene["feature_path"], model, tiny_config, device_str="cpu")
    assert prob.ndim == 2
    assert prob.min() >= 0.0 and prob.max() <= 1.0
    assert crs is not None


def test_tiling_reconstructs_constant_field(tiny_config, trained_model, tmp_path):
    """A constant-input raster should yield a (near-)constant probability field."""
    _result, ts_path = trained_model
    model = load_model(str(ts_path), "cpu")

    n_ch = tiny_config.model.in_channels
    size = 200
    arr = np.zeros((n_ch, size, size), dtype=np.float32)
    profile = {
        "driver": "GTiff", "dtype": "float32", "count": n_ch,
        "height": size, "width": size, "crs": "EPSG:32755",
        "transform": rasterio.transform.from_origin(700000, 6100000, 10, 10),
    }
    p = tmp_path / "const.tif"
    with rasterio.open(p, "w", **profile) as ds:
        ds.write(arr)

    prob, _, _ = predict_geotiff(p, model, tiny_config, device_str="cpu")
    interior = prob[20:-20, 20:-20]
    assert interior.std() < 0.05


def test_health_endpoint(api_client):
    client, _ = api_client
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["feature_bands"]


def test_detect_endpoint_returns_feature_collection(api_client, feature_manifest):
    client, _ = api_client
    scene = feature_manifest["scenes"][0]
    with open(scene["feature_path"], "rb") as f:
        r = client.post("/api/v1/detect", files={"file": ("scene.tif", f, "image/tiff")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["detections"]["type"] == "FeatureCollection"
    assert body["n_features"] == len(body["detections"]["features"])
    assert body["inference_ms"] > 0


def test_detect_rejects_too_few_bands(api_client, tmp_path):
    client, _ = api_client
    p = tmp_path / "twoband.tif"
    profile = {
        "driver": "GTiff", "dtype": "float32", "count": 2,
        "height": 32, "width": 32, "crs": "EPSG:4326",
        "transform": rasterio.transform.from_origin(0, 0, 0.01, 0.01),
    }
    with rasterio.open(p, "w", **profile) as ds:
        ds.write(np.zeros((2, 32, 32), dtype=np.float32))
    with open(p, "rb") as f:
        r = client.post("/api/v1/detect", files={"file": ("twoband.tif", f, "image/tiff")})
    assert r.status_code == 422


def test_drift_endpoint_flags_shifted_distribution(api_client, tiny_config, feature_manifest):
    client, _ = api_client
    from app.monitoring.drift import build_baseline

    build_baseline(tiny_config, feature_manifest)

    # Same distribution -> no drift.
    baseline = tiny_config.data.processed_path / "baseline_backscatter.json"
    import json

    ref = json.loads(baseline.read_text())
    band = next(iter(ref))
    same = {band: ref[band]}
    r = client.post("/api/v1/monitor/drift", json={"incoming": same})
    assert r.status_code == 200
    assert r.json()["drift_detected"] is False

    # Shift by +15 dB -> drift.
    shifted = {band: [v + 15.0 for v in ref[band]]}
    r = client.post("/api/v1/monitor/drift", json={"incoming": shifted})
    assert r.json()["drift_detected"] is True
