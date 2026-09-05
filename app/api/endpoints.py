"""API routes for the fire-detection service (spec Task 6.2)."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import rasterio
import torch
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.schemas import (
    BandDrift,
    DetectionResponse,
    DriftRequest,
    DriftResponse,
    GeoJSONFeatureCollection,
    HealthResponse,
)
from app.monitoring.drift import ks_drift, load_baseline
from src.models.infer import detect
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(request: Request) -> HealthResponse:
    """Service status, GPU availability and loaded-model metadata."""
    state = request.app.state
    card = getattr(state, "model_card", {}) or {}
    model_loaded = getattr(state, "model", None) is not None
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        gpu_available=torch.cuda.is_available(),
        device=getattr(state, "device_str", "cpu"),
        model_loaded=model_loaded,
        model_version=card.get("git_sha") or card.get("export_mode"),
        encoder=card.get("encoder"),
        feature_bands=card.get("feature_bands", []),
    )


@router.post("/api/v1/detect", response_model=DetectionResponse, tags=["inference"])
async def detect_endpoint(request: Request, file: UploadFile = File(...)) -> DetectionResponse:
    """Detect active fire fronts in an uploaded multi-band SAR GeoTIFF.

    The upload must be a GeoTIFF with at least ``model.in_channels`` bands in the
    order given by ``features.selected_bands``. Returns a GeoJSON
    ``FeatureCollection`` of fire-perimeter polygons with confidence + area.
    """
    cfg = request.app.state.cfg
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on this instance")

    raw = await file.read()
    max_bytes = cfg.api.max_upload_mb * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Upload exceeds {cfg.api.max_upload_mb} MB")

    suffix = Path(file.filename or "upload.tif").suffix or ".tif"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)

    try:
        with rasterio.open(tmp_path) as ds:
            if ds.count < cfg.model.in_channels:
                raise HTTPException(
                    status_code=422,
                    detail=f"GeoTIFF has {ds.count} bands; need >= {cfg.model.in_channels}",
                )
    except HTTPException:
        raise
    except Exception as exc:  # not a readable raster
        raise HTTPException(status_code=422, detail=f"Unreadable GeoTIFF: {exc}") from exc

    t0 = time.perf_counter()
    try:
        # Vectorisation + inference are CPU/GPU-bound and blocking -> threadpool.
        geojson = await run_in_threadpool(
            detect,
            tmp_path,
            cfg,
            model_path=request.app.state.model_path,
            device_str=request.app.state.device_str,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    props = geojson.get("properties", {})
    return DetectionResponse(
        detections=GeoJSONFeatureCollection(**geojson),
        n_features=props.get("n_features", len(geojson.get("features", []))),
        total_area_ha=props.get("total_area_ha", 0.0),
        threshold=props.get("threshold", cfg.api.confidence_threshold),
        inference_ms=round(elapsed_ms, 1),
        source_filename=file.filename or "upload.tif",
    )


@router.post("/api/v1/monitor/drift", response_model=DriftResponse, tags=["monitoring"])
async def drift_endpoint(request: Request, payload: DriftRequest) -> DriftResponse:
    """Two-sample KS test of incoming backscatter distributions vs the baseline.

    Returns ``drift_detected=True`` if any band's p-value falls below ``alpha``
    (default ``api.drift_alpha``).
    """
    cfg = request.app.state.cfg
    alpha = payload.alpha if payload.alpha is not None else cfg.api.drift_alpha
    baseline = load_baseline(cfg)
    if not baseline:
        raise HTTPException(
            status_code=503,
            detail="No drift baseline available. Build one with app.monitoring.drift.build_baseline().",
        )
    result = await run_in_threadpool(ks_drift, payload.incoming, baseline, alpha=alpha)
    return DriftResponse(
        drift_detected=result["drift_detected"],
        alpha=result["alpha"],
        bands=[BandDrift(**b) for b in result["bands"]],
    )
