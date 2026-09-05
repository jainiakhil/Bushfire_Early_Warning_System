"""Pydantic request/response models for the inference API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Payload for ``GET /health``."""

    status: Literal["ok", "degraded"] = "ok"
    gpu_available: bool
    device: str
    model_loaded: bool
    model_version: str | None = None
    encoder: str | None = None
    feature_bands: list[str] = Field(default_factory=list)
    api_version: str = "v1"


class GeoJSONFeatureCollection(BaseModel):
    """A minimal GeoJSON ``FeatureCollection`` wrapper.

    The individual geometries are left as free-form dicts (GeoJSON is verbose to
    type fully) but the envelope is validated so clients get a stable contract.
    """

    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[dict[str, Any]]
    properties: dict[str, Any] = Field(default_factory=dict)


class DetectionResponse(BaseModel):
    """Payload for ``POST /api/v1/detect``."""

    detections: GeoJSONFeatureCollection
    n_features: int
    total_area_ha: float
    threshold: float
    inference_ms: float
    source_filename: str


class DriftRequest(BaseModel):
    """Payload for ``POST /api/v1/monitor/drift``.

    Either provide raw per-band samples in *incoming* or rely on the server's
    stored baseline (built from the training set).
    """

    incoming: dict[str, list[float]] = Field(
        ..., description="Band name -> flat list of backscatter samples (dB)."
    )
    alpha: float | None = Field(
        None, description="Override the KS-test p-value alert threshold."
    )


class BandDrift(BaseModel):
    """Per-band KS-test result."""

    band: str
    ks_statistic: float
    p_value: float
    drift: bool
    n_incoming: int
    n_baseline: int


class DriftResponse(BaseModel):
    """Payload returned by the drift endpoint."""

    drift_detected: bool
    alpha: float
    bands: list[BandDrift]
