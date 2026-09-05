"""FastAPI application factory for the SAR bushfire detection service.

The model is loaded once at startup (``lifespan``) and stored on ``app.state``.
If no exported model is present the service still starts in a *degraded* state so
``/health`` is reachable (useful for orchestration probes and CI).
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router
from src.config import load_config
from src.models.infer import load_model, load_model_card
from src.utils.logging import get_logger
from src.utils.runtime import resolve_device

logger = get_logger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load config + TorchScript model into ``app.state`` for the process lifetime."""
    cfg = load_config()
    app.state.cfg = cfg

    device = resolve_device(cfg.training.device)
    app.state.device_str = device.type

    model_path = Path(cfg.data.output_path) / "model.torchscript"
    app.state.model_path = model_path
    if model_path.is_file():
        app.state.model = load_model(str(model_path), device.type)
        app.state.model_card = load_model_card(model_path)
        logger.info("Model ready (%s) on %s", model_path.name, device.type)
    else:
        app.state.model = None
        app.state.model_card = {}
        logger.warning("No model at %s - service starts DEGRADED (train + export first)", model_path)

    yield

    app.state.model = None


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="Sentinel-1 SAR Bushfire Detection API",
        version="1.0.0",
        description=(
            "Detect active fire fronts in Sentinel-1 dual-pol SAR imagery. "
            "Upload a 6-band feature GeoTIFF to /api/v1/detect and receive GeoJSON "
            "fire-perimeter polygons."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # dashboard runs on a different port; tighten in prod
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
