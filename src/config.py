"""Typed configuration for the SAR bushfire pipeline.

The whole pipeline is driven by a single YAML file (``configs/config.yaml``).
This module parses it into nested `pydantic` models so that:

* every consumer gets attribute access with IDE completion and type checking;
* invalid values fail fast at load time with a clear error;
* any field can be overridden by an environment variable using the pattern
  ``SAR__<SECTION>__<KEY>`` (double underscore as the nesting delimiter), which
  keeps CI, Docker and ad-hoc experiments from having to edit the YAML.

Example:
    >>> from src.config import load_config
    >>> cfg = load_config()                     # reads configs/config.yaml
    >>> cfg.training.batch_size
    8
    >>> import os; os.environ["SAR__TRAINING__BATCH_SIZE"] = "4"
    >>> load_config().training.batch_size       # env override wins
    4
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

# Repository root = two levels up from this file (src/config.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"

# Environment-variable override prefix and nesting delimiter.
_ENV_PREFIX = "SAR__"
_ENV_DELIM = "__"


class DataCfg(BaseModel):
    """Filesystem layout, sensor geometry and coordinate reference systems."""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    output_dir: str = "data/outputs"
    spatial_resolution: float = 10.0
    tile_size: int = 256
    tile_overlap: int = 32
    source: Literal["synthetic", "cdse"] = "synthetic"
    crs_working: str = "auto-utm"
    crs_output: str = "EPSG:4326"

    @property
    def raw_path(self) -> Path:
        """Absolute path to the raw-data directory."""
        return _abs(self.raw_dir)

    @property
    def processed_path(self) -> Path:
        """Absolute path to the processed-data directory."""
        return _abs(self.processed_dir)

    @property
    def output_path(self) -> Path:
        """Absolute path to the outputs directory."""
        return _abs(self.output_dir)


class FeaturesCfg(BaseModel):
    """Feature-stack composition and texture/speckle parameters."""

    selected_bands: list[str] = Field(
        default_factory=lambda: [
            "sigma0_vv",
            "sigma0_vh",
            "delta_vv",
            "delta_vh",
            "pol_ratio",
            "glcm_contrast",
        ]
    )
    glcm_window: int = 7
    glcm_levels: int = 32
    refined_lee_window: int = 5
    equivalent_looks: float = 4.0
    mi_sample_pixels: int = 200_000
    vif_threshold: float = 10.0


class TrainingCfg(BaseModel):
    """Optimisation, cross-validation and hardware settings."""

    batch_size: int = 8
    grad_accum_steps: int = 2
    num_workers: int = 4
    learning_rate: float = 3e-4
    epochs: int = 40
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    dice_weight: float = 1.0
    spatial_block_size_km: float = 25.0
    spatial_buffer_km: float = 5.0
    n_folds: int = 5
    early_stopping_patience: int = 8
    device: Literal["auto", "cuda", "cpu"] = "auto"
    precision: Literal["16-mixed", "32"] = "16-mixed"
    seed: int = 42

    @property
    def effective_batch_size(self) -> int:
        """Batch size seen by the optimiser after gradient accumulation."""
        return self.batch_size * max(1, self.grad_accum_steps)


class ModelCfg(BaseModel):
    """Segmentation architecture definition."""

    arch: Literal["unet"] = "unet"
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    in_channels: int = 6
    classes: int = 1
    bottleneck_dropout: float = 0.2


class MlflowCfg(BaseModel):
    """MLflow experiment-tracking backend."""

    tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    experiment: str = "sar-bushfire"

    def resolved_tracking_uri(self) -> str:
        """Anchor a relative ``sqlite:///`` path to the repo root and mkdir it.

        Absolute paths and non-sqlite URIs (http, file, postgresql, ...) pass
        through unchanged.
        """
        uri = self.tracking_uri
        prefix = "sqlite:///"
        if uri.startswith(prefix):
            db_path = Path(uri[len(prefix) :])
            if not db_path.is_absolute():
                db_path = REPO_ROOT / db_path
            db_path.parent.mkdir(parents=True, exist_ok=True)
            return f"{prefix}{db_path.as_posix()}"
        return uri


class SyntheticCfg(BaseModel):
    """Synthetic-scene generator parameters (credential-free development data)."""

    n_scenes: int = 8
    image_size: int = 1024
    n_fires: int = 3
    noise_looks: int = 4
    seed: int = 42


class ApiCfg(BaseModel):
    """FastAPI service and vectorisation parameters."""

    host: str = "0.0.0.0"
    port: int = 8000
    confidence_threshold: float = 0.65
    simplify_tolerance: float = 1e-4
    min_polygon_area_ha: float = 0.5
    max_upload_mb: int = 200
    drift_alpha: float = 0.01


class Config(BaseModel):
    """Root configuration object."""

    data: DataCfg = Field(default_factory=DataCfg)
    features: FeaturesCfg = Field(default_factory=FeaturesCfg)
    training: TrainingCfg = Field(default_factory=TrainingCfg)
    model: ModelCfg = Field(default_factory=ModelCfg)
    mlflow: MlflowCfg = Field(default_factory=MlflowCfg)
    synthetic: SyntheticCfg = Field(default_factory=SyntheticCfg)
    api: ApiCfg = Field(default_factory=ApiCfg)

    @model_validator(mode="after")
    def _check_channel_count(self) -> Config:
        """Guard against a mismatch between the feature list and the model input."""
        n_bands = len(self.features.selected_bands)
        if self.model.in_channels != n_bands:
            raise ValueError(
                f"model.in_channels ({self.model.in_channels}) must equal the number of "
                f"features.selected_bands ({n_bands}): {self.features.selected_bands}"
            )
        return self


def _abs(path_str: str) -> Path:
    """Resolve *path_str* against the repo root unless it is already absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _apply_env_overrides(raw: dict) -> dict:
    """Overlay ``SAR__SECTION__KEY`` environment variables onto the parsed YAML.

    Values are decoded as YAML scalars so ``"null"``, ``"true"`` and ``"3"`` are
    coerced to ``None``, ``bool`` and ``int`` respectively.
    """
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(_ENV_PREFIX):
            continue
        parts = env_key[len(_ENV_PREFIX) :].lower().split(_ENV_DELIM)
        if len(parts) < 2:
            continue
        cursor = raw
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):  # pragma: no cover - defensive
                break
        else:
            cursor[parts[-1]] = yaml.safe_load(env_val)
    return raw


def load_config(path: str | Path | None = None, *, use_env: bool = True) -> Config:
    """Load and validate the pipeline configuration.

    Args:
        path: Path to a YAML config file. Defaults to ``configs/config.yaml`` at
            the repository root.
        use_env: When True, apply ``SAR__*`` environment-variable overrides.

    Returns:
        A fully validated :class:`Config` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        pydantic.ValidationError: If any value fails validation.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if use_env:
        raw = _apply_env_overrides(raw)
    return Config.model_validate(raw)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return a process-wide cached config (handy for FastAPI dependencies)."""
    return load_config()
