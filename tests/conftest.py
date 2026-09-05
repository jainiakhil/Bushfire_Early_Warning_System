"""Shared pytest fixtures.

Everything is built from the synthetic generator so the suite needs no network,
no credentials and no GPU. Heavy fixtures are session-scoped and use small tile /
GLCM sizes to keep the run fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.data.synthetic import generate_dataset


@pytest.fixture(scope="session")
def tiny_config(tmp_path_factory: pytest.TempPathFactory):
    """A Config pointed at a temp workspace with small, fast parameters."""
    ws = tmp_path_factory.mktemp("sar_ws")
    cfg = load_config(use_env=False)

    for sub in ("raw", "processed", "outputs"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    cfg.data.raw_dir = str(ws / "raw")
    cfg.data.processed_dir = str(ws / "processed")
    cfg.data.output_dir = str(ws / "outputs")

    cfg.synthetic.n_scenes = 3
    cfg.synthetic.image_size = 160
    cfg.synthetic.n_fires = 2

    cfg.data.tile_size = 96
    cfg.data.tile_overlap = 16
    cfg.features.glcm_window = 5
    cfg.features.glcm_levels = 16

    cfg.training.epochs = 1
    cfg.training.batch_size = 2
    cfg.training.grad_accum_steps = 1
    cfg.training.num_workers = 0
    cfg.training.n_folds = 2
    cfg.training.device = "cpu"
    cfg.training.precision = "32"
    cfg.model.encoder_weights = None

    cfg.mlflow.tracking_uri = f"sqlite:///{(ws / 'mlflow.db').as_posix()}"
    return cfg


@pytest.fixture(scope="session")
def raw_manifest(tiny_config) -> dict:
    """Generate the synthetic raw dataset once for the session."""
    path = generate_dataset(tiny_config)
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def processed_manifest(tiny_config, raw_manifest) -> dict:
    """Calibrated + speckle-filtered scenes."""
    from src.data.preprocess import preprocess_manifest

    return preprocess_manifest(tiny_config, raw_manifest)


@pytest.fixture(scope="session")
def feature_manifest(tiny_config, processed_manifest) -> dict:
    """6-channel feature stacks."""
    from src.features.texture import build_features_for_manifest

    return build_features_for_manifest(tiny_config, processed_manifest)


@pytest.fixture(scope="session")
def trained_model(tiny_config, feature_manifest):
    """Train one fold for a single epoch and export TorchScript.

    Returns ``(TrainResult, torchscript_path)``.
    """
    from src.models.train import train_and_export

    return train_and_export(tiny_config, feature_manifest, fold=0)


@pytest.fixture
def two_blob_probability() -> tuple[np.ndarray, object]:
    """A 200x200 probability raster with two clean high-probability blobs."""
    from affine import Affine

    arr = np.zeros((200, 200), dtype=np.float32)
    yy, xx = np.mgrid[0:200, 0:200]
    arr[((yy - 50) ** 2 + (xx - 50) ** 2) < 20**2] = 0.9
    arr[((yy - 150) ** 2 + (xx - 140) ** 2) < 15**2] = 0.8
    # 10 m pixels, placed somewhere in UTM zone 55S (SE Australia).
    transform = Affine(10.0, 0.0, 700_000.0, 0.0, -10.0, 6_100_000.0)
    return arr, transform
