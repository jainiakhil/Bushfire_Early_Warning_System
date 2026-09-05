"""CLI: train the U-Net for one spatial-CV fold and export TorchScript.

Example:
    python scripts/run_training.py --config configs/config.yaml --fold 0 --epochs 40
    # quick smoke run:
    SAR__TRAINING__DEVICE=cpu python scripts/run_training.py --epochs 1
"""

from __future__ import annotations

import argparse
import json

from src.config import load_config
from src.data.download import resolve_scenes
from src.data.preprocess import preprocess_manifest
from src.features.texture import build_features_for_manifest
from src.models.train import train_and_export
from src.utils.logging import get_logger

logger = get_logger("run_training")


def _feature_manifest(cfg):
    """Load feature_manifest.json, building the whole chain if it is missing."""
    path = cfg.data.processed_path / "feature_manifest.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    logger.info("feature_manifest.json missing; running preprocessing first")
    raw = resolve_scenes(cfg)
    processed = preprocess_manifest(cfg, raw)
    return build_features_for_manifest(cfg, processed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.training.epochs = args.epochs

    feature_manifest = _feature_manifest(cfg)
    result, ts_path = train_and_export(cfg, feature_manifest, fold=args.fold)

    logger.info("Best val IoU %.4f at epoch %d", result.best_iou, result.best_epoch)
    logger.info("Checkpoint: %s", result.checkpoint_path)
    logger.info("TorchScript: %s", ts_path)


if __name__ == "__main__":
    main()
