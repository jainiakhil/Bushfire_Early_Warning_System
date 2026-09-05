"""CLI: calibrate + speckle-filter scenes, then build 6-channel feature stacks.

Reads the raw manifest (generating a synthetic one if ``data.source: synthetic``
and none exists), writes calibrated GeoTIFFs and feature stacks into
``data/processed``, and emits ``processed_manifest.json`` + ``feature_manifest.json``.

Example:
    python scripts/run_preprocess.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse

from src.config import load_config
from src.data.download import resolve_scenes
from src.data.preprocess import preprocess_manifest
from src.features.texture import build_features_for_manifest
from src.utils.logging import get_logger

logger = get_logger("run_preprocess")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-features", action="store_true", help="Only calibrate; skip GLCM/feature stack")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info("Data source: %s", cfg.data.source)

    raw_manifest = resolve_scenes(cfg)
    processed = preprocess_manifest(cfg, raw_manifest)

    if not args.skip_features:
        build_features_for_manifest(cfg, processed)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
