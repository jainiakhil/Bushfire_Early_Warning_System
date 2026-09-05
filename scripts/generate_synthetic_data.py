"""CLI: generate a credential-free synthetic Sentinel-1 dataset.

Example:
    python scripts/generate_synthetic_data.py --n-scenes 8 --out data/raw
"""

from __future__ import annotations

import argparse

from src.config import load_config
from src.data.synthetic import generate_dataset
from src.utils.logging import get_logger

logger = get_logger("generate_synthetic_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--n-scenes", type=int, default=None, help="Number of scene pairs")
    parser.add_argument("--out", default=None, help="Output directory (default: data.raw_dir)")
    parser.add_argument("--seed", type=int, default=None, help="Override synthetic.seed")
    parser.add_argument("--image-size", type=int, default=None, help="Override synthetic.image_size")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.synthetic.seed = args.seed
    if args.image_size is not None:
        cfg.synthetic.image_size = args.image_size

    out_dir = None
    if args.out:
        from pathlib import Path

        out_dir = Path(args.out)

    path = generate_dataset(cfg, n_scenes=args.n_scenes, out_dir=out_dir)
    logger.info("Done. Manifest: %s", path)


if __name__ == "__main__":
    main()
