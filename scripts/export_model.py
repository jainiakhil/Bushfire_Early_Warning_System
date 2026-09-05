"""CLI: export a trained checkpoint to TorchScript for production inference.

Example:
    python scripts/export_model.py --checkpoint data/outputs/checkpoints/fold0_best.pt
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch

from src.config import load_config
from src.models.unet import build_model, export_torchscript
from src.utils.logging import get_logger

logger = get_logger("export_model")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", required=True, help="Path to a fold*_best.pt checkpoint")
    parser.add_argument("--out", default=None, help="Output .torchscript path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])

    out = Path(args.out) if args.out else (cfg.data.output_path / "model.torchscript")
    export_torchscript(
        model, cfg, out,
        metrics=ckpt.get("metrics", {}),
        norm_stats=ckpt.get("norm_stats"),
        git_sha=_git_sha(),
    )
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
