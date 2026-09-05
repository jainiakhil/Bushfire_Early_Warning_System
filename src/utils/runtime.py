"""Runtime helpers shared by training and inference: device + determinism."""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING

import numpy as np

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a device spec into a ``torch.device``.

    Args:
        spec: ``"auto"`` picks CUDA when available else CPU; ``"cuda"`` or
            ``"cpu"`` force the choice (``"cuda"`` falls back with a warning if
            no GPU is visible).

    Returns:
        A ``torch.device``. ``torch`` is imported lazily so importing this
        module stays cheap for non-torch code paths.
    """
    import torch

    spec = (spec or "auto").lower()
    cuda_ok = torch.cuda.is_available()
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not cuda_ok:
            logger.warning("device='cuda' requested but no CUDA GPU is visible; using CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    # auto
    device = torch.device("cuda" if cuda_ok else "cpu")
    if cuda_ok:
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
    else:
        logger.info("No CUDA GPU visible; using CPU")
    return device


def seed_everything(seed: int = 42, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if importable) PyTorch RNGs.

    Args:
        seed: The seed value.
        deterministic: When True, also set cuDNN to deterministic mode. This
            slows training but makes runs bit-reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass
    logger.debug("Seeded RNGs with %d (deterministic=%s)", seed, deterministic)
