"""Backscatter distribution drift detection (spec Task 6.2, ``/monitor/drift``).

A two-sample Kolmogorov-Smirnov test compares the distribution of each incoming
backscatter band against a stored baseline (built from the training data). A
small p-value means the two samples are unlikely to come from the same
distribution - i.e. the input statistics have drifted and the model's
predictions should be treated with caution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ks_2samp

from src.config import Config
from src.utils.logging import get_logger

logger = get_logger(__name__)

_BASELINE_FILENAME = "baseline_backscatter.json"
_MAX_SAMPLES = 20_000  # KS test is O(n log n); cap for responsiveness


def baseline_path(cfg: Config) -> Path:
    """Location of the baseline reference file."""
    return cfg.data.processed_path / _BASELINE_FILENAME


def build_baseline(cfg: Config, feature_manifest: dict[str, Any], *, per_scene_samples: int = 5000) -> Path:
    """Sample per-band backscatter values from the feature stacks and persist them.

    Args:
        cfg: Pipeline config.
        feature_manifest: Manifest with ``feature_path`` per scene.
        per_scene_samples: Random pixels to draw from each scene per band.

    Returns:
        Path to the written baseline JSON (``{band: [values...]}``).
    """
    import rasterio

    bands = cfg.features.selected_bands
    rng = np.random.default_rng(cfg.training.seed)
    acc: dict[str, list[float]] = {b: [] for b in bands}

    for scene in feature_manifest["scenes"]:
        with rasterio.open(scene["feature_path"]) as ds:
            arr = ds.read().astype(np.float32)
        for i, b in enumerate(bands):
            v = arr[i][np.isfinite(arr[i])].ravel()
            if v.size == 0:
                continue
            take = rng.choice(v, size=min(per_scene_samples, v.size), replace=False)
            acc[b].extend(take.tolist())

    trimmed = {b: vals[:_MAX_SAMPLES] for b, vals in acc.items()}
    path = baseline_path(cfg)
    path.write_text(json.dumps(trimmed), encoding="utf-8")
    logger.info("Wrote drift baseline (%d bands) -> %s", len(trimmed), path)
    return path


def load_baseline(cfg: Config) -> dict[str, list[float]]:
    """Load the stored baseline, or return ``{}`` if it has not been built."""
    path = baseline_path(cfg)
    if not path.is_file():
        logger.warning("No drift baseline at %s; build one with build_baseline()", path)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def ks_drift(
    incoming: dict[str, list[float]],
    baseline: dict[str, list[float]],
    *,
    alpha: float = 0.01,
) -> dict[str, Any]:
    """Run a per-band two-sample KS test of *incoming* against *baseline*.

    Args:
        incoming: ``{band: [samples]}`` from a new acquisition.
        baseline: ``{band: [samples]}`` reference distributions.
        alpha: A band drifts if its KS p-value is ``< alpha``.

    Returns:
        ``{"drift_detected": bool, "alpha": float, "bands": [BandDrift-like dicts]}``.
    """
    results: list[dict[str, Any]] = []
    any_drift = False

    for band, inc_vals in incoming.items():
        base_vals = baseline.get(band, [])
        inc = np.asarray(inc_vals, dtype=np.float64)
        inc = inc[np.isfinite(inc)]
        base = np.asarray(base_vals, dtype=np.float64)
        base = base[np.isfinite(base)]

        if inc.size < 20 or base.size < 20:
            results.append(
                {
                    "band": band,
                    "ks_statistic": float("nan"),
                    "p_value": float("nan"),
                    "drift": False,
                    "n_incoming": int(inc.size),
                    "n_baseline": int(base.size),
                }
            )
            continue

        stat, p = ks_2samp(inc, base)
        drift = bool(p < alpha)
        any_drift = any_drift or drift
        results.append(
            {
                "band": band,
                "ks_statistic": float(stat),
                "p_value": float(p),
                "drift": drift,
                "n_incoming": int(inc.size),
                "n_baseline": int(base.size),
            }
        )

    logger.info("Drift check: detected=%s over %d bands (alpha=%.3g)", any_drift, len(results), alpha)
    return {"drift_detected": any_drift, "alpha": alpha, "bands": results}
