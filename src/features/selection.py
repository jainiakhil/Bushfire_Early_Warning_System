r"""Feature-selection diagnostics (spec Module 3, Task 3.2).

Two independent checks help justify the ``features.selected_bands`` list:

* **Mutual information** between each candidate feature and the ground-truth
  label, estimated on a random pixel sample
  (:func:`sklearn.feature_selection.mutual_info_classif`).
* **Variance Inflation Factor (VIF)** to detect multicollinearity; a feature
  with :math:`\\mathrm{VIF} \\ge 10` is largely predictable from the others and
  is a candidate for removal.

These are run offline (see ``scripts/`` / notebooks). The training path just
consumes the curated list from config.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LinearRegression

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _sample_valid_pixels(
    feature_stack: np.ndarray, label_mask: np.ndarray, n_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten and randomly sample finite pixels.

    Args:
        feature_stack: ``(C, H, W)`` feature array.
        label_mask: ``(H, W)`` binary label array.
        n_samples: Target number of sampled pixels.
        seed: RNG seed.

    Returns:
        ``(X, y)`` with ``X`` shape ``(n, C)`` and ``y`` shape ``(n,)``.
    """
    c = feature_stack.shape[0]
    flat = feature_stack.reshape(c, -1).T           # (H*W, C)
    labels = label_mask.reshape(-1).astype(int)     # (H*W,)
    finite = np.isfinite(flat).all(axis=1)
    flat, labels = flat[finite], labels[finite]

    rng = np.random.default_rng(seed)
    n = min(n_samples, flat.shape[0])
    # Stratify lightly so rare fire pixels are represented.
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    n_pos = min(len(pos_idx), n // 2)
    n_neg = n - n_pos
    take = np.concatenate(
        [rng.choice(pos_idx, n_pos, replace=False) if n_pos else np.array([], int),
         rng.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False)]
    )
    rng.shuffle(take)
    return flat[take], labels[take]


def mutual_information(
    feature_stack: np.ndarray,
    label_mask: np.ndarray,
    band_names: list[str],
    *,
    n_samples: int = 200_000,
    seed: int = 42,
) -> dict[str, float]:
    """Mutual information between each feature and the binary label.

    Args:
        feature_stack: ``(C, H, W)`` feature array.
        label_mask: ``(H, W)`` binary fire mask.
        band_names: Names for the C channels (length must match).
        n_samples: Pixels to sample.
        seed: RNG seed.

    Returns:
        ``{band_name: mi_score}`` sorted descending by score.
    """
    if len(band_names) != feature_stack.shape[0]:
        raise ValueError("band_names length must equal number of channels")
    x, y = _sample_valid_pixels(feature_stack, label_mask, n_samples, seed)
    if len(np.unique(y)) < 2:
        logger.warning("Only one class present in sample; MI scores are meaningless")
        return {name: 0.0 for name in band_names}
    mi = mutual_info_classif(x, y, random_state=seed)
    scores = dict(sorted(zip(band_names, mi.tolist(), strict=False), key=lambda kv: kv[1], reverse=True))
    logger.info("Mutual information: %s", {k: round(v, 4) for k, v in scores.items()})
    return scores


def compute_vif(feature_stack: np.ndarray, band_names: list[str], *, sample: int = 50_000, seed: int = 42) -> dict[str, float]:
    r"""Variance Inflation Factor per feature.

    :math:`\\mathrm{VIF}_i = 1 / (1 - R_i^2)` where :math:`R_i^2` is from
    regressing feature *i* on all the others. No statsmodels dependency: a plain
    least-squares fit is enough.

    Args:
        feature_stack: ``(C, H, W)`` feature array.
        band_names: Channel names.
        sample: Pixels to sample for the regressions.
        seed: RNG seed.

    Returns:
        ``{band_name: vif}``. ``inf`` means perfect collinearity.
    """
    c = feature_stack.shape[0]
    flat = feature_stack.reshape(c, -1).T
    flat = flat[np.isfinite(flat).all(axis=1)]
    rng = np.random.default_rng(seed)
    if flat.shape[0] > sample:
        flat = flat[rng.choice(flat.shape[0], sample, replace=False)]
    # Standardise so the regression R^2 is scale-free.
    flat = (flat - flat.mean(0)) / (flat.std(0) + 1e-9)

    vifs: dict[str, float] = {}
    for i, name in enumerate(band_names):
        others = np.delete(flat, i, axis=1)
        target = flat[:, i]
        r2 = LinearRegression().fit(others, target).score(others, target)
        vifs[name] = float("inf") if r2 >= 1.0 - 1e-9 else 1.0 / (1.0 - r2)
    for name, v in vifs.items():
        if v >= 10.0:
            logger.warning("VIF(%s) = %.2f  >= 10  (multicollinear)", name, v)
    return vifs


def select_features(
    feature_stack: np.ndarray,
    label_mask: np.ndarray,
    band_names: list[str],
    *,
    vif_threshold: float = 10.0,
    n_samples: int = 200_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Rank features by MI and prune the more collinear member of each bad pair.

    Returns:
        A report dict: ``{"mi": {...}, "vif": {...}, "kept": [...], "dropped": [...]}``.
    """
    mi = mutual_information(feature_stack, label_mask, band_names, n_samples=n_samples, seed=seed)
    vif = compute_vif(feature_stack, band_names, seed=seed)

    kept = list(band_names)
    dropped: list[str] = []
    # Greedy: while any kept feature exceeds the VIF threshold, drop the one with
    # the lowest MI among the offenders, then recompute VIF on the survivors.
    while True:
        idx = [band_names.index(b) for b in kept]
        sub = feature_stack[idx]
        vif = compute_vif(sub, kept, seed=seed)
        offenders = [b for b, v in vif.items() if v >= vif_threshold]
        if not offenders or len(kept) <= 2:
            break
        worst = min(offenders, key=lambda b: mi.get(b, 0.0))
        kept.remove(worst)
        dropped.append(worst)
        logger.info("Dropped %s (VIF=%.1f, MI=%.4f)", worst, vif[worst], mi.get(worst, 0.0))

    return {"mi": mi, "vif": vif, "kept": kept, "dropped": dropped}
