"""Spatial block cross-validation (spec Module 4, Task 4.1).

Random k-fold splits leak information in gridded geospatial data because
neighbouring pixels are spatially autocorrelated: a pixel in the training set
that sits next to a test pixel makes the test score optimistic.

:class:`SpatialBlockKFold` avoids this by

1. tiling the study area into square blocks (default 25 km);
2. assigning whole blocks to folds;
3. discarding training samples within a buffer (default 5 km) of any test block,
   so no training sample is spatially adjacent to a test sample.

The class mirrors the scikit-learn splitter API (``get_n_splits`` / ``split``)
so it drops into existing cross-validation loops.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class SpatialBlockKFold:
    """Block-wise k-fold splitter with an exclusion buffer.

    Args:
        n_splits: Number of folds.
        block_size_km: Edge length of each spatial block, in kilometres.
        buffer_km: Width of the exclusion buffer around test blocks, in
            kilometres. Training samples inside this buffer are dropped.
        shuffle: Shuffle block-to-fold assignment.
        random_state: Seed for the shuffle.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        block_size_km: float = 25.0,
        buffer_km: float = 5.0,
        shuffle: bool = True,
        random_state: int | None = 42,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.block_size_m = block_size_km * 1_000.0
        self.buffer_m = buffer_km * 1_000.0
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: N803 - sklearn API
        """Return the number of splitting iterations (sklearn compatibility)."""
        return self.n_splits

    def _block_ids(self, xy: np.ndarray) -> np.ndarray:
        """Map each ``(x, y)`` coordinate (projected, metres) to a block index.

        Returns an integer array of shape ``(n_samples,)`` where equal values
        mean "same spatial block".
        """
        bx = np.floor(xy[:, 0] / self.block_size_m).astype(np.int64)
        by = np.floor(xy[:, 1] / self.block_size_m).astype(np.int64)
        # Combine the 2-D block coordinate into a single hashable id.
        bx_shift = bx - bx.min()
        by_shift = by - by.min()
        stride = by_shift.max() + 1
        return bx_shift * stride + by_shift

    def split(
        self, xy: np.ndarray, y: np.ndarray | None = None, groups=None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` index arrays for each fold.

        Args:
            xy: ``(n_samples, 2)`` array of projected coordinates in **metres**
                (e.g. tile-centre easting/northing in the working UTM CRS).
            y: Ignored; present for API compatibility.
            groups: Ignored.

        Yields:
            Tuples of 1-D integer index arrays into *xy*. Every sample appears in
            exactly one fold's test set; buffered samples appear in no train set
            for that fold.
        """
        xy = np.asarray(xy, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"xy must have shape (n_samples, 2), got {xy.shape}")
        n = xy.shape[0]

        block_ids = self._block_ids(xy)
        unique_blocks = np.unique(block_ids)
        rng = np.random.default_rng(self.random_state)
        if self.shuffle:
            rng.shuffle(unique_blocks)

        # Round-robin blocks into folds -> balanced fold sizes.
        fold_of_block = {
            blk: (i % self.n_splits) for i, blk in enumerate(unique_blocks)
        }
        fold_ids = np.array([fold_of_block[b] for b in block_ids])

        # Pre-compute block-centre coordinates for buffer distance checks.
        block_centres = {
            blk: xy[block_ids == blk].mean(axis=0) for blk in unique_blocks
        }
        # A training sample is buffered out if it lies within buffer_m of the
        # bounding envelope of any test block. We approximate the block envelope
        # by its centre +/- half the block size.
        half = 0.5 * self.block_size_m
        all_idx = np.arange(n)

        for fold in range(self.n_splits):
            test_mask = fold_ids == fold
            test_idx = all_idx[test_mask]
            if test_idx.size == 0:  # pragma: no cover - tiny inputs
                continue

            test_blocks = np.unique(block_ids[test_mask])
            # Distance from every candidate train sample to each test block edge.
            train_candidates = all_idx[~test_mask]
            keep = np.ones(train_candidates.size, dtype=bool)
            cand_xy = xy[train_candidates]
            for tb in test_blocks:
                cx, cy = block_centres[tb]
                dx = np.abs(cand_xy[:, 0] - cx) - half
                dy = np.abs(cand_xy[:, 1] - cy) - half
                # Chebyshev-style distance to the block rectangle (0 if inside).
                dist = np.maximum(np.maximum(dx, dy), 0.0)
                keep &= dist > self.buffer_m
            train_idx = train_candidates[keep]

            logger.debug(
                "fold %d/%d: %d test, %d train (%d buffered out)",
                fold + 1,
                self.n_splits,
                test_idx.size,
                train_idx.size,
                train_candidates.size - train_idx.size,
            )
            yield train_idx, test_idx
