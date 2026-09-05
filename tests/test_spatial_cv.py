"""Tests for the spatial block cross-validation splitter."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.spatial_cv import SpatialBlockKFold


@pytest.fixture
def grid_points() -> np.ndarray:
    """A 40x40 regular grid of points spaced 2 km apart (metres)."""
    xs = np.arange(40) * 2_000.0
    ys = np.arange(40) * 2_000.0
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def test_train_test_are_disjoint(grid_points):
    splitter = SpatialBlockKFold(n_splits=4, block_size_km=10, buffer_km=2, random_state=0)
    for train_idx, test_idx in splitter.split(grid_points):
        assert set(train_idx).isdisjoint(set(test_idx))


def test_every_sample_tested_exactly_once(grid_points):
    splitter = SpatialBlockKFold(n_splits=5, block_size_km=10, buffer_km=0, random_state=1)
    seen = np.zeros(len(grid_points), dtype=int)
    for _, test_idx in splitter.split(grid_points):
        seen[test_idx] += 1
    assert (seen == 1).all()


def test_buffer_excludes_points_near_test_blocks(grid_points):
    no_buffer = SpatialBlockKFold(n_splits=4, block_size_km=10, buffer_km=0, random_state=2)
    with_buffer = SpatialBlockKFold(n_splits=4, block_size_km=10, buffer_km=4, random_state=2)
    n_train_nobuf = sum(len(tr) for tr, _ in no_buffer.split(grid_points))
    n_train_buf = sum(len(tr) for tr, _ in with_buffer.split(grid_points))
    assert n_train_buf < n_train_nobuf


def test_deterministic_under_seed(grid_points):
    a = list(SpatialBlockKFold(n_splits=3, random_state=7).split(grid_points))
    b = list(SpatialBlockKFold(n_splits=3, random_state=7).split(grid_points))
    for (tr_a, te_a), (tr_b, te_b) in zip(a, b, strict=False):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(te_a, te_b)


def test_rejects_bad_input_shape():
    splitter = SpatialBlockKFold(n_splits=3)
    with pytest.raises(ValueError):
        list(splitter.split(np.zeros((10, 3))))
