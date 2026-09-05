"""PyTorch dataset for tiled SAR feature stacks with rasterised FIRMS targets.

:class:`SARDataset` (spec Task 4.2):

* reads multi-band feature rasters with **windowed** IO so gigabyte scenes never
  hit RAM in full;
* crops :math:`256\\times256` patches with a 32-pixel overlap;
* rasterises FIRMS hotspot points into a binary target mask
  (1 = active fire front, 0 = background);
* applies geometry-only augmentation (flips, 90 degrees rotations) - never
  radiometric jitter, which would corrupt the physical dB values;
* standardises each channel with dataset statistics.

:func:`build_dataloaders` partitions the tiles into train/val with
:class:`~src.utils.spatial_cv.SpatialBlockKFold` on tile-centre coordinates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.features import rasterize
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset

from src.config import Config
from src.utils.geo import iter_tile_windows
from src.utils.logging import get_logger
from src.utils.spatial_cv import SpatialBlockKFold

logger = get_logger(__name__)

_NORM_STATS_FILENAME = "norm_stats.json"


@dataclass(frozen=True)
class TileRef:
    """A single training tile: which scene, which window, and its map centre."""

    scene_id: str
    feature_path: str
    firms_path: str
    window: tuple[int, int, int, int]  # col_off, row_off, width, height
    centre_xy: tuple[float, float]     # projected coords (metres) of tile centre


def _load_feature_scene_index(processed_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the per-scene feature entries, tolerating either manifest shape."""
    return processed_manifest["scenes"]


def enumerate_tiles(
    feature_manifest: dict[str, Any], cfg: Config
) -> list[TileRef]:
    """Build the full list of tiles across all feature scenes.

    Args:
        feature_manifest: Manifest with ``scenes: [{id, feature_path, firms, ...}]``.
        cfg: Pipeline config (tile size / overlap).

    Returns:
        A flat list of :class:`TileRef`, skipping tiles that are entirely nodata.
    """
    tiles: list[TileRef] = []
    for scene in feature_manifest["scenes"]:
        fpath = scene["feature_path"]
        with rasterio.open(fpath) as ds:
            transform = ds.transform
            windows = iter_tile_windows(ds.width, ds.height, cfg.data.tile_size, cfg.data.tile_overlap)
            for w in windows:
                # Cheap nodata check on a downsampled read of band 1.
                sample = ds.read(1, window=w, out_shape=(1, 16, 16), boundless=True, fill_value=np.nan)
                if not np.isfinite(sample).any():
                    continue
                cx, cy = transform * (w.col_off + w.width / 2, w.row_off + w.height / 2)
                tiles.append(
                    TileRef(
                        scene_id=scene["id"],
                        feature_path=fpath,
                        firms_path=scene["firms"],
                        window=(int(w.col_off), int(w.row_off), int(w.width), int(w.height)),
                        centre_xy=(float(cx), float(cy)),
                    )
                )
    logger.info("Enumerated %d tiles across %d scenes", len(tiles), len(feature_manifest["scenes"]))
    return tiles


def compute_norm_stats(
    tiles: list[TileRef], n_bands: int, cfg: Config, *, max_tiles: int = 200
) -> dict[str, list[float]]:
    """Estimate per-channel mean/std over a sample of tiles and cache to disk.

    Args:
        tiles: Tile references to sample from.
        n_bands: Number of feature channels.
        cfg: Pipeline config (for the cache location).
        max_tiles: Cap on tiles read for the estimate.

    Returns:
        ``{"mean": [...], "std": [...]}`` with one entry per channel.
    """
    cache = cfg.data.processed_path / _NORM_STATS_FILENAME
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))

    rng = np.random.default_rng(cfg.training.seed)
    sample = rng.permutation(len(tiles))[:max_tiles]
    acc_sum = np.zeros(n_bands, dtype=np.float64)
    acc_sq = np.zeros(n_bands, dtype=np.float64)
    acc_n = np.zeros(n_bands, dtype=np.float64)

    for i in sample:
        t = tiles[i]
        col, row, w, h = t.window
        with rasterio.open(t.feature_path) as ds:
            arr = ds.read(window=Window(col, row, w, h), boundless=True, fill_value=np.nan)
        for b in range(n_bands):
            v = arr[b][np.isfinite(arr[b])]
            acc_sum[b] += v.sum()
            acc_sq[b] += np.square(v).sum()
            acc_n[b] += v.size

    mean = acc_sum / np.maximum(acc_n, 1)
    var = np.maximum(acc_sq / np.maximum(acc_n, 1) - mean**2, 1e-6)
    stats = {"mean": mean.tolist(), "std": np.sqrt(var).tolist()}
    cache.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Computed norm stats over %d tiles -> %s", len(sample), cache)
    return stats


class SARDataset(Dataset):
    """Windowed multi-band SAR tiles with rasterised FIRMS fire masks."""

    def __init__(
        self,
        tiles: list[TileRef],
        cfg: Config,
        *,
        norm_stats: dict[str, list[float]],
        augment: bool = False,
        target_buffer_px: int = 2,
    ) -> None:
        """Initialise.

        Args:
            tiles: Tile references (typically one fold's train or val split).
            cfg: Pipeline config.
            norm_stats: ``{"mean": [...], "std": [...]}`` per channel.
            augment: Enable flips / 90 degrees rotations (train only).
            target_buffer_px: Dilation radius (pixels) applied to each hotspot
                point so fire fronts are a few pixels wide, not single pixels.
        """
        self.tiles = tiles
        self.cfg = cfg
        self.n_bands = len(cfg.features.selected_bands)
        self.mean = np.asarray(norm_stats["mean"], dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.asarray(norm_stats["std"], dtype=np.float32).reshape(-1, 1, 1)
        self.augment = augment
        self.target_buffer_px = target_buffer_px
        self._firms_cache: dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.tiles)

    def _load_hotspots(self, path: str) -> list[dict[str, Any]]:
        """Load and memoise a FIRMS GeoJSON feature list."""
        if path not in self._firms_cache:
            self._firms_cache[path] = json.loads(Path(path).read_text(encoding="utf-8"))["features"]
        return self._firms_cache[path]

    def _target_mask(
        self, firms_path: str, window: Window, transform: Any, src_crs: Any, shape: tuple[int, int]
    ) -> np.ndarray:
        """Rasterise hotspots that fall inside *window* into a binary mask."""
        from rasterio.warp import transform_geom

        feats = self._load_hotspots(firms_path)
        win_transform = rasterio.windows.transform(window, transform)
        shapes = []
        for f in feats:
            geom = f["geometry"]
            # FIRMS points are lon/lat; project into the raster CRS.
            geom_proj = transform_geom("EPSG:4326", src_crs, geom)
            shapes.append((geom_proj, 1))
        if not shapes:
            return np.zeros(shape, dtype=np.float32)
        mask = rasterize(
            shapes,
            out_shape=shape,
            transform=win_transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        )
        if self.target_buffer_px > 0 and mask.any():
            from scipy.ndimage import binary_dilation

            mask = binary_dilation(mask, iterations=self.target_buffer_px).astype(np.uint8)
        return mask.astype(np.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.tiles[idx]
        col, row, w, h = t.window
        window = Window(col, row, w, h)

        with rasterio.open(t.feature_path) as ds:
            arr = ds.read(
                window=window, boundless=True, fill_value=np.nan, out_dtype="float32"
            )  # (C, H, W)
            transform, src_crs = ds.transform, ds.crs

        # Replace nodata with the channel mean (post-standardisation -> ~0).
        nan_mask = ~np.isfinite(arr)
        if nan_mask.any():
            for b in range(arr.shape[0]):
                arr[b][nan_mask[b]] = self.mean[b, 0, 0]

        arr = (arr - self.mean) / self.std
        target = self._target_mask(t.firms_path, window, transform, src_crs, arr.shape[1:])

        if self.augment:
            arr, target = _augment(arr, target)

        x = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        y = torch.from_numpy(np.ascontiguousarray(target[None], dtype=np.float32))
        return x, y


def _augment(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same random flip + k*90 degrees rotation to image and mask."""
    rng = np.random
    if rng.random() < 0.5:
        x, y = x[:, ::-1, :], y[::-1, :]
    if rng.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    k = rng.randint(0, 4)
    if k:
        x = np.rot90(x, k=k, axes=(1, 2))
        y = np.rot90(y, k=k, axes=(0, 1))
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def build_dataloaders(
    cfg: Config,
    feature_manifest: dict[str, Any],
    *,
    fold: int = 0,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """Construct train/val dataloaders for one spatial-CV fold.

    Args:
        cfg: Pipeline config.
        feature_manifest: Manifest with feature-stack paths per scene.
        fold: Which fold (0-based) of ``cfg.training.n_folds`` to use as val.

    Returns:
        ``(train_loader, val_loader, info)`` where *info* has tile counts and
        the norm stats actually used.
    """
    tiles = enumerate_tiles(feature_manifest, cfg)
    if not tiles:
        raise RuntimeError("No tiles enumerated - check feature manifest / rasters")

    stats = compute_norm_stats(tiles, len(cfg.features.selected_bands), cfg)

    xy = np.array([t.centre_xy for t in tiles], dtype=np.float64)
    splitter = SpatialBlockKFold(
        n_splits=cfg.training.n_folds,
        block_size_km=cfg.training.spatial_block_size_km,
        buffer_km=cfg.training.spatial_buffer_km,
        random_state=cfg.training.seed,
    )
    splits = list(splitter.split(xy))
    fold = min(fold, len(splits) - 1)
    train_idx, val_idx = splits[fold]
    if train_idx.size == 0:  # tiny dataset fallback: ignore the buffer
        logger.warning("Fold %d has no training tiles after buffering; using a random split", fold)
        perm = np.random.default_rng(cfg.training.seed).permutation(len(tiles))
        cut = max(1, int(0.8 * len(tiles)))
        train_idx, val_idx = perm[:cut], perm[cut:]

    train_tiles = [tiles[i] for i in train_idx]
    val_tiles = [tiles[i] for i in val_idx] or [tiles[i] for i in train_idx[:1]]

    train_ds = SARDataset(train_tiles, cfg, norm_stats=stats, augment=True)
    val_ds = SARDataset(val_tiles, cfg, norm_stats=stats, augment=False)

    pin = cfg.training.device != "cpu"
    common: dict[str, Any] = {
        "batch_size": cfg.training.batch_size,
        "num_workers": cfg.training.num_workers,
        "pin_memory": pin,
    }
    train_loader = DataLoader(
        train_ds, shuffle=True, drop_last=len(train_ds) > cfg.training.batch_size, **common
    )
    val_loader = DataLoader(val_ds, shuffle=False, **common)

    info = {
        "n_tiles": len(tiles),
        "n_train": len(train_tiles),
        "n_val": len(val_tiles),
        "fold": fold,
        "norm_stats": stats,
    }
    logger.info("Fold %d: %d train tiles, %d val tiles", fold, len(train_tiles), len(val_tiles))
    return train_loader, val_loader, info
