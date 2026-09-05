# Build log

An append-only, timestamped record of how this repository was built: decisions,
deviations from the original spec, commands run and verification results. Newest
entries at the bottom.

---

## 2026-09-06 — Initial implementation (v0.1.0)

### Environment

| Item | Value |
|---|---|
| OS | Windows 11 |
| Python | 3.13.15 (`py -3.13`), venv at repo root `venv/` — the only interpreter used |
| GPU | NVIDIA RTX 2060 Max-Q, 6 GB VRAM, driver 592.82 |
| PyTorch | 2.6.0+cu124 (installed from `https://download.pytorch.org/whl/cu124`) |
| Key libs | rasterio 1.5.1, geopandas 1.1.4, segmentation-models-pytorch 0.5.0, mlflow 3.16, fastapi 0.141, streamlit 1.63 |

### Decisions & deviations from `Bushfire_EWS_Plan.docx`

1. **`segmentation-models-pytorch`** added (spec listed no segmentation lib). Used
   for the ResNet-34 U-Net; `smp` adapts the first conv to 6 channels automatically.
   Also added `torchmetrics`, `pytest-cov`, `tqdm`, `pydantic-settings`.
2. **Project flattened to the repo root** instead of a nested `sar_bushfire_pipeline/`
   directory, so the GitHub repo *is* the project.
3. **`device: "auto"`** (was `"cuda"`) with `src/utils/runtime.resolve_device` fallback.
   Code runs on CPU (CI) and CUDA (local) unchanged.
4. **Synthetic data generator** `src/data/synthetic.py` — new module. Produces
   physically plausible VV/VH GeoTIFF scene pairs + FIRMS-style hotspot GeoJSON so
   the whole pipeline, the test-suite and CI run with **zero credentials**. The real
   CDSE + FIRMS download wrappers (`src/data/download.py`) are kept as the production
   path, exercised only when credentials are supplied.
5. **CRS strategy**: work in a local UTM zone for all metric operations
   (features, tiling, area); reproject to EPSG:4326 at the vectorisation boundary
   so the Douglas-Peucker tolerance is in degrees as the spec's `0.0001` implies.
   `EPSG:3857` is retained as a `data.crs_working` option.
6. **MLflow backend** = local SQLite (`sqlite:///mlruns/mlflow.db`); the file store
   is deprecated in MLflow 3. Relative sqlite paths are anchored to the repo root.
7. **Batch size default 8** (+ `grad_accum_steps: 2` → effective 16) and
   `precision: 16-mixed` for the 6 GB GPU. Spec's 16 is preserved as the effective
   batch.
8. **Python 3.13 everywhere** (dev, Docker base image, CI) instead of the spec's 3.11.
9. **TorchScript export** via `torch.jit.trace` with an eager-vs-traced parity check
   and a `torch.jit.script` fallback; a `.modelcard.json` sidecar records encoder,
   norm stats, threshold, metrics and git SHA.
10. **CI** installs CPU torch wheels (`requirements-cpu.txt`) and runs with
    `SAR__MODEL__ENCODER_WEIGHTS=null` to avoid network flakiness; coverage gate 70%.

### Modules implemented

- `src/config.py` — nested pydantic config + `SAR__SECTION__KEY` env overrides.
- `src/utils/` — `logging`, `runtime` (device/seed), `geo` (CRS, tiling,
  `raster_to_geojson`), `spatial_cv` (`SpatialBlockKFold` with exclusion buffer).
- `src/data/` — `synthetic`, `download` (CDSE OData + FIRMS area API + source-agnostic
  `resolve_scenes`), `preprocess` (`to_db`, vectorised Refined Lee, reproject),
  `dataset` (`SARDataset` windowed IO + rasterised FIRMS masks + geometry-only aug).
- `src/features/` — `texture` (6-channel stack incl. sliding-window GLCM contrast),
  `selection` (mutual information + VIF, greedy pruning).
- `src/models/` — `loss` (Focal, Dice, hybrid), `metrics` (IoU, PR-AUC),
  `unet` (`SegModel` wrapper with bottleneck spatial dropout + TorchScript export),
  `train` (MLflow, cosine LR, AMP, grad accumulation, early stopping), `infer`
  (tiled prediction with cosine-taper stitching + vectorisation).
- `app/` — FastAPI factory + lifespan model load, `/health`, `/api/v1/detect`,
  `/api/v1/monitor/drift`; `app/monitoring/drift.py` (KS test + baseline builder);
  `app/dashboard/app.py` (Streamlit + Folium).
- `scripts/` — `generate_synthetic_data`, `run_preprocess`, `run_training`,
  `export_model`.
- `tests/` — 32 tests across preprocess, features, spatial CV, loss, geo, inference
  (incl. FastAPI `TestClient` for all three endpoints).
- `docker/` — multi-stage `Dockerfile.api` / `Dockerfile.dashboard`, `docker-compose.yml`.
- `.github/workflows/ci.yml` — lint + type-check + tests + Docker image builds.

### Verification (all on the synthetic dataset, no credentials)

```
generate_synthetic_data.py            OK  (3 scenes, EPSG:326xx/327xx)
run_preprocess.py                     OK  (calibration + Refined Lee + 6-band feature stacks)
run_training.py  (CPU, 1 epoch)       OK  -> checkpoint + model.torchscript
run_training.py  (CUDA, 1 epoch, AMP) OK  -> RTX 2060, GradScaler path exercised
FastAPI TestClient /health            200
FastAPI TestClient /api/v1/detect     200  -> valid GeoJSON FeatureCollection
FastAPI TestClient /monitor/drift     drift=False on baseline, drift=True on +15 dB shift
ruff check .                          All checks passed
mypy src/ app/                        Success: no issues found in 29 source files
pytest tests/                         32 passed, coverage 91%
```

### Known limitations / follow-ups

- GLCM contrast is a pixel-by-pixel loop (~9 s per 256² tile); fine when cached
  per scene, but the dominant cost for very large real scenes. Candidate for a
  vectorised / numba implementation.
- `download.py` CDSE/FIRMS wrappers are written to the documented APIs but have not
  been run against the live services in this build (no credentials). First real
  run should validate the OData `$filter` and FIRMS CSV schema.
- Real-data end-to-end (`data.source: cdse`) is deferred until credentials are
  provided.

---

## 2026-09-06 — CI fixes (post first push)

First GitHub Actions run surfaced two failures, both fixed:

1. **mypy** `src/models/train.py`: the per-epoch `row` dict was inferred as
   `dict[str, float | Tensor]` on a clean (non-incremental) check because
   `scheduler.get_last_lr()[0]` is loosely typed. Fixed by declaring
   `row: dict[str, float]` and wrapping each value in `float(...)`. Local runs had
   passed only due to a stale `.mypy_cache`.
2. **Docker build** failed on `apt-get install gdal-bin libgdal32` — the runtime
   GDAL package name differs across the Debian release behind `python:3.13-slim`.
   Fixed by **removing all system GDAL** from both Dockerfiles: GDAL/PROJ/GEOS are
   bundled inside the `rasterio` / `pyogrio` / `pyproj` / `shapely` manylinux
   wheels, so no apt package is needed. Also added `PYTHONPATH=/app` to the images
   (Streamlit puts the script dir, not the workdir, on `sys.path`), pinned
   `torch==2.6.0+cpu` / `torchvision==0.21.0+cpu` in `requirements-cpu.txt` so the
   CUDA wheel is never pulled on Linux, and dropped the now-unnecessary
   `apt-get install libgdal-dev` step from the CI test job.
