# 🔥 Bushfire Early Warning System — Sentinel-1 SAR Active-Fire Detection

An end-to-end, production-grade machine-learning and remote-sensing pipeline that
detects **active fire fronts through cloud and smoke** using Sentinel-1
Synthetic Aperture Radar (SAR) imagery.

SAR sees through smoke and cloud that blind optical/thermal sensors, so a
radar-based detector complements MODIS/VIIRS thermal products during the worst of
a fire when they are most obscured.

The system ingests Sentinel-1 dual-polarisation GRD scenes, aligns them with NASA
FIRMS thermal hotspots as ground truth, performs radiometric calibration and
speckle filtering, engineers a 6-channel feature stack, trains a **ResNet-34
U-Net** with a hybrid **Focal + Dice** loss under **spatial block
cross-validation**, and serves the model as an asynchronous **FastAPI**
microservice returning **GeoJSON** fire perimeters, with a **Streamlit + Folium**
dashboard, all containerised with Docker Compose and covered by GitHub Actions CI.

> **Runs with zero credentials out of the box.** A synthetic Sentinel-1 scene
> generator stands in for real data so you can exercise the entire pipeline,
> tests and CI offline. Point it at real Copernicus + FIRMS data whenever you
> have API keys — see [Using real data](#using-real-data).

---

## Table of contents

- [Architecture](#architecture)
- [How it works (the science)](#how-it-works-the-science)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Configuration reference](#configuration-reference)
- [Using real data](#using-real-data)
- [API reference](#api-reference)
- [Dashboard](#dashboard)
- [Docker](#docker)
- [Project layout](#project-layout)
- [Development](#development)
- [Scalability](#scalability)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing & licence](#contributing--licence)

---

## Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    INGESTION                            │
  Copernicus CDSE ─▶│  download.py  (Sentinel-1 IW GRDH, VV+VH, t0 & t1)      │
      NASA FIRMS ──▶│  download.py  (VIIRS/MODIS hotspots, ±6 h of overpass)  │
   (or synthetic) ─▶│  synthetic.py (credential-free stand-in scenes)         │
                    └───────────────────────────┬─────────────────────────────┘
                                                ▼
       ┌────────────────────────────────────────────────────────────────────┐
       │  PREPROCESS (preprocess.py)                                         │
       │   • DN → σ⁰ dB      • Refined Lee speckle filter (5×5, edge-aware)  │
       │   • reproject to a local UTM zone (10 m)                            │
       └───────────────────────────┬────────────────────────────────────────┘
                                   ▼
       ┌────────────────────────────────────────────────────────────────────┐
       │  FEATURES (texture.py)  →  6-channel stack                          │
       │   σ⁰_VV, σ⁰_VH, Δσ⁰_VV, Δσ⁰_VH, VH/VV ratio, GLCM contrast (7×7)    │
       │  SELECTION (selection.py): mutual information + VIF                  │
       └───────────────────────────┬────────────────────────────────────────┘
                                   ▼
       ┌────────────────────────────────────────────────────────────────────┐
       │  DATASET (dataset.py) — windowed 256² tiles, 32-px overlap          │
       │   FIRMS points → rasterised binary masks;  flip / 90° rot aug       │
       │  SPATIAL CV (spatial_cv.py) — 25 km blocks, 5 km exclusion buffer   │
       └───────────────────────────┬────────────────────────────────────────┘
                                   ▼
       ┌────────────────────────────────────────────────────────────────────┐
       │  MODEL (unet.py) — ResNet-34 U-Net, 6-ch input, bottleneck dropout  │
       │  LOSS (loss.py) — Focal (α .25, γ 2) + Dice                         │
       │  TRAIN (train.py) — AMP, cosine LR, MLflow, early stop on val IoU   │
       │  EXPORT — TorchScript + model card                                  │
       └───────────────────────────┬────────────────────────────────────────┘
                                   ▼
       ┌───────────────────────────────────┐   ┌────────────────────────────┐
       │  FastAPI  (app/)                   │   │  Streamlit + Folium        │
       │   GET  /health                     │◀──│  (app/dashboard/app.py)    │
       │   POST /api/v1/detect  → GeoJSON   │   │   upload → map + charts    │
       │   POST /api/v1/monitor/drift (KS)  │   └────────────────────────────┘
       └───────────────────────────────────┘
```

## How it works (the science)

| Stage | What & why |
|---|---|
| **Calibration** | Raw intensity → backscatter decibels: σ⁰(dB) = 10·log₁₀(DN² + ε). Puts VV/VH on a physical, additive scale. |
| **Speckle filter** | SAR is grainy (multiplicative speckle). The **Refined Lee** filter shrinks each pixel toward a local mean by a weight derived from the local vs. speckle coefficient of variation, estimated from the *edge-aligned* sub-window so fire fronts stay sharp. |
| **Temporal differencing** | Δσ⁰ = co-event − pre-event. An active fire collapses canopy structure and dries fuel, dropping VH (volume scattering) by several dB — a strong, cloud-independent signal. |
| **Polarisation ratio** | VH/VV separates rough bare ground from vegetated/volume scatterers. |
| **GLCM contrast** | Grey-Level Co-occurrence Matrix texture over a 7×7 window on VH captures the turbulent, broken texture of a fire perimeter. |
| **Spatial block CV** | Neighbouring pixels are correlated, so random k-fold leaks. We assign whole 25 km blocks to folds and drop a 5 km buffer around test blocks — an honest estimate of generalisation to unseen fire regions. |
| **Focal + Dice loss** | Fire pixels are <1% of the scene. Focal loss down-weights the easy background; Dice optimises region overlap directly. |

## Prerequisites

- **Python 3.13** (3.11+ works; the project is developed and CI-tested on 3.13).
- **~4 GB disk** for the virtualenv (PyTorch + GDAL stack).
- **NVIDIA GPU + driver** for training (optional — CPU works, just slower).
  A 6 GB card trains the default config; see [Troubleshooting](#troubleshooting)
  for smaller cards.
- **Docker Desktop** (optional) for the containerised deployment.
- **GDAL** is bundled in the `rasterio` / `pyproj` wheels — no system install
  needed for local dev on Windows/macOS. Linux Docker images install it from apt.

## Installation

### 1. Create the virtual environment (once)

**Windows (PowerShell):**

```powershell
cd "path\to\Bushfire EWS"
py -3.13 -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS / Linux:**

```bash
cd path/to/Bushfire-EWS
python3.13 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install PyTorch (choose one)

| Target | Command |
|---|---|
| **NVIDIA GPU (CUDA 12.4+ driver)** | `pip install "torch>=2.2" "torchvision>=0.17" --index-url https://download.pytorch.org/whl/cu124` |
| **CPU only** | `pip install -r requirements-cpu.txt` (skip step 3) |
| Other CUDA versions | pick the matching index from <https://pytorch.org/get-started/locally/> |

### 3. Install the rest

```bash
pip install -r requirements.txt
pip install -e .          # makes `import src...` / `import app...` work everywhere
```

Verify:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
pytest -q
```

## Quickstart

The full loop on **synthetic data** (no credentials, ~2 min on a GPU):

**PowerShell:**

```powershell
venv\Scripts\python.exe scripts\generate_synthetic_data.py --n-scenes 8
venv\Scripts\python.exe scripts\run_preprocess.py
venv\Scripts\python.exe scripts\run_training.py --fold 0 --epochs 40
venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

**bash:**

```bash
python scripts/generate_synthetic_data.py --n-scenes 8
python scripts/run_preprocess.py
python scripts/run_training.py --fold 0 --epochs 40
uvicorn app.main:app --reload
```

Then, in another shell, start the dashboard and open <http://localhost:8501>:

```bash
streamlit run app/dashboard/app.py
```

Test the API directly (a feature GeoTIFF written by `run_preprocess.py`):

```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -F "file=@data/processed/scene_000_features.tif"
```

Interactive API docs: <http://localhost:8000/docs>. MLflow run history:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

### Quick smoke run

```bash
SAR__TRAINING__EPOCHS=1 SAR__SYNTHETIC__N_SCENES=3 SAR__SYNTHETIC__IMAGE_SIZE=256 \
  python scripts/run_training.py
```

## Configuration reference

All behaviour is driven by [`configs/config.yaml`](configs/config.yaml), parsed
into a typed model by `src/config.py`. **Override any key** with an environment
variable `SAR__<SECTION>__<KEY>` (double underscores), e.g.
`SAR__TRAINING__BATCH_SIZE=4`.

<details>
<summary>Key fields</summary>

| Key | Meaning | Default |
|---|---|---|
| `data.source` | `synthetic` or `cdse` | `synthetic` |
| `data.spatial_resolution` | metres / pixel | `10.0` |
| `data.tile_size` / `tile_overlap` | patch geometry (px) | `256` / `32` |
| `data.crs_working` | `auto-utm`, `EPSG:3857`, or a fixed CRS | `auto-utm` |
| `data.crs_output` | CRS of emitted GeoJSON | `EPSG:4326` |
| `features.selected_bands` | ordered model input channels | 6 bands |
| `features.glcm_window` / `glcm_levels` | GLCM texture params | `7` / `32` |
| `features.refined_lee_window` | speckle filter window (px) | `5` |
| `features.equivalent_looks` | ENL for speckle variance | `4.0` |
| `training.batch_size` / `grad_accum_steps` | effective batch = product | `8` / `2` |
| `training.learning_rate` / `epochs` | AdamW LR / max epochs | `3e-4` / `40` |
| `training.focal_alpha` / `focal_gamma` | focal loss params | `0.25` / `2.0` |
| `training.spatial_block_size_km` / `spatial_buffer_km` | spatial CV geometry | `25` / `5` |
| `training.n_folds` | spatial CV folds | `5` |
| `training.early_stopping_patience` | epochs w/o val-IoU gain | `8` |
| `training.device` | `auto` / `cuda` / `cpu` | `auto` |
| `training.precision` | `16-mixed` (AMP) / `32` | `16-mixed` |
| `model.encoder` / `encoder_weights` | timm encoder / `imagenet`\|`null` | `resnet34` / `imagenet` |
| `model.bottleneck_dropout` | spatial dropout p | `0.2` |
| `mlflow.tracking_uri` | SQLite path or tracking-server URL | `sqlite:///mlruns/mlflow.db` |
| `api.confidence_threshold` | prob → binary fire mask | `0.65` |
| `api.simplify_tolerance` | Douglas-Peucker tol (deg) | `0.0001` |
| `api.min_polygon_area_ha` | drop smaller detections | `0.5` |
| `api.drift_alpha` | KS-test p-value alert threshold | `0.01` |

</details>

## Using real data

1. **Get free credentials**

   - Copernicus Data Space Ecosystem — <https://dataspace.copernicus.eu/>
   - NASA FIRMS map key — <https://firms.modaps.eosdis.nasa.gov/api/map_key/>

2. **Configure**

   ```bash
   cp .env.example .env      # fill in CDSE_USERNAME / CDSE_PASSWORD / FIRMS_MAP_KEY
   ```

   Set `data.source: cdse` in `configs/config.yaml` (or `SAR__DATA__SOURCE=cdse`).

3. **Ingest** — use `src.data.download.download_sentinel1_grd` /
   `download_nasa_firms` for a chosen AOI + date range (a known fire, e.g. the
   2019–20 SE-Australia bushfires), write a `data/raw/manifest.json` in the same
   shape the synthetic generator produces, then run the same
   `run_preprocess.py` → `run_training.py` pipeline unchanged.

## API reference

Base URL `http://localhost:8000`. OpenAPI/Swagger at `/docs`.

### `GET /health`

```json
{
  "status": "ok",
  "gpu_available": true,
  "device": "cuda",
  "model_loaded": true,
  "model_version": "trace",
  "encoder": "resnet34",
  "feature_bands": ["sigma0_vv", "sigma0_vh", "delta_vv", "delta_vh", "pol_ratio", "glcm_contrast"]
}
```

### `POST /api/v1/detect`

`multipart/form-data` with `file` = a GeoTIFF of ≥ 6 bands in
`features.selected_bands` order. Runs TorchScript inference (tiled + stitched),
vectorises, returns:

```json
{
  "detections": {
    "type": "FeatureCollection",
    "features": [
      { "type": "Feature",
        "geometry": { "type": "Polygon", "coordinates": [[[150.1, -35.2], ...]] },
        "properties": { "mean_confidence": 0.82, "area_ha": 43.5,
                        "risk_level": "high", "detected_at": "2026-09-06T..." } }
    ],
    "properties": { "n_features": 3, "total_area_ha": 121.7, "threshold": 0.65 }
  },
  "n_features": 3,
  "total_area_ha": 121.7,
  "threshold": 0.65,
  "inference_ms": 812.4,
  "source_filename": "scene.tif"
}
```

### `POST /api/v1/monitor/drift`

Body `{ "incoming": { "<band>": [values...] }, "alpha": 0.01 }`. Runs a
two-sample Kolmogorov–Smirnov test per band against a stored baseline (build it
with `app.monitoring.drift.build_baseline`). Returns per-band statistic, p-value
and a `drift_detected` flag (`p < alpha` for any band).

## Dashboard

`streamlit run app/dashboard/app.py` (set `API_URL` if the API is not on
`localhost:8000`). Upload a 6-band SAR feature GeoTIFF in the sidebar and press
**Detect fire fronts** to get:

- an interactive Folium map with fire-perimeter polygons coloured by risk level;
- per-band backscatter histograms of the upload;
- predicted burnt area in hectares (1 px = 100 m² = 0.01 ha) and inference time.

## Docker

```bash
# from the repo root — needs an exported model at data/outputs/model.torchscript
docker compose -f docker/docker-compose.yml up --build -d
```

- API → <http://localhost:8000> (`/docs`)
- Dashboard → <http://localhost:8501>

The compose file bind-mounts `configs/` and `data/outputs/` so the containers
pick up your config and trained model. Images are CPU-only; for GPU serving,
re-base `docker/Dockerfile.api` on an `nvidia/cuda` image.

## Project layout

```
.
├── configs/config.yaml        # single source of truth (typed by src/config.py)
├── src/
│   ├── config.py              # nested pydantic config + env overrides
│   ├── data/                  # synthetic, download (CDSE/FIRMS), preprocess, dataset
│   ├── features/              # texture (6-ch stack + GLCM), selection (MI + VIF)
│   ├── models/                # loss, metrics, unet, train, infer
│   └── utils/                 # logging, runtime (device/seed), geo, spatial_cv
├── app/
│   ├── main.py                # FastAPI factory + lifespan model load
│   ├── api/                   # endpoints, schemas
│   ├── monitoring/drift.py    # KS-test drift detection
│   └── dashboard/app.py       # Streamlit + Folium
├── scripts/                   # generate_synthetic_data, run_preprocess, run_training, export_model
├── tests/                     # 32 tests, synthetic fixtures, FastAPI TestClient
├── docker/                    # Dockerfile.api, Dockerfile.dashboard, docker-compose.yml
├── .github/workflows/ci.yml   # lint + types + tests + image builds
└── docs/BUILD_LOG.md          # how this repo was built
```

## Development

```bash
ruff check .                              # lint + import sort + docstrings
mypy --ignore-missing-imports src/ app/   # static types
pytest -q                                 # 32 tests, ~1 min, CPU-only
pytest --cov=src --cov=app                # with coverage
```

CI (`.github/workflows/ci.yml`) runs all three on every push/PR to `main`, plus
builds both Docker images. The test-suite uses only synthetic data — no network,
no GPU, no credentials.

**Extending the model** (all config-driven, no code changes needed):

- *Add a feature channel*: implement it in `src/features/texture.py::build_feature_stack`,
  add its name to `_KNOWN_BANDS`, then add it to `features.selected_bands` and bump
  `model.in_channels`.
- *Swap the encoder*: set `model.encoder` to any `timm`/`smp` encoder
  (`resnet50`, `efficientnet-b3`, …).
- *Change tiling / CV geometry*: `data.tile_size`, `training.spatial_block_size_km`, …

## Scalability

- **Windowed raster IO** everywhere — gigabyte scenes are never fully loaded.
- **Tiled inference** with cosine-taper overlap stitching handles arbitrarily
  large scenes at bounded memory.
- **Config-driven** feature set, encoder and geometry — experiment without edits.
- **Source-agnostic ingestion** (`synthetic` ↔ `cdse`) behind one interface.
- **Stateless API** — scale horizontally behind a load balancer; the model is
  loaded once per worker at startup.
- **MLflow** tracking scales from local SQLite to a shared tracking server /
  Postgres by changing one URI.
- **Docker Compose** → a straightforward Kubernetes port (Deployment + Service
  per component).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | Lower `training.batch_size` (e.g. 4) and/or raise `grad_accum_steps`; keep `precision: 16-mixed`. |
| `torch` installed but `CUDA: False` | The CPU wheel got installed. Reinstall torch from the `cu124` index *before* `requirements.txt`. |
| `rasterio` / GDAL import errors on Linux | `sudo apt-get install libgdal-dev gdal-bin` (already handled in Docker/CI). |
| MLflow `filesystem tracking backend … maintenance mode` | Use the default `sqlite:///…` URI, or set `MLFLOW_ALLOW_FILE_STORE=true`. |
| `Model not found: …/model.torchscript` | Train + export first: `python scripts/run_training.py`. |
| GLCM feature step is slow | Expected (pixel-wise loop); it is cached per scene in `data/processed/*_features.tif`. |
| Dashboard shows "API unreachable" | Start `uvicorn app.main:app`; set `API_URL` if not on `localhost:8000`. |

## Roadmap

- Vectorised / numba GLCM contrast.
- Multi-fold ensembling + calibrated probabilities.
- Real-data validation notebook (2019–20 AU bushfires).
- Time-series inference over Sentinel-1 revisit stacks.
- Kubernetes manifests / Helm chart.

## Contributing & licence

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Licensed under the **MIT License** —
see [`LICENSE`](LICENSE). Contributions welcome; the goal is an open, reusable
reference implementation for SAR-based wildfire detection.

*This project uses Copernicus Sentinel data and NASA FIRMS data, subject to their
respective terms of use.*
