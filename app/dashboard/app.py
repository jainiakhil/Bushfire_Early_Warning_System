"""Streamlit + Folium dashboard for the SAR bushfire detection service.

Workflow (spec Module 7):

1. Upload a 6-band SAR feature GeoTIFF in the sidebar.
2. The file is POSTed to the FastAPI ``/api/v1/detect`` endpoint.
3. The returned GeoJSON fire perimeters are drawn on an interactive Folium map.
4. Backscatter histograms and the predicted burnt area (hectares) are shown.

Configure the API location with the ``API_URL`` environment variable
(default ``http://localhost:8000``).

Run:
    streamlit run app/dashboard/app.py
"""

from __future__ import annotations

import io
import os

import folium
import httpx
import numpy as np
import rasterio
import streamlit as st
from folium.plugins import Fullscreen
from streamlit_folium import st_folium

API_URL = os.environ.get("API_URL", "http://localhost:8000")
DETECT_ENDPOINT = f"{API_URL}/api/v1/detect"
HEALTH_ENDPOINT = f"{API_URL}/health"

# 1 pixel at 10 m = 100 m^2 = 0.01 ha
HA_PER_PIXEL = 0.01

_RISK_COLOURS = {
    "extreme": "#7f0000",
    "high": "#d7301f",
    "moderate": "#fc8d59",
    "low": "#fdcc8a",
}

st.set_page_config(page_title="SAR Bushfire Detection", page_icon="🔥", layout="wide")


def _check_api() -> dict | None:
    """Return the /health payload, or None if the API is unreachable."""
    try:
        r = httpx.get(HEALTH_ENDPOINT, timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _post_detect(name: str, data: bytes) -> dict:
    """POST the uploaded GeoTIFF to the detection endpoint and return JSON."""
    files = {"file": (name, data, "image/tiff")}
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(DETECT_ENDPOINT, files=files)
    resp.raise_for_status()
    return resp.json()


def _feature_collection_bounds(fc: dict) -> list[list[float]] | None:
    """Compute ``[[min_lat, min_lon], [max_lat, max_lon]]`` for a FeatureCollection."""
    xs: list[float] = []
    ys: list[float] = []

    def _walk(coords) -> None:
        if isinstance(coords[0], (int, float)):
            xs.append(coords[0])
            ys.append(coords[1])
        else:
            for c in coords:
                _walk(c)

    for feat in fc.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("coordinates"):
            _walk(geom["coordinates"])
    if not xs:
        return None
    return [[min(ys), min(xs)], [max(ys), max(xs)]]


def _style_fn(feature: dict) -> dict:
    risk = feature.get("properties", {}).get("risk_level", "moderate")
    colour = _RISK_COLOURS.get(risk, "#fc8d59")
    return {"fillColor": colour, "color": colour, "weight": 2, "fillOpacity": 0.45}


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("🔥 SAR Bushfire Detection")
st.sidebar.caption(f"API: `{API_URL}`")

health = _check_api()
if health is None:
    st.sidebar.error("API unreachable. Start it with:\n\n`uvicorn app.main:app`")
elif not health.get("model_loaded"):
    st.sidebar.warning("API is up but no model is loaded (train + export first).")
else:
    st.sidebar.success(
        f"API healthy · device: {health.get('device')} · "
        f"encoder: {health.get('encoder', '?')}"
    )

uploaded = st.sidebar.file_uploader(
    "6-band SAR feature GeoTIFF", type=["tif", "tiff"], accept_multiple_files=False
)
run = st.sidebar.button("Detect fire fronts", type="primary", disabled=uploaded is None)

# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------
st.title("Active fire front detection")

if uploaded is None:
    st.info("Upload a 6-band SAR feature GeoTIFF in the sidebar to begin.")
    st.stop()

file_bytes = uploaded.getvalue()

# Local raster preview / histograms.
with rasterio.open(io.BytesIO(file_bytes)) as ds:
    band_names = list(ds.descriptions) or [f"band {i}" for i in range(1, ds.count + 1)]
    preview = ds.read(out_shape=(ds.count, min(ds.height, 512), min(ds.width, 512))).astype("float32")

col_map, col_stats = st.columns([3, 2])

if run:
    try:
        with st.spinner("Running inference on the API…"):
            result = _post_detect(uploaded.name, file_bytes)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Detection failed: {exc}")
        st.stop()

    fc = result["detections"]
    n_feat = result["n_features"]
    area_ha = result["total_area_ha"]

    with col_stats:
        st.metric("Fire polygons", n_feat)
        st.metric("Predicted burnt area", f"{area_ha:,.1f} ha")
        st.metric("Inference time", f"{result['inference_ms']:.0f} ms")
        st.caption(f"Confidence threshold: {result['threshold']:.2f}")

        st.subheader("Backscatter distributions")
        for i, name in enumerate(band_names[: preview.shape[0]]):
            vals = preview[i][np.isfinite(preview[i])].ravel()
            if vals.size:
                counts, edges = np.histogram(vals, bins=40)
                st.caption(name)
                st.bar_chart(
                    {"count": counts},
                    x=None,
                    height=120,
                )

    with col_map:
        bounds = _feature_collection_bounds(fc)
        centre = (
            [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
            if bounds
            else [0.0, 0.0]
        )
        fmap = folium.Map(location=centre, zoom_start=11, tiles="OpenStreetMap")
        Fullscreen().add_to(fmap)
        if fc.get("features"):
            folium.GeoJson(
                fc,
                name="Fire fronts",
                style_function=_style_fn,
                tooltip=folium.GeoJsonTooltip(
                    fields=["risk_level", "mean_confidence", "area_ha"],
                    aliases=["Risk", "Confidence", "Area (ha)"],
                ),
            ).add_to(fmap)
            if bounds:
                fmap.fit_bounds(bounds)
        else:
            st.warning("No fire fronts detected above the confidence threshold.")
        st_folium(fmap, width=None, height=560)

    with st.expander("Raw GeoJSON response"):
        st.json(fc)
else:
    with col_map:
        st.info("Press **Detect fire fronts** to run the model.")
    with col_stats:
        st.caption(f"Uploaded: {uploaded.name} · {preview.shape[0]} bands")
