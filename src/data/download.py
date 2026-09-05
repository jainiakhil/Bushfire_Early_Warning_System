"""Real-data ingestion wrappers: Copernicus (Sentinel-1) and NASA FIRMS.

These functions are the production ingestion path (spec Task 2.1). They are
**not** exercised by the test-suite or CI because they need credentials and
network access; the synthetic path in :mod:`src.data.synthetic` stands in there.

Credentials are read from function arguments first, then environment variables:

* ``CDSE_USERNAME`` / ``CDSE_PASSWORD`` - Copernicus Data Space Ecosystem
* ``FIRMS_MAP_KEY``                      - NASA FIRMS API map key

Both are free to obtain; see ``.env.example`` and the README.
"""

from __future__ import annotations

import io
import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import Config
from src.utils.logging import get_logger

logger = get_logger(__name__)

CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products({pid})/$value"
FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{bbox}/{days}/{start}"


class CredentialsError(RuntimeError):
    """Raised when a required API credential is missing."""


def _session(retries: int = 4, backoff: float = 1.5) -> requests.Session:
    """A requests session with retry/backoff on transient failures."""
    s = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


# ---------------------------------------------------------------------------
# Copernicus Data Space Ecosystem - Sentinel-1 GRD
# ---------------------------------------------------------------------------
def _cdse_token(username: str, password: str, session: requests.Session) -> str:
    """Exchange username/password for a short-lived CDSE access token."""
    resp = session.post(
        CDSE_TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": username,
            "password": password,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _odata_filter(aoi_wkt: str, start: datetime, end: datetime) -> str:
    """Build the OData ``$filter`` string for Sentinel-1 IW GRDH dual-pol scenes."""
    return (
        "Collection/Name eq 'SENTINEL-1' "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}') "
        f"and ContentDate/Start gt {start.isoformat()}Z "
        f"and ContentDate/Start lt {end.isoformat()}Z "
        "and Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' "
        "and a/Value eq 'IW_GRDH_1S')"
    )


def download_sentinel1_grd(
    aoi_geojson: dict[str, Any],
    date_range: tuple[str, str],
    output_dir: str | Path,
    *,
    cdse_user: str | None = None,
    cdse_pass: str | None = None,
    max_scenes: int = 2,
) -> list[Path]:
    """Download Sentinel-1 IW GRDH dual-pol (VV+VH) scenes over an AOI.

    Retrieves the pre-event (t0) and co-event (t1) scenes bracketing *date_range*
    - the nearest scene before the start and the nearest after, or simply the
    first ``max_scenes`` in ascending time.

    Args:
        aoi_geojson: A GeoJSON geometry (Polygon) in WGS84.
        date_range: ``(start_iso, end_iso)`` date strings, e.g.
            ``("2019-12-30", "2020-01-05")``.
        output_dir: Directory to extract the ``.SAFE`` products into.
        cdse_user: CDSE username (else ``CDSE_USERNAME`` env).
        cdse_pass: CDSE password (else ``CDSE_PASSWORD`` env).
        max_scenes: Cap on the number of products downloaded.

    Returns:
        Paths to the extracted ``.SAFE`` directories.

    Raises:
        CredentialsError: If CDSE credentials are not provided.
    """
    user = cdse_user or os.environ.get("CDSE_USERNAME")
    pw = cdse_pass or os.environ.get("CDSE_PASSWORD")
    if not user or not pw:
        raise CredentialsError(
            "CDSE credentials missing. Set CDSE_USERNAME / CDSE_PASSWORD or pass them "
            "explicitly. Register free at https://dataspace.copernicus.eu/"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session = _session()

    # AOI as WKT for the OData spatial predicate.
    from shapely.geometry import shape as shapely_shape

    aoi_wkt = shapely_shape(aoi_geojson).wkt
    start = datetime.fromisoformat(date_range[0])
    end = datetime.fromisoformat(date_range[1])

    logger.info("Querying CDSE OData for Sentinel-1 GRDH between %s and %s", start, end)
    params = {
        "$filter": _odata_filter(aoi_wkt, start - timedelta(days=14), end + timedelta(days=14)),
        "$orderby": "ContentDate/Start asc",
        "$top": str(max(max_scenes * 3, 10)),
    }
    resp = session.get(CDSE_ODATA_URL, params=params, timeout=120)
    resp.raise_for_status()
    products = resp.json().get("value", [])
    if not products:
        logger.warning("No Sentinel-1 products found for the given AOI/date range")
        return []

    token = _cdse_token(user, pw, session)
    headers = {"Authorization": f"Bearer {token}"}
    extracted: list[Path] = []
    for product in products[:max_scenes]:
        pid, name = product["Id"], product["Name"]
        logger.info("Downloading %s", name)
        dl = session.get(CDSE_DOWNLOAD_URL.format(pid=pid), headers=headers, timeout=1800)
        dl.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(dl.content)) as zf:
            zf.extractall(output_dir)
        safe_dir = output_dir / name
        extracted.append(safe_dir)
    logger.info("Downloaded %d Sentinel-1 products", len(extracted))
    return extracted


# ---------------------------------------------------------------------------
# NASA FIRMS - active fire hotspots
# ---------------------------------------------------------------------------
def download_nasa_firms(
    aoi_bbox: tuple[float, float, float, float],
    date_range: tuple[str, str],
    api_key: str | None = None,
    *,
    source: str = "VIIRS_SNPP_NRT",
    overpass: datetime | None = None,
    window_hours: float = 6.0,
) -> dict[str, Any]:
    """Download MODIS/VIIRS thermal hotspots as a GeoJSON ``FeatureCollection``.

    Args:
        aoi_bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in WGS84.
        date_range: ``(start_iso, end_iso)`` date strings (<= 10 days apart).
        api_key: FIRMS map key (else ``FIRMS_MAP_KEY`` env).
        source: FIRMS product, e.g. ``VIIRS_SNPP_NRT``, ``MODIS_NRT``.
        overpass: If given, keep only hotspots within +/- *window_hours* of this
            timestamp (spec: +/- 6 h of the satellite overpass).
        window_hours: Half-width of the temporal filter around *overpass*.

    Returns:
        GeoJSON ``FeatureCollection`` of Point features with FIRMS attributes.

    Raises:
        CredentialsError: If the FIRMS map key is not provided.
    """
    key = api_key or os.environ.get("FIRMS_MAP_KEY")
    if not key:
        raise CredentialsError(
            "FIRMS map key missing. Set FIRMS_MAP_KEY or pass api_key. "
            "Get one free at https://firms.modaps.eosdis.nasa.gov/api/map_key/"
        )

    start = datetime.fromisoformat(date_range[0])
    end = datetime.fromisoformat(date_range[1])
    days = max(1, min(10, (end - start).days + 1))
    bbox_str = ",".join(str(round(v, 4)) for v in aoi_bbox)
    url = FIRMS_AREA_URL.format(
        key=key, source=source, bbox=bbox_str, days=days, start=start.strftime("%Y-%m-%d")
    )

    logger.info("Querying NASA FIRMS %s over bbox %s for %d day(s)", source, bbox_str, days)
    resp = _session().get(url, timeout=120)
    resp.raise_for_status()

    import csv

    reader = csv.DictReader(io.StringIO(resp.text))
    feats: list[dict[str, Any]] = []
    for row in reader:
        try:
            lat, lon = float(row["latitude"]), float(row["longitude"])
        except (KeyError, ValueError):
            continue
        if overpass is not None:
            acq = _parse_firms_datetime(row)
            if acq and abs((acq - overpass).total_seconds()) > window_hours * 3600:
                continue
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": row,
            }
        )
    logger.info("FIRMS returned %d hotspots (after temporal filter)", len(feats))
    return {"type": "FeatureCollection", "features": feats}


def _parse_firms_datetime(row: dict[str, str]) -> datetime | None:
    """Parse ``acq_date`` + ``acq_time`` (HHMM) into a datetime, or ``None``."""
    try:
        d = datetime.strptime(row["acq_date"], "%Y-%m-%d")
        t = row.get("acq_time", "0000").zfill(4)
        return d.replace(hour=int(t[:2]), minute=int(t[2:]))
    except (KeyError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Source-agnostic resolver
# ---------------------------------------------------------------------------
def resolve_scenes(cfg: Config) -> dict[str, Any]:
    """Return the scene manifest for the configured data source.

    * ``cfg.data.source == "synthetic"`` -> generate (if missing) and load the
      synthetic manifest.
    * ``cfg.data.source == "cdse"`` -> load an existing ``manifest.json`` written
      by a prior real-data ingestion run (this function does not itself trigger
      large downloads; use the scripts for that).

    Returns:
        The parsed ``manifest.json`` dict.
    """
    import json

    manifest_path = cfg.data.raw_path / "manifest.json"
    if cfg.data.source == "synthetic":
        if not manifest_path.is_file():
            from src.data.synthetic import generate_dataset

            logger.info("No manifest found; generating synthetic dataset")
            generate_dataset(cfg)
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"data.source='cdse' but {manifest_path} is missing. Run a real-data ingestion "
            "first (see README > Using real data)."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))
