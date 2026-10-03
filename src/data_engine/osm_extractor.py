"""Pre-event OpenStreetMap snapshot via the ohsome API v2.

Pulls buildings, roads, bridges, settlements and health facilities *as they were on a given
date before the event*. Edits made after the event are never fetched: the snapshot date is
forced to be strictly earlier than the event date, and clamped to the latest timestamp the
ohsome instance has (27 July 2026 at the time of the challenge).

Attribution: (c) OpenStreetMap contributors, ODbL. A published, modified version of this
data must stay under ODbL.

Example
-------
    export OHSOME_API_KEY=...
    python -m src.data_engine.osm_extractor --bbox 85.10 27.85 85.45 28.30 \
        --event-date 2026-08-26 --out data/raw/osm
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Optional, Sequence
load_dotenv()

import requests

log = logging.getLogger(__name__)

OHSOME = "https://api.heigit.org/ohsome-api/v2-rc"

# ohsome filter expressions (https://docs.ohsome.org/ohsome-api/v2/reference/filter.html).
# v2 renamed ``geometry:other`` to ``geometry:collection``; none of the filters below use it.
LAYER_FILTERS: dict[str, str] = {
    "buildings": "building=* and geometry:polygon",
    "roads": "highway=* and geometry:line",
    "bridges": "(bridge=* and not bridge=no or man_made=bridge) and geometry:line",
    "settlements": "place in (city,town,village,hamlet,suburb,isolated_dwelling,neighbourhood) "
                   "and geometry:point",
    "health": "(amenity in (hospital,clinic,doctors) or healthcare in (hospital,clinic)) "
              "and (geometry:point or geometry:polygon)",
}

# Highway classes that can carry vehicles (used by the cut-off analysis); footways etc. are
# kept in the snapshot but flagged so that downstream code can choose.
DRIVABLE_HIGHWAYS = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
    "service", "track", "living_street", "road",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
}
NON_EXISTENT_HIGHWAYS = {"proposed", "construction", "abandoned", "razed", "planned"}

# Columns ohsome v2 adds next to the tags; everything else in a layer is an OSM tag.
_CORE_COLUMNS = ["osm_type", "osm_id", "edit_timestamp", "valid_to_timestamp", "version",
                 "minor_version", "edits", "user_id", "user_name", "changeset_id",
                 "tags", "bbox", "geom_type", "geom", "clipped"]
# Tags worth keeping in the GeoPackage export (all tags stay in the raw GeoParquet files).
_EXPORT_TAGS = [
    "name", "name:en", "name:ne", "building", "amenity", "healthcare", "place", "population",
    "highway", "surface", "smoothness", "bridge", "man_made", "layer", "tunnel", "ford",
    "oneway", "access", "bicycle", "foot", "motor_vehicle", "maxspeed", "lanes", "width",
    "ref", "waterway", "natural", "height", "building:levels", "roof:material",
]


@dataclass
class Snapshot:
    layer: str
    timestamp: str
    n_features: int
    path: Path


# --------------------------------------------------------------------------- HTTP
def _headers() -> dict[str, str]:
    key = os.environ.get("OHSOME_API_KEY")
    if not key:
        raise RuntimeError(
            "OHSOME_API_KEY is not set. ohsome API v2 needs a (free) API key: "
            "sign up at https://account.heigit.org/signup and export OHSOME_API_KEY."
        )
    return {"Authorization": key}


def _post(path: str, body: dict, retries: int = 5, timeout: int = 900) -> requests.Response:
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = requests.post(f"{OHSOME}/{path}", json=body, headers=_headers(), timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (401, 403):
                raise RuntimeError(
                    f"ohsome rejected the API key ({r.status_code}): {r.text[:200]}")
            # 429/5xx are worth retrying, other 4xx are request errors
            if r.status_code < 500 and r.status_code != 429:
                raise RuntimeError(f"ohsome error {r.status_code}: {r.text[:300]}")
            last = RuntimeError(f"ohsome {r.status_code}: {r.text[:200]}")
        except requests.RequestException as exc:
            last = exc
        log.warning("ohsome request failed (%s); retry %d/%d", last, attempt + 1, retries)
        time.sleep(min(60, 3 * 2 ** attempt))
    raise RuntimeError(f"ohsome request failed after {retries} attempts: {last}")


def _find_key(obj: Any, names: Sequence[str]) -> Optional[str]:
    """Depth-first search of a JSON document for the first string value under one of ``names``."""
    wanted = {n.lower() for n in names}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in wanted and isinstance(v, str):
                return v
        for v in obj.values():
            hit = _find_key(v, names)
            if hit:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = _find_key(v, names)
            if hit:
                return hit
    return None


def latest_available_timestamp() -> Optional[datetime]:
    """Latest OSM timestamp the ohsome instance holds (None if it cannot be determined).

    The v2 metadata schema is still a release candidate, so the timestamp is located by key
    name rather than by a fixed path.
    """
    try:
        r = requests.get(f"{OHSOME}/metadata", headers=_headers(), timeout=60)
        r.raise_for_status()
        ts = _find_key(r.json(), ["toTimestamp", "to_timestamp", "to", "end", "latest"])
        return datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
    except Exception as exc:  # network/schema problems must not block the run
        log.warning("Could not read ohsome metadata: %s", exc)
        return None


def resolve_snapshot_date(event_date: date, requested: Optional[date] = None) -> date:
    """Pick the snapshot day: never on/after the event, never later than the data available."""
    snap = requested or (event_date - timedelta(days=1))
    if snap >= event_date:
        raise ValueError(
            f"Snapshot date {snap} is not before the event date {event_date}. "
            "Post-event OSM edits are not allowed as inputs."
        )
    latest = latest_available_timestamp()
    if latest is not None and snap > latest.date():
        log.info("Clamping snapshot %s to latest available ohsome date %s", snap, latest.date())
        snap = latest.date()
    return snap


def tile_bbox(bbox: Sequence[float], max_deg: float = 0.25) -> list[tuple]:
    """Split a bbox into <=max_deg cells so each ohsome request stays small."""
    w, s, e, n = bbox
    nx = max(1, math.ceil((e - w) / max_deg))
    ny = max(1, math.ceil((n - s) / max_deg))
    dx, dy = (e - w) / nx, (n - s) / ny
    return [(w + i * dx, s + j * dy, w + (i + 1) * dx, s + (j + 1) * dy)
            for i in range(nx) for j in range(ny)]


# --------------------------------------------------------------------------- parsing
def _tags_to_dict(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return dict(value)  # list of (key, value) pairs from a Parquet MAP


def parquet_to_gdf(content: bytes):
    """Decode an ohsome v2 GeoParquet response into a GeoDataFrame (EPSG:4326).

    Tags (a Parquet MAP) are flattened into one column per tag key and geometries (WKB in
    the ``geom`` column) are converted to shapely objects.
    """
    import geopandas as gpd
    import pandas as pd
    import pyarrow.parquet as pq
    import shapely

    table = pq.read_table(io.BytesIO(content))
    try:
        df = table.to_pandas(maps_as_pydicts="strict")
    except TypeError:  # older pyarrow: maps come back as lists of tuples
        df = table.to_pandas()
    if df.empty:
        return gpd.GeoDataFrame({"osm_id": []}, geometry=[], crs="EPSG:4326")

    geom = df["geom"]
    if geom.map(lambda g: isinstance(g, (bytes, bytearray, memoryview))).any():
        geoms = shapely.from_wkb(geom.map(bytes).to_numpy())
    else:  # already decoded by the reader
        geoms = geom.to_numpy()

    tags = pd.DataFrame([_tags_to_dict(t) for t in df["tags"]], index=df.index)
    meta = df.drop(columns=[c for c in ("tags", "geom", "bbox") if c in df.columns])
    # a tag key that collides with a metadata column keeps its metadata meaning
    tags = tags.drop(columns=[c for c in tags.columns if c in meta.columns])
    return gpd.GeoDataFrame(pd.concat([meta, tags], axis=1), geometry=geoms, crs="EPSG:4326")


# --------------------------------------------------------------------------- extraction
def fetch_layer(bbox: Sequence[float], snapshot: date, layer: str):
    """Return a GeoDataFrame for one layer at the snapshot date (00:00 UTC)."""
    import geopandas as gpd
    import pandas as pd

    if layer not in LAYER_FILTERS:
        raise KeyError(f"Unknown layer '{layer}'. Choose from {sorted(LAYER_FILTERS)}")
    frames = []
    for cell in tile_bbox(bbox):
        body = {
            "aoi": [round(v, 6) for v in cell],
            "filter": LAYER_FILTERS[layer],
            "time": f"{snapshot.isoformat()}T00:00:00Z",   # single timestamp = snapshot
            "clip": False,   # keep whole objects; duplicates across tiles are dropped below
        }
        frames.append(parquet_to_gdf(_post("extraction/features.parquet", body).content))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return gpd.GeoDataFrame({"osm_id": []}, geometry=[], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry",
                           crs="EPSG:4326")
    return gdf.drop_duplicates(subset=["osm_type", "osm_id"]).reset_index(drop=True)


def extract_pre_event_osm(
    bbox: Sequence[float],
    event_date: date,
    out_dir: str | Path = "data/raw/osm",
    snapshot_date: Optional[date] = None,
    layers: Optional[Sequence[str]] = None,
) -> dict[str, Snapshot]:
    """Download all layers as GeoParquet files plus a ``snapshot.json`` provenance file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = resolve_snapshot_date(event_date, snapshot_date)
    result: dict[str, Snapshot] = {}
    for layer in layers or LAYER_FILTERS:
        log.info("Fetching OSM layer '%s' @ %s", layer, snap)
        gdf = fetch_layer(bbox, snap, layer)
        path = out_dir / f"{layer}_{snap.isoformat()}.parquet"
        gdf.to_parquet(path)
        result[layer] = Snapshot(layer, snap.isoformat(), len(gdf), path)
    (out_dir / "snapshot.json").write_text(json.dumps({
        "event_date": event_date.isoformat(),
        "snapshot_date": snap.isoformat(),
        "bbox": list(bbox),
        "api": OHSOME,
        "layers": {k: {"features": v.n_features, "file": v.path.name} for k, v in result.items()},
        "attribution": "© OpenStreetMap contributors",
    }, indent=2))
    return result


# --------------------------------------------------------------------------- GeoDataFrames
def load_layer(path: str | Path):
    """Read a saved layer into a GeoDataFrame (EPSG:4326) with one column per OSM tag."""
    import geopandas as gpd

    return gpd.read_parquet(path)


def prepare_roads(gdf):
    """Add ``drivable`` flag and drop features that were not built at the snapshot."""
    if gdf.empty or "highway" not in gdf.columns:
        gdf = gdf.copy()
        gdf["drivable"] = False
        return gdf
    hw = gdf["highway"].astype(str)
    out = gdf[~hw.isin(NON_EXISTENT_HIGHWAYS)].copy()
    out["drivable"] = out["highway"].astype(str).isin(DRIVABLE_HIGHWAYS)
    return out


def export_geopackage(snapshots: dict[str, Snapshot],
                      out_path: str | Path = "data/processed/vector_layers/osm_pre_event.gpkg"):
    """Combine saved layers into one GeoPackage (layer per file) for the analysis stages.

    Only core columns and a curated set of tags are written, so the file stays small and
    GeoPackage-safe; the full tag set remains in the raw GeoParquet files.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    keep_core = ["osm_type", "osm_id", "edit_timestamp", "version"]
    for name, snap in snapshots.items():
        gdf = load_layer(snap.path)
        if name == "roads":
            gdf = prepare_roads(gdf)
        if gdf.empty:
            continue
        cols = [c for c in keep_core + _EXPORT_TAGS + ["drivable"] if c in gdf.columns]
        gdf[cols + ["geometry"]].to_file(out_path, layer=name, driver="GPKG")
    return out_path


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N"))
    ap.add_argument("--event-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--snapshot-date", help="YYYY-MM-DD, must be before the event")
    ap.add_argument("--out", default="data/raw/osm")
    ap.add_argument("--gpkg", help="also write a combined GeoPackage at this path")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    snaps = extract_pre_event_osm(
        tuple(args.bbox), date.fromisoformat(args.event_date), args.out,
        date.fromisoformat(args.snapshot_date) if args.snapshot_date else None,
    )
    for s in snaps.values():
        print(f"{s.layer:12s} {s.n_features:8d} features  {s.path}")
    if args.gpkg:
        print("GeoPackage:", export_geopackage(snaps, args.gpkg))


if __name__ == "__main__":
    _cli()