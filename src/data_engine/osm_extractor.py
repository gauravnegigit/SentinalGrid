import os
import json
from io import BytesIO

import geopandas as gpd
import requests
from dotenv import load_dotenv
load_dotenv()


class OSMExtractor:
    """Extracts pre-event OSM features via the ohsome API v2 (API key required)."""

    def __init__(
        self,
        api_url: str = "https://api.heigit.org/ohsome-api/v2-rc",
        api_key: str | None = None,
        timeout: int = 600,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key or os.getenv("OHSOME_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key. Get a free one at https://account.heigit.org/signup "
                "and set it as the OHSOME_API_KEY environment variable."
            )
        self.timeout = timeout

    @staticmethod
    def _validate_bbox(bbox: list) -> list:
        """bbox format: [min_lon, min_lat, max_lon, max_lat]"""
        if len(bbox) != 4:
            raise ValueError("bbox must have 4 values: [min_lon, min_lat, max_lon, max_lat]")
        min_lon, min_lat, max_lon, max_lat = map(float, bbox)
        if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
            raise ValueError(f"Invalid bbox: {bbox}")
        return [min_lon, min_lat, max_lon, max_lat]

    def _fetch_features(self, bbox: list, osm_filter: str, snapshot_date: str) -> gpd.GeoDataFrame:
        response = requests.post(
            f"{self.api_url}/extraction/features.parquet",
            json={
                "aoi": self._validate_bbox(bbox),
                "filter": osm_filter,
                "time": snapshot_date,  # single snapshot, e.g. "2026-07-27"
                "clip": False,
            },
            headers={"Authorization": self.api_key},
            timeout=self.timeout,
        )
        if response.status_code in (401, 403):
            raise RuntimeError(
                f"ohsome API auth error {response.status_code}. Check that OHSOME_API_KEY is "
                f"valid and that you are using the v2 URL: {self.api_url}"
            )
        if not response.ok:
            raise RuntimeError(f"ohsome API error {response.status_code}: {response.text[:500]}")

        gdf = gpd.read_parquet(
            BytesIO(response.content),
            to_pandas_kwargs={"maps_as_pydicts": "strict"},
        )
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        return gdf

    def fetch_preevent_buildings(self, bbox: list, snapshot_date: str = "2026-07-27") -> gpd.GeoDataFrame:
        """Fetches pre-event building footprints."""
        return self._fetch_features(
            bbox, "building in (yes, house, residential) and geometry:polygon", snapshot_date
        )

    def fetch_preevent_roads(self, bbox: list, snapshot_date: str = "2026-07-27") -> gpd.GeoDataFrame:
        """Fetches pre-event road networks and bridges."""
        return self._fetch_features(
            bbox, "type:way and (highway=* or bridge=yes)", snapshot_date
        )


def _save(gdf: gpd.GeoDataFrame, path: str, label: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if gdf.empty:
        print(f"No {label} found; skipping {path}")
        return
    # GeoJSON can't store dict/list columns (e.g. the tags map), so serialise them
    out = gdf.copy()
    for col in out.columns:
        if col != out.geometry.name and out[col].map(lambda v: isinstance(v, (dict, list))).any():
            out[col] = out[col].map(lambda v: json.dumps(v, default=str) if v is not None else None)
    out.to_file(path, driver="GeoJSON")
    print(f"Extracted {len(gdf)} pre-event {label}.")


if __name__ == "__main__":
    # Example Trishuli Corridor Bounding Box
    trishuli_bbox = [85.1, 27.8, 85.4, 28.2]
    extractor = OSMExtractor()

    roads_gdf = extractor.fetch_preevent_roads(trishuli_bbox)
    _save(roads_gdf, "data/raw/osm/preevent_roads.geojson", "road/bridge segments")

    buildings_gdf = extractor.fetch_preevent_buildings(trishuli_bbox)
    _save(buildings_gdf, "data/raw/osm/preevent_buildings.geojson", "buildings")


