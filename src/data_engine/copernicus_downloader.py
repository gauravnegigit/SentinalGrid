import os
import shutil
import zipfile
import logging
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional

import requests
import rasterio
from dotenv import load_dotenv

# Load environment variables (.env file with Copernicus credentials)
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CopernicusDownloader:
    """
    Automated Data Downloader for Copernicus Data Space Ecosystem (CDSE).
    Queries and downloads paired pre/post-event Sentinel-1 GRD SAR imagery
    matching relative orbit numbers over any dynamic Bounding Box and Event Date.
    """

    def __init__(self, output_dir: str = "data/raw/sentinel1", timeout: int = 120):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.timeout = timeout

        # Retrieve credentials from environment
        self.username = os.getenv("COPERNICUS_USER")
        self.password = os.getenv("COPERNICUS_PASSWORD")

        self.auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        self.odata_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        # Downloads are served from a separate host. Using it directly avoids a redirect,
        # which makes `requests` drop the Authorization header (resulting in 401/403).
        self.download_base = "https://download.dataspace.copernicus.eu/odata/v1/Products"
        self.access_token: Optional[str] = None

    # ------------------------------------------------------------------ auth
    def get_auth_token(self) -> str:
        """Authenticates against Copernicus Keycloak and retrieves a Bearer Token."""
        if not self.username or not self.password:
            raise ValueError(
                "Missing credentials! Ensure COPERNICUS_USER and COPERNICUS_PASSWORD "
                "are set in your .env file."
            )

        data = {
            "client_id": "cdse-public",
            "username": self.username,
            "password": self.password,
            "grant_type": "password",
        }

        response = requests.post(self.auth_url, data=data, timeout=self.timeout)
        if response.status_code != 200:
            logger.error(f"Authentication failed: {response.text}")
            response.raise_for_status()

        self.access_token = response.json()["access_token"]
        logger.info("Successfully authenticated with Copernicus Data Space Ecosystem.")
        return self.access_token

    # ---------------------------------------------------------------- search
    @staticmethod
    def _bbox_to_polygon_wkt(bbox: List[float]) -> str:
        """Converts bbox [min_lon, min_lat, max_lon, max_lat] to a WKT Polygon string."""
        min_lon, min_lat, max_lon, max_lat = bbox
        return (
            f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, "
            f"{max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
        )

    def search_sentinel1_scenes(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        relative_orbit: Optional[int] = None,
        newest_first: bool = False,
    ) -> List[Dict]:
        """
        Queries the CDSE OData catalogue for Sentinel-1 IW GRDH products intersecting the bbox.
        Catalogue searches do not need a token, so none is sent here.
        """
        polygon_wkt = self._bbox_to_polygon_wkt(bbox)

        filters = [
            "Collection/Name eq 'SENTINEL-1'",                  # was: CollectionName (invalid property)
            "contains(Name,'IW_GRDH_1S')",                      # productType 'GRD' does not exist
            "not contains(Name,'COG')",                         # skip COG variants (different layout)
            f"ContentDate/Start ge {start_date}T00:00:00.000Z",
            f"ContentDate/Start le {end_date}T23:59:59.999Z",
            f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon_wkt}')",
        ]

        if relative_orbit is not None:
            filters.append(
                "Attributes/OData.CSC.IntegerAttribute/any("
                f"att:att/Name eq 'relativeOrbitNumber' and att/OData.CSC.IntegerAttribute/Value eq {relative_orbit})"
            )

        params = {
            "$filter": " and ".join(filters),
            "$orderby": f"ContentDate/Start {'desc' if newest_first else 'asc'}",
            "$expand": "Attributes",  # without this, product["Attributes"] is not returned
            "$top": 20,
        }

        # `params=` handles URL-encoding of spaces, quotes and the WKT string
        response = requests.get(self.odata_url, params=params, timeout=self.timeout)
        if not response.ok:
            raise RuntimeError(f"Catalogue query failed ({response.status_code}): {response.text[:500]}")

        products = response.json().get("value", [])
        logger.info(f"Query found {len(products)} Sentinel-1 scene(s) between {start_date} and {end_date}.")
        return products

    @staticmethod
    def _get_attribute(product: Dict, name: str):
        for att in product.get("Attributes", []):
            if att.get("Name") == name:
                return att.get("Value")
        return None

    # -------------------------------------------------------------- download
    def download_product(self, product_id: str, product_name: str) -> str:
        """Downloads a product ZIP by its OData Product ID (resumable-safe: .part then rename)."""
        output_zip_path = os.path.join(self.output_dir, f"{product_name}.zip")
        if os.path.exists(output_zip_path):
            logger.info(f"Product zip already exists locally: {output_zip_path}")
            return output_zip_path

        download_url = f"{self.download_base}({product_id})/$value"
        part_path = output_zip_path + ".part"

        logger.info(f"Downloading product {product_name}...")
        for attempt in range(2):
            # Token lifetime is only ~10 minutes, so refresh before every attempt
            token = self.get_auth_token()
            headers = {"Authorization": f"Bearer {token}"}
            with requests.get(download_url, headers=headers, stream=True, timeout=self.timeout) as response:
                if response.status_code == 401 and attempt == 0:
                    continue  # token problem: re-authenticate and retry once
                response.raise_for_status()
                with open(part_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            break

        os.replace(part_path, output_zip_path)  # only a complete file gets the final name
        logger.info(f"Downloaded successfully to {output_zip_path}")
        return output_zip_path

    # --------------------------------------------------------------- extract
    def extract_vv_geotiff(self, zip_path: str, target_tif_name: str) -> str:
        """
        Extracts the VV polarization raster from the downloaded .SAFE archive.
        Note: the output is raw GRD digital numbers in radar geometry (GCP-georeferenced),
        not calibrated backscatter. See notes in the README/answer.
        """
        output_tif_path = os.path.join(self.output_dir, target_tif_name)
        if os.path.exists(output_tif_path):
            logger.info(f"Extracted GeoTIFF already exists: {output_tif_path}")
            return output_tif_path

        temp_dir = os.path.join(self.output_dir, "temp")
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                vv_file = [
                    f for f in z.namelist()
                    if "/measurement/" in f.replace("\\", "/")
                    and "-vv-" in os.path.basename(f).lower()
                    and f.lower().endswith(".tiff")
                ]
                if not vv_file:
                    raise FileNotFoundError(f"Could not find VV polarization raster inside {zip_path}")
                extracted_path = z.extract(vv_file[0], path=temp_dir)

            with rasterio.open(extracted_path) as src:
                profile = src.profile.copy()
                profile.update(driver="GTiff", compress="deflate", tiled=True, blockxsize=512, blockysize=512)
                gcps, gcp_crs = src.gcps  # GRD tiffs are georeferenced by GCPs, not by an affine transform

                with rasterio.open(output_tif_path, "w", **profile) as dst:
                    # Copy block by block instead of loading the whole ~1 GB scene into memory
                    for _, window in dst.block_windows(1):
                        dst.write(src.read(1, window=window), 1, window=window)
                    if gcps:
                        dst.gcps = (gcps, gcp_crs or rasterio.crs.CRS.from_epsg(4326))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)  # remove the extracted temp files

        logger.info(f"Exported VV GeoTIFF to {output_tif_path}")
        return output_tif_path

    # ----------------------------------------------------------- orchestrate
    def fetch_paired_orbit_scenes(self, bbox: List[float], event_date: str) -> Tuple[str, str]:
        """
        1. Finds the earliest post-event Sentinel-1 scene on/after the event date.
        2. Reads its relative orbit number.
        3. Finds the closest earlier scene on the SAME relative orbit (up to 24 days before).
        4. Downloads both and returns local file paths (pre_sar_path, post_sar_path).
        """
        event_dt = datetime.strptime(event_date, "%Y-%m-%d")  # format: YYYY-MM-DD

        # 1. Post-event scene
        post_start = event_dt.strftime("%Y-%m-%d")
        post_end = (event_dt + timedelta(days=5)).strftime("%Y-%m-%d")
        post_scenes = self.search_sentinel1_scenes(bbox, start_date=post_start, end_date=post_end)
        if not post_scenes:
            raise RuntimeError(f"No Sentinel-1 post-event scene found for BBOX {bbox} near {event_date}")

        post_product = post_scenes[0]
        relative_orbit = self._get_attribute(post_product, "relativeOrbitNumber")
        if relative_orbit is None:
            raise RuntimeError("Post-event scene has no relativeOrbitNumber attribute; cannot pair orbits.")
        logger.info(f"Post-event scene: {post_product['Name']} (relative orbit {relative_orbit})")

        # 2. Pre-event scene on the same track, closest before the event first
        pre_start = (event_dt - timedelta(days=24)).strftime("%Y-%m-%d")
        pre_end = (event_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        pre_scenes = self.search_sentinel1_scenes(
            bbox, start_date=pre_start, end_date=pre_end,
            relative_orbit=relative_orbit, newest_first=True,
        )
        if not pre_scenes:
            raise RuntimeError(f"No matching pre-event scene found on relative orbit {relative_orbit}")
        pre_product = pre_scenes[0]
        logger.info(f"Pre-event scene: {pre_product['Name']}")

        # 3. Download both products
        post_zip = self.download_product(post_product["Id"], post_product["Name"])
        pre_zip = self.download_product(pre_product["Id"], pre_product["Name"])

        # 4. Extract VV GeoTIFFs
        post_sar_path = self.extract_vv_geotiff(post_zip, f"S1_POST_{event_date}_orbit{relative_orbit}_VV.tif")
        pre_sar_path = self.extract_vv_geotiff(pre_zip, f"S1_PRE_{event_date}_orbit{relative_orbit}_VV.tif")

        return pre_sar_path, post_sar_path


if __name__ == "__main__":
    # Trishuli River Corridor [min_lon, min_lat, max_lon, max_lat]
    trishuli_bbox = [85.1, 27.8, 85.4, 28.2]
    flood_event_date = "2026-08-26"

    downloader = CopernicusDownloader()
    try:
        pre_path, post_path = downloader.fetch_paired_orbit_scenes(bbox=trishuli_bbox, event_date=flood_event_date)
        print("\n--- Download & Extraction Complete ---")
        print(f"Pre-Event SAR Path: {pre_path}")
        print(f"Post-Event SAR Path: {post_path}")
    except Exception as e:
        logger.error(f"Execution failed: {e}")