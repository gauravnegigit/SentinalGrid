import numpy as np
import rasterio
from rasterio.enums import Resampling

class SARProcessor:
    def __init__(self, pre_sar_path: str, post_sar_path: str):
        self.pre_path = pre_sar_path
        self.post_path = post_sar_path

    def compute_backscatter_delta(self, output_path: str) -> np.ndarray:
        """
        Computes log-ratio backscatter difference (VV/VH bands) between co-registered 
        pre- and post-event Sentinel-1 GRD rasters on identical orbit tracks.
        """
        with rasterio.open(self.pre_path) as pre_src, rasterio.open(self.post_path) as post_src:
            # Read VV polarization (Band 1) and convert to dB
            pre_vv = pre_src.read(1).astype(np.float32)
            post_vv = post_src.read(1, out_shape=pre_vv.shape, resampling=Resampling.bilinear).astype(np.float32)

            # Avoid division by zero
            pre_vv = np.where(pre_vv <= 0, 1e-6, pre_vv)
            post_vv = np.where(post_vv <= 0, 1e-6, post_vv)

            # Calculate dB Backscatter Difference Delta = 10 * log10(Post / Pre)
            delta_db = 10 * np.log10(post_vv / pre_vv)

            # Save Co-registered Delta Raster
            profile = pre_src.profile
            profile.update(dtype=rasterio.float32, count=1)

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(delta_db.astype(rasterio.float32), 1)

            print(f"Co-registered SAR Delta exported successfully to {output_path}")
            return delta_db

if __name__ == "__main__":
    processor = SARProcessor(
        pre_sar_path="data/raw/sentinel1/S1_20260814_orbit121.tif",
        post_sar_path="data/raw/sentinel1/S1_20260826_orbit121.tif"
    )
    processor.compute_backscatter_delta("data/processed/co_registered/sar_delta.tif")