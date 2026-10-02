from dotenv import load_dotenv
import os

from src.data_engine.osm_extractor import OSMExtractor 
load_dotenv()

trishuli_bbox = [85.1, 27.8, 85.4, 28.2]
extractor = OSMExtractor(api_key=os.getenv("OHSOME_API_KEY"))

roads_gdf = extractor.fetch_preevent_roads(trishuli_bbox)
roads_gdf.to_file("data/raw/osm/preevent_roads.geojson", driver="GeoJSON")
print(f"Extracted {len(roads_gdf)} pre-event road segments.")