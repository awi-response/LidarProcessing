import os
import time
import json
import glob
import pdal
import laspy
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull, QhullError
from shapely.geometry import mapping
from tqdm import tqdm
from datetime import timedelta

def extract_footprint_batch(input_folder, output_folder):
    """Extracts footprints (convex hulls) from LAS/LAZ files and saves them in a specified folder."""
    
    os.makedirs(output_folder, exist_ok=True)
    
    print("\nStarting footprint extraction...")
    start = time.time()

    laz_files = glob.glob(os.path.join(input_folder, "*.laz")) + glob.glob(os.path.join(input_folder, "*.las"))

    if not laz_files:
        print("No LAS/LAZ files found in the input directory. Exiting.")
        return

    for laz_file in tqdm(laz_files, desc="Processing footprints", unit="file"):
        try:
            with laspy.open(laz_file) as file:
                point_cloud = file.read()
                x, y = point_cloud.x, point_cloud.y
                las_crs = file.header.parse_crs()
                crs = las_crs.to_epsg() if las_crs and las_crs.to_epsg() else "EPSG:4326"

            unique_points = np.unique(np.vstack((x, y)).T, axis=0)
            try:
                hull = ConvexHull(unique_points)
                footprint = Polygon(unique_points[hull.vertices])
            except QhullError:
                print(f"Warning: Convex Hull computation failed for {laz_file}. Creating an empty polygon.")
                footprint = Polygon()

            gdf = gpd.GeoDataFrame({'geometry': [footprint]}, crs=crs)

            output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(laz_file))[0] + ".gpkg")
            gdf.to_file(output_path, driver="GPKG")

        except Exception as e:
            print(f"Error processing {laz_file}: {e}")

    print(f"Footprint extraction completed in {timedelta(seconds=int(time.time() - start))}.")