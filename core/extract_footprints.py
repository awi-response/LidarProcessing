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

# --- new imports for robust CRS handling ---
from pyproj import CRS
from pyproj.exceptions import CRSError
try:                                    # laspy ≥ 2.5
    from laspy.vlrs.known import GeoKeyDirectoryVLR
except ImportError:
    try:                                # laspy 2.0 – 2.4 (lower-case “r”)
        from laspy.vlrs.known import GeoKeyDirectoryVlr as GeoKeyDirectoryVLR
    except ImportError:                 # very old laspy (≤ 1.x)
        GeoKeyDirectoryVLR = None


# --- helpers copied/condensed from your robust logic ---

def _decode_geotiff_key_list(key_list):
    """
    Convert a raw GeoKeyDirectory array (list of uint16) to {key_id: value}.
    Works for LAS written by very old laspy/libLAS that stored only the
    key directory and inline values (tiffTagLocation == 0).
    """
    if len(key_list) < 4:
        return {}
    num_keys = key_list[3]
    out = {}
    for i in range(num_keys):
        base = 4 + i * 4
        key_id, tag_loc, count, value = key_list[base: base + 4]
        if tag_loc == 0 and count == 1:   # value stored inline
            out[key_id] = value
    return out


def _safe_parse_crs(header):
    """
    Return a pyproj.CRS or None.
      1) Try laspy's header.parse_crs() (covers WKT + clean GeoTIFF).
      2) If that fails, manually walk GeoTIFF VLRs to recover EPSG.
    """
    try:
        crs = header.parse_crs()
        if crs:
            return crs
    except CRSError:
        pass

    # Manual scan of GeoTIFF keys in VLRs
    for vlr in header.vlrs:
        if (GeoKeyDirectoryVLR and isinstance(vlr, GeoKeyDirectoryVLR)) or (
            getattr(vlr, "user_id", "") == "LASF_Projection" and getattr(vlr, "record_id", None) == 34735
        ):
            raw = getattr(vlr, "geo_keys", None)
            if isinstance(raw, dict):            # laspy ≥ 2.3
                key_dict = raw
            elif isinstance(raw, list):          # laspy ≤ 2.2
                key_dict = _decode_geotiff_key_list(raw)
            else:                                 # last-chance: raw directory
                key_dict = _decode_geotiff_key_list(getattr(vlr, "key_directory", []))

            epsg = key_dict.get(3072) or key_dict.get(2048)  # PCS or GCS
            if epsg:
                try:
                    return CRS.from_epsg(epsg)
                except CRSError:
                    pass

    return None


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

                # --- robust CRS parse (replaces direct header.parse_crs()) ---
                try:
                    las_crs = file.header.parse_crs()
                    if not las_crs:
                        las_crs = _safe_parse_crs(file.header)
                except CRSError:
                    las_crs = _safe_parse_crs(file.header)

                epsg = (las_crs.to_epsg() if las_crs else None) or 4326
                crs = CRS.from_epsg(epsg)  # pass a real CRS to GeoDataFrame

            unique_points = np.unique(np.vstack((x, y)).T, axis=0)
            try:
                hull = ConvexHull(unique_points)
                footprint = Polygon(unique_points[hull.vertices])
            except QhullError:
                print(f"Warning: Convex Hull computation failed for {laz_file}. Creating an empty polygon.")
                footprint = Polygon()

            gdf = gpd.GeoDataFrame({'geometry': [footprint]}, crs=crs)

            output_path = os.path.join(
                output_folder, os.path.splitext(os.path.basename(laz_file))[0] + ".gpkg"
            )
            gdf.to_file(output_path, driver="GPKG")

        except Exception as e:
            print(f"Error processing {laz_file}: {e}")

    print(f"Footprint extraction completed in {timedelta(seconds=int(time.time() - start))}.")
