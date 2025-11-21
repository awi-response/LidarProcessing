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

# --- robust CRS handling imports ---
from pyproj import CRS
from pyproj.exceptions import CRSError

try:                                    # laspy ≥ 2.5
    from laspy.vlrs.known import GeoKeyDirectoryVLR
except ImportError:
    try:                                # laspy 2.0 – 2.4 (lower-case “r”)
        from laspy.vlrs.known import GeoKeyDirectoryVlr as GeoKeyDirectoryVLR
    except ImportError:                 # very old laspy (≤ 1.x)
        GeoKeyDirectoryVLR = None


# ----------------------------------------------------------------------
# CRS helper functions – shared with preprocessing
# ----------------------------------------------------------------------

def _decode_geotiff_key_list(key_list):
    """
    Convert GeoTIFF key info to {key_id: value}.

    Handles:
      - old style: flat list of uint16 (key_directory array)
      - new style: list of GeoKeyEntryStruct objects
    """
    if not key_list:
        return {}

    first = key_list[0]

    # New style: list of GeoKeyEntryStruct
    if hasattr(first, "key_id"):
        out = {}
        for entry in key_list:
            if getattr(entry, "tiff_tag_location", None) == 0 and getattr(entry, "count", None) == 1:
                out[entry.key_id] = entry.value_offset
        return out

    # Old style: flat uint16 array
    if len(key_list) < 4:
        return {}

    num_keys = int(key_list[3])
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
      3) Extra heuristic: detect truncated WGS84 GEOGCS WKT and fall back to EPSG:4326.
    """
    # 1) Try laspy's built-in parsing (handles WKT VLRs & GeoTIFF)
    try:
        crs = header.parse_crs()
        if crs:
            return crs
    except CRSError as e:
        msg = str(e)

        # Heuristic: truncated WGS84 GEOGCS → treat as EPSG:4326
        # e.g. "Invalid WKT string: GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID[... AUTHORITY["EPSG","432"
        if 'GEOGCS["WGS 84"' in msg or 'DATUM["WGS_1984"' in msg:
            try:
                return CRS.from_epsg(4326)
            except CRSError:
                # fall through to GeoTIFF scan if this somehow fails
                pass
    except Exception:
        # Any other unexpected parse error: fall through to GeoTIFF scan
        pass

    # 2) Manual scan of GeoTIFF keys in VLRs to recover EPSG
    for vlr in getattr(header, "vlrs", []):
        is_geotiff_vlr = (
            (GeoKeyDirectoryVLR and isinstance(vlr, GeoKeyDirectoryVLR)) or
            (
                getattr(vlr, "user_id", "") == "LASF_Projection"
                and getattr(vlr, "record_id", None) == 34735
            )
        )
        if not is_geotiff_vlr:
            continue

        raw = getattr(vlr, "geo_keys", None)
        if isinstance(raw, dict):                  # laspy ≥ 2.3
            key_dict = raw
        elif isinstance(raw, list):                # list of GeoKeyEntryStruct or uint16s
            key_dict = _decode_geotiff_key_list(raw)
        else:                                      # last-chance: raw directory
            key_dict = _decode_geotiff_key_list(getattr(vlr, "key_directory", []))

        epsg = key_dict.get(3072) or key_dict.get(2048)  # PCS or GCS
        if epsg:
            try:
                return CRS.from_epsg(epsg)
            except CRSError:
                pass

    return None


def _split_horizontal_vertical(crs):
    """
    Given a pyproj.CRS (which may be compound), return (horizontal_crs, vertical_crs).

    - For compound CRS: pick projected/geographic part as horizontal, vertical part as vertical.
    - For non-compound CRS: treat it as horizontal, vertical = None.
    - For None: returns (None, None).
    """
    if crs is None:
        return None, None

    if getattr(crs, "is_compound", False) and hasattr(crs, "sub_crs_list"):
        horizontal = None
        vertical = None
        for sub in crs.sub_crs_list:
            if getattr(sub, "is_vertical", False):
                vertical = vertical or sub
            elif sub.is_projected or sub.is_geographic:
                horizontal = horizontal or sub

        if horizontal or vertical:
            return horizontal or crs, vertical

    return crs, None


def get_crs_components_from_header(header):
    """
    From a laspy header return:

        full_crs, horiz_crs, vert_crs, horiz_epsg, vert_epsg

    Always tries its best to return a sensible horizontal CRS / EPSG:
    - If nothing can be parsed, falls back to WGS84 3D (EPSG:4979) and its horizontal part.
    """
    full_crs = _safe_parse_crs(header)
    if full_crs is None:
        full_crs = CRS.from_epsg(4979)  # reasonable default if everything fails

    horiz_crs, vert_crs = _split_horizontal_vertical(full_crs)
    if horiz_crs is None:
        horiz_crs = full_crs

    try:
        horiz_epsg = horiz_crs.to_epsg()
    except Exception:
        horiz_epsg = None

    try:
        vert_epsg = vert_crs.to_epsg() if vert_crs is not None else None
    except Exception:
        vert_epsg = None

    return full_crs, horiz_crs, vert_crs, horiz_epsg, vert_epsg


# ----------------------------------------------------------------------
# Footprint extraction using robust CRS helpers
# ----------------------------------------------------------------------

def extract_footprint_batch(input_folder, output_folder):
    """
    Extracts footprints (convex hulls) from LAS/LAZ files and saves them in a specified folder.

    - Uses get_crs_components_from_header() to get full + horizontal CRS.
    - Uses the horizontal CRS for the 2D footprint GeoDataFrame.
    """
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

                # --- robust CRS parse (full CRS, possibly compound) ---
                full_crs, horiz_crs, vert_crs, horiz_epsg, vert_epsg = get_crs_components_from_header(file.header)

                # For footprints we only need the horizontal CRS.
                # Fallback to WGS 84 if absolutely nothing sensible was found.
                crs_for_footprint = horiz_crs or CRS.from_epsg(4326)

                # Debug/info prints (optional)
                print(f"\nFile: {os.path.basename(laz_file)}")
                print(f"  Full   CRS: {full_crs}")
                print(f"  Horiz. CRS: {horiz_crs}  (EPSG: {horiz_epsg})")
                print(f"  Vert.  CRS: {vert_crs}    (EPSG: {vert_epsg})")
                print(f"  Using horizontal CRS for footprint: {crs_for_footprint}")

            # 2D convex hull on XY
            unique_points = np.unique(np.vstack((x, y)).T, axis=0)
            try:
                hull = ConvexHull(unique_points)
                footprint = Polygon(unique_points[hull.vertices])
            except QhullError:
                print(f"Warning: Convex Hull computation failed for {laz_file}. Creating an empty polygon.")
                footprint = Polygon()

            gdf = gpd.GeoDataFrame({'geometry': [footprint]}, crs=crs_for_footprint)

            output_path = os.path.join(
                output_folder, os.path.splitext(os.path.basename(laz_file))[0] + ".gpkg"
            )
            gdf.to_file(output_path, driver="GPKG")

        except Exception as e:
            print(f"Error processing {laz_file}: {e}")

    print(f"Footprint extraction completed in {timedelta(seconds=int(time.time() - start))}.")
