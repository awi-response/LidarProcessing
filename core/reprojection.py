

import json
import laspy
import numpy as np
import pdal
import pyproj
from pyproj.exceptions import CRSError
try:                                    # laspy ≥ 2.5
    from laspy.vlrs.known import GeoKeyDirectoryVLR
except ImportError:
    try:                                # laspy 2.0 – 2.4  (lower-case “r”)
        from laspy.vlrs.known import GeoKeyDirectoryVlr as GeoKeyDirectoryVLR
    except ImportError:                 # very old laspy (≤ 1.x) – no typed class
        GeoKeyDirectoryVLR = None       # we’ll handle that later

def _decode_geotiff_key_list(key_list):
    """
    Convert a raw GeoKeyDirectory array (list of uint16) to {key_id: value}.
    Works for LAS files written by very old laspy / libLAS that store only the
    key directory and inline values (tiffTagLocation == 0).
    """
    if len(key_list) < 4:                       # need at least the header
        return {}

    num_keys = key_list[3]
    out = {}
    # keys start at offset 4, four uint16 per entry
    for i in range(num_keys):
        base = 4 + i * 4
        key_id, tag_loc, count, value = key_list[base : base + 4]
        if tag_loc == 0 and count == 1:         # “value is in 'value' field”
            out[key_id] = value
    return out

def _safe_parse_crs(header):
    """
    Return a pyproj.CRS or None.
      • First ask laspy.header.parse_crs()  → covers WKT + clean GeoTIFF.
      • If that fails, walk the GeoTIFF VLRs manually.
    """
    try:
        crs = header.parse_crs()    
        
        print(f"[info] CRS from WKT: {crs}")  # debug output
                # may raise CRSError on bad WKT
        if crs and not crs.is_empty:
            return crs                      # success via WKT or GeoTIFF
    except CRSError:
        pass                                # fall through to manual path


    for vlr in header.vlrs:
        if (GeoKeyDirectoryVLR and isinstance(vlr, GeoKeyDirectoryVLR)) or (
            vlr.user_id == "LASF_Projection" and vlr.record_id == 34735
        ):
            raw = getattr(vlr, "geo_keys", None)
            if isinstance(raw, dict):                   # laspy ≥ 2.3
                key_dict = raw
            elif isinstance(raw, list):                 # laspy ≤ 2.2
                key_dict = _decode_geotiff_key_list(raw)
            else:                                       # last-chance: raw dir
                key_dict = _decode_geotiff_key_list(
                    getattr(vlr, "key_directory", [])
                )

            epsg = key_dict.get(3072) or key_dict.get(2048)  # PCS or GCS
            if epsg:
                try:
                    return pyproj.CRS.from_epsg(epsg)
                except CRSError:
                    pass

    return None



def is_utm_crs(las_file):
    """True iff the file already uses a UTM EPSG code."""
    with laspy.open(las_file) as f:
        try:
            crs = f.header.parse_crs()
        except CRSError:
            crs = _safe_parse_crs(f.header)

        epsg = crs.to_epsg() if crs else None

    if epsg is None:
        print(f"[warn] {las_file}: CRS missing or unreadable – will re-project")
        return False

    # UTM North = 32601-32660, South = 32701-32760
    return 32600 < epsg < 32800


def get_utm_epsg(las_file, src_crs):
    """
    Choose a UTM EPSG code that fits the point cloud’s centroid (in WGS-84).
    """
    with laspy.open(las_file) as f:
        pts = f.read()
        x = pts.X * f.header.scale[0] + f.header.offset[0]
        y = pts.Y * f.header.scale[1] + f.header.offset[1]

    lon, lat = (x.mean(), y.mean())

    # Convert projected coordinates back to lon/lat if necessary
    if src_crs and not src_crs.is_geographic:
        transformer = pyproj.Transformer.from_crs(src_crs, 4326, always_xy=True)
        lon, lat = transformer.transform(lon, lat)

    zone      = int((lon + 180) / 6) + 1
    northern  = lat >= 0
    return (32600 if northern else 32700) + zone


def reproject_las(input_las, output_las):
    """
    Reproject *input_las* to an appropriate UTM CRS (if needed) and write the
    result to *output_las*.  Returns the path that should be used downstream
    (either the original file or the new one).
    """
    if is_utm_crs(input_las):
        return input_las                      # nothing to do

    with laspy.open(input_las) as f:

        try:
            src_crs = f.header.parse_crs()
        except CRSError:
            src_crs = _safe_parse_crs(f.header) or pyproj.CRS("EPSG:4326")

    dst_epsg = get_utm_epsg(input_las, src_crs)
    #print(f"[info] Reprojecting {input_las}  →  EPSG:{dst_epsg}")

    pipeline = [
        {"type": "readers.las",         "filename": input_las},
        {"type": "filters.reprojection",
         "in_srs": src_crs.to_wkt(),    "out_srs": f"EPSG:{dst_epsg}"},
        {"type": "writers.las",         "filename": output_las},
    ]
    pdal.Pipeline(json.dumps(pipeline)).execute()
    return output_las
