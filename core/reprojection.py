import json
import laspy
import numpy as np
import pdal

def is_utm_crs(las_file):
    """Checks if a LAS/LAZ file is in a UTM projection."""
    with laspy.open(las_file) as file:
        crs = file.header.parse_crs()
        crs_epsg = crs.to_epsg() if crs else None

    if crs_epsg is None:
        print(f"Warning: No CRS found in {las_file}. Assuming it needs reprojection.")
        return False

    return 32600 <= crs_epsg <= 32799  # UTM zones are EPSG:32600-32660 (Northern) & EPSG:32700-32760 (Southern)


def get_utm_epsg(las_file):
    """Detects the best UTM EPSG code based on the LAS file's longitude."""
    with laspy.open(las_file) as file:
        point_cloud = file.read()
        avg_longitude = np.mean(point_cloud.x)

    utm_zone = int((avg_longitude + 180) / 6) + 1
    is_northern = np.mean(point_cloud.y) >= 0  # Check if the point cloud is in the northern hemisphere
    epsg_code = 32600 + utm_zone if is_northern else 32700 + utm_zone

    #print(f"Detected UTM Zone: {utm_zone}, EPSG: {epsg_code}")
    return epsg_code


def reproject_las(input_las, output_las):
    """Reprojects a LAS/LAZ file to UTM if it's not already in a UTM CRS."""
    if is_utm_crs(input_las):
        print(f"Skipping reprojection for {input_las}: Already in UTM.")
        return input_las  # Return original file if already projected

    target_epsg = get_utm_epsg(input_las)

    with laspy.open(input_las) as file:
        crs = file.header.parse_crs()
        print(f"CRS found in {input_las}: {crs}")

        if crs is None:
            crs = 4326
            print (f"No CRS found in {input_las}. Assuming EPSG:4326 for reprojection.")

        print(f"Reprojecting {input_las} from {crs} to EPSG:{target_epsg}")


    pipeline = [
        {"type": "readers.las", "filename": input_las},
        {"type": "filters.reprojection", "in_srs": f"{crs}", "out_srs": f"EPSG:{target_epsg}"},
        {"type": "writers.las", "filename": output_las}
    ]

    #print(f"Reprojecting {input_las} to EPSG:{target_epsg} -> {output_las}")
    pdal.pipeline.Pipeline(json.dumps(pipeline)).execute()
    return output_las  # Return new file path
