import laspy
import pdal
import numpy as np
import pandas as pd
import os
import rasterio
from rasterio.merge import merge
from osgeo import gdal
import json
import subprocess
from shapely.geometry import box, shape
from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps

def create_chunks_from_wkt(input_wkt, chunk_size=1000, overlap=0.2):
    """
    create processing chunks based on wkt geometry of target area, with overlap enlarged by x percent, 
    returns geometries for enlarged chunks and original chunk size, use original chunk size for cliping of results and large for cloth method
    """

    geom= wkt_loads(input_wkt)
    min_x, min_y, max_x, max_y = geom.bounds
    large_chunks = []
    orig_chunk_size = []

    enlarged_chunk_size = chunk_size * (1 + overlap)
    half_extra = (enlarged_chunk_size - chunk_size) / 2

    for x in np.arange(min_x, max_x, chunk_size):
        for y in np.arange(min_y, max_y, chunk_size):
            large_chunk_bbox = box(x - half_extra, y - half_extra, x + enlarged_chunk_size - half_extra, y + enlarged_chunk_size - half_extra)
            orig_chunk_box = box(x, y, x + chunk_size, y + chunk_size)
            if geom.intersects(orig_chunk_box):
                large_chunks.append(large_chunk_bbox)
                orig_chunk_size.append(orig_chunk_box)
    
    return large_chunks, orig_chunk_size

def process_chunk_to_dsm(input_file, chunk_bbox, temp_dir, resolution):

    chunk_file = os.path.join(temp_dir, f"{os.path.basename(input_file).replace('.las', '')}_chunk_{int(chunk_bbox.bounds[0])}_{int(chunk_bbox.bounds[1])}.tif")

    pipeline = [
        {"type": "readers.las", "filename": input_file},
        {"type": "filters.crop", "polygon": wkt_dumps(chunk_bbox)},
        {"type": "filters.ferry", "dimensions": "Z=>Elevation"},
        {
            "type": "filters.range",
            "limits": "Classification[0:0]"  # Use all points for initial DSM
        },
        {
            "type": "writers.gdal",
            "filename": chunk_file,
            "resolution": resolution,
            "output_type": "max",
            "nodata": -9999,
            "gdalopts": "COMPRESS=LZW"
                }
            ]
            
            # Run PDAL pipeline.
    pdal.pipeline.Pipeline(json.dumps(pipeline)).execute()

    subprocess.run([
        "gdal_fillnodata.py",
        "-md", "10",
        "-si", "2",
        chunk_file,
        chunk_file
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        #print(f"Filled DSM saved (temp): {temp_filled_dsm_path}")
        # Move the gap-filled DSM to the final output folder.


def merge_chunks(input_files, output_file):
    """
    Merges multiple raster files into a single raster and saves the output.
    
    Parameters:
        input_files (list): List of file paths to raster files.
        output_file (str): Path to the output merged raster file.
    """
    
    # Open all raster files
    src_files = [rasterio.open(f) for f in input_files]
    
    # Merge rasters
    mosaic, out_transform = merge(src_files)
    
    # Copy metadata from one of the source files
    out_meta = src_files[0].meta.copy()
    
    # Update metadata for the merged raster
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform
    })
    
    # Write the merged raster to disk
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    # Close all source files
    for src in src_files:
        src.close()
    
    print(f"Merged raster saved at: {output_file}")