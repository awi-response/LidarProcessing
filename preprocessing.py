import pdal
import json
import laspy
import shutil
from multiprocessing import Pool
from shapely.geometry import shape, box, mapping
import os
import time
import numpy as np
import geopandas as gpd
from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps
from tqdm import tqdm
from datetime import timedelta
from core.reprojection import get_utm_epsg, reproject_las, is_utm_crs
from core.preprocess_windowed import create_chunks_from_wkt, process_chunk, merge_and_crop_chunks
from core.extract_footprints import extract_footprint_batch


def match_footprints(target_footprint_dir, las_footprint_dir, las_file_dir, threshold=0.5):
    """Matches target footprints with LAS files and returns a dictionary mapping target area names to LAS file paths."""
    
    os.makedirs(las_footprint_dir, exist_ok=True)
    
    print("\nMatching Lidar footprints...")
    start = time.time()

    if not os.listdir(las_footprint_dir):
        print("No footprint files found. Generating footprints first.")
        extract_footprint_batch(las_file_dir, las_footprint_dir)

    target_footprints = [os.path.join(target_footprint_dir, f) for f in os.listdir(target_footprint_dir) if f.endswith(".gpkg")]
    las_footprints = [os.path.join(las_footprint_dir, f) for f in os.listdir(las_footprint_dir) if f.endswith(".gpkg")]

    target_dict = {}

    for target_fp in target_footprints:
        target_gdf = gpd.read_file(target_fp)
        target_name = os.path.splitext(os.path.basename(target_fp))[0]
        las_paths = []

        for las_fp in las_footprints:
            las_gdf = gpd.read_file(las_fp)
            if target_gdf.crs != las_gdf.crs:
                las_gdf = las_gdf.to_crs(target_gdf.crs)

            joined = gpd.sjoin(las_gdf, target_gdf, predicate="intersects")
            if not joined.empty:
                intersection = gpd.overlay(las_gdf, target_gdf, how="intersection")
                intersection_area = intersection.area.sum()
                target_gef_area = target_gdf.geometry.area.sum()

                if intersection_area / target_gef_area > threshold:
                    las_name = os.path.splitext(os.path.basename(las_fp))[0] + ".las"
                    las_path = os.path.join(las_file_dir, las_name)

                    if os.path.exists(las_path):
                        las_paths.append(las_path)

        target_dict[target_name] = las_paths

        # Print the number of LAS files found for each target area
        print(f"Target area: {target_name}, LAS files found: {len(las_paths)}")

    print(f"Footprint matching completed in {timedelta(seconds=int(time.time() - start))}. Found {len(target_dict)} target areas.")
    return target_dict


def get_las_header(las_file):
    """Extracts scale, offset, and CRS from an input LAS file."""
    with laspy.open(las_file) as las:
        header = las.header
        scale = header.scales
        offset = header.offsets
        crs = header.parse_crs()
        crs_epsg = crs.to_epsg() if crs else 4979  # Default to EPSG:4979 if unknown
    return scale, offset, crs_epsg


def process_chunk_wrapper(args):
    """Unpacks arguments for multiprocessing."""
    return process_chunk(*args)


def merge_and_clean_las(las_dict, preprocessed_dir, run_name, target_footprint_dir, sor_knn, sor_multiplier, num_workers, chunk_size=1000):
    """
    Processes LAS files by splitting them into chunks, applying SOR filtering, merging the results,
    and cropping them to the target area's boundary. Also ensures LAS files and target geometries are in UTM projection.
    """
    run_merged_dir = os.path.join(preprocessed_dir, run_name)
    os.makedirs(run_merged_dir, exist_ok=True)
    
    print("\nProcessing LAS files in chunks...")
    start = time.time()
    
    for target_fp, las_files in tqdm(las_dict.items(), desc="Processing target areas", unit="area"):
        if not las_files:
            print(f"No valid LAS files for {target_fp}. Skipping.")
            continue

        # Load the corresponding footprint file
        footprint_path = os.path.join(target_footprint_dir, target_fp if target_fp.endswith('.gpkg') else f"{target_fp}.gpkg")
        if not os.path.exists(footprint_path):
            print(f"Footprint file {footprint_path} not found. Skipping clip.")
            continue

        gdf = gpd.read_file(footprint_path)
        temp_dir = os.path.join(run_merged_dir, target_fp, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        # Process each LAS file assigned to this target area
        processed_chunks = []
        process_args = []
        for input_file in las_files:
            
            # Ensure LAS file is reprojected to UTM
            if not is_utm_crs(input_file):
                utm_output_file = os.path.join(temp_dir, f"{os.path.basename(input_file).replace('.las', '_utm.las')}")
                input_file = reproject_las(input_file, input_file)
            
            ref_scale, ref_offset, ref_crs = get_las_header(input_file)
            
            # Reproject target geometry if necessary
            if gdf.crs.to_epsg() != ref_crs:
                gdf = gdf.to_crs(epsg=ref_crs)
            
            target_geom_wkt = wkt_dumps(shape(gdf.geometry.iloc[0]))
            chunks = create_chunks_from_wkt(target_geom_wkt, chunk_size)  # 100m chunk size
            
            for chunk in chunks:
                process_args.append((input_file, chunk, temp_dir, sor_knn, sor_multiplier, ref_scale, ref_offset, ref_crs))

        # Parallel processing of chunks
        with tqdm(total=len(process_args), desc=f"Processing {target_fp}", unit="chunk") as pbar:
            with Pool(processes=num_workers) as pool:  # Adjust workers as needed
                for processed_chunk in pool.imap_unordered(process_chunk_wrapper, process_args):
                    if processed_chunk:
                        processed_chunks.append(processed_chunk)
                    pbar.update(1)

        # Merge and crop chunks
        if processed_chunks:
            clean_target_fp = os.path.splitext(target_fp)[0]
            final_output_file = os.path.join(run_merged_dir, f"{clean_target_fp}.las")
            merge_and_crop_chunks(processed_chunks, target_geom_wkt, final_output_file)
            print(f"Final processed LAS file saved: {final_output_file}")
        else:
            print(f"No processed chunks available for {target_fp}.")

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    print(f"\nProcessing completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.")





def preprocess_all(conf):
    """Runs full preprocessing pipeline for LAS data."""

    global config
    config = conf

    print("\n========== Starting Preprocessing ==========")
    start = time.time()

    run_name = config.run_name

    # Ensure required directories exist
    os.makedirs(os.path.join(config.preprocessed_dir, run_name), exist_ok=True)
    os.makedirs(os.path.join(config.results_dir, run_name), exist_ok=True)

    # Step 1: Match footprints with LAS files
    print("\n--- Matching footprints to LAS files ---")
    target_dict = match_footprints(
        target_footprint_dir=config.target_area_dir, 
        las_footprint_dir=config.las_footprints_dir, 
        las_file_dir=config.las_files_dir,
        threshold=config.overlap
        #run_name=run_name
    )

    # Step 2: Merge and Clean LAS files using Statistical Outlier Removal (SOR)
    print("\n--- Merging and Cleaning LAS files ---")
    merge_and_clean_las(
        target_footprint_dir=config.target_area_dir,
        las_dict=target_dict, 
        preprocessed_dir=config.preprocessed_dir, 
        sor_knn=config.knn,  # Adjust based on density
        sor_multiplier=config.multiplier,  # Adjust based on noise level
        num_workers=config.num_workers,
        run_name=run_name,
        chunk_size=config.chunk_size
        
    )

    print(f"\nPreprocessing completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n")
