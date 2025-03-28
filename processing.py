import os
import time
import json
import glob
import pdal
import laspy
import numpy as np
import subprocess
from datetime import timedelta
from tqdm import tqdm
from scipy.spatial import KDTree
import rasterio
import shutil 
import laspy
from shapely.geometry import box
import multiprocessing
from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps

from core.processing_windowed import create_chunks_from_wkt, process_chunk_to_dsm, process_chunk_to_dem ,merge_chunks


def check_resolution(las_file, resolution, method="sampling", num_samples=10000):
    """
    Checks if the DSM resolution is appropriate based on point cloud density.

    Parameters:
        las_file (str): Path to the LAS/LAZ file.
        resolution (float): Desired DSM resolution.
        method (str): "sampling" (nearest neighbor) or "density" (Poisson estimate).
        num_samples (int): Number of random samples for nearest neighbor method.

    Returns:
        float: Estimated average point spacing.
        bool: Whether the resolution is appropriate.
    """
    with laspy.open(las_file) as file:
        point_cloud = file.read()
        points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).T

    if len(points) == 0:
        raise ValueError(f"Point cloud {las_file} is empty.")

    if method == "sampling":
        num_samples = min(num_samples, len(points))
        sampled_points = points[np.random.choice(len(points), num_samples, replace=False)]
        tree = KDTree(points)
        distances, _ = tree.query(sampled_points, k=2)
        avg_distance = np.mean(distances[:, 1])  # Ignore self-distance

    elif method == "density":
        bbox_volume = np.prod(points.max(axis=0) - points.min(axis=0))
        density = len(points) / bbox_volume if bbox_volume > 0 else float('inf')
        avg_distance = (1 / density) ** (1 / 3)

    else:
        raise ValueError("Invalid method. Choose 'sampling' or 'density'.")

    #if avg_distance > resolution:
        #print(f"Warning: DSM resolution ({resolution}m) is finer than average point spacing ({avg_distance:.3f}m). "
              #f"This may cause interpolation gaps.")

    return avg_distance, avg_distance <= resolution


def get_las_footprint_wkt(las_file):
    """Extracts the WKT footprint (bounding box) from a LAS file."""
    
    with laspy.open(las_file) as las:
        header = las.header
        min_x, min_y, max_x, max_y = header.min[0], header.min[1], header.max[0], header.max[1]

    # Create a bounding box polygon
    footprint = box(min_x, min_y, max_x, max_y)
    return footprint.wkt  # Convert to WKT format

def process_las_file_dsm(las_file, temp_folder, final_output_folder, resolution, method, fill_gaps, counter, chunk_size):
    """Processes a single LAS file: generates DSM chunks, merges them, and fills gaps if needed."""
    
    base_name = os.path.splitext(os.path.basename(las_file))[0]
    
    # Generate WKT footprint from the LAS file
    target_wkt = get_las_footprint_wkt(las_file)

    avg_spacing, is_resolution_ok = check_resolution(las_file, resolution, method)
    if not is_resolution_ok:
        print(f"Warning: DSM resolution ({resolution}m) is finer than average point spacing ({avg_spacing:.3f}m).")
        print("   Consider increasing the resolution to avoid interpolation gaps.")

    # Create a temporary directory for DSM chunks
    temp_dsm_dir = os.path.join(temp_folder, base_name)
    os.makedirs(temp_dsm_dir, exist_ok=True)


    final_dsm_path = os.path.join(final_output_folder, f"{base_name}.tif")
    
    # Generate overlapping chunks from WKT
    large_chunks, small_chunks = create_chunks_from_wkt(target_wkt, chunk_size=chunk_size, overlap=0.2)

    dsm_chunks = []

    # Local progress bar for chunk processing (each file)
    for large_chunk, small_chunk in tqdm(zip(large_chunks, small_chunks), desc=f"Processing Chunks ({base_name})", unit="chunk", leave=False):
        chunk_dsm_path = process_chunk_to_dsm(las_file, large_chunk, small_chunk, temp_dsm_dir, resolution)
        if chunk_dsm_path:
            dsm_chunks.append(chunk_dsm_path)

    # Merge chunks into a single DSM
    if temp_dsm_dir:
        chunk_files = sorted(glob.glob(os.path.join(temp_dsm_dir, "*.tif")))

        merged_dsm = merge_chunks(chunk_files, final_dsm_path)
        
        # Fill gaps if needed
        if fill_gaps and merged_dsm:
            filled_dsm_path = os.path.join(temp_dsm_dir, f"{base_name}_filled.tif")
            subprocess.run([
                "gdal_fillnodata.py",
                "-md", "10",
                "-si", "2",
                merged_dsm,
                filled_dsm_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.replace(filled_dsm_path, final_dsm_path)

    else:
        print(f"No chunks found for {base_name}. Skipping merge.")

    # Cleanup temporary directory
    temp_root_dir = os.path.join(final_output_folder, "temp")

    if os.path.exists(temp_root_dir):
        shutil.rmtree(temp_root_dir, ignore_errors=True)

    # Increment the counter (safe in multiprocessing)
    counter.value += 1

    return final_dsm_path


def generate_dsm(input_folder, output_folder, run_name, method, resolution, chunk_size, num_workers, fill_gaps=True):
    """Parallelized DSM generation for all LAS files in a folder."""
    
    # Ensure final output folders exist
    final_output_folder = os.path.join(output_folder, run_name, 'DSM')
    os.makedirs(final_output_folder, exist_ok=True)
    
    temp_folder = os.path.join(final_output_folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)

    start_time = time.time()

    # Find all LAS/LAZ files
    las_files = glob.glob(os.path.join(input_folder, run_name, "*.las")) + \
                glob.glob(os.path.join(input_folder, run_name, "*.laz"))

    if not las_files:
        print("No LAS/LAZ files found. Exiting DSM generation.")
        return

    # Use multiprocessing Manager to create a shared counter
    with multiprocessing.Manager() as manager:
        counter = manager.Value('i', 0)  # Shared integer counter

        # Progress bar in the main process
        with tqdm(total=len(las_files), desc="Processing LAS Files", unit="file") as progress_bar:
            with multiprocessing.Pool(processes=num_workers) as pool:
                async_results = [
                    pool.apply_async(
                        process_las_file_dsm, 
                        (las_file, temp_folder, final_output_folder, resolution, method, fill_gaps, counter, chunk_size)
                    ) for las_file in las_files
                ]

                # Update progress bar dynamically
                while counter.value < len(las_files):
                    progress_bar.n = counter.value
                    progress_bar.refresh()
                    time.sleep(1)  # Small delay to prevent excessive updates
                
                # Wait for all processes to finish
                for result in async_results:
                    result.get()

    elapsed_time = timedelta(seconds=int(time.time() - start_time))
    print(f"\nDSM generation completed in {elapsed_time}.")


def process_las_file_dem(las_file, temp_folder, final_output_folder, resolution, method, rigidness, iterations, fill_gaps, counter, chunk_size):
    """Processes a single LAS file: generates DTM chunks, merges them, and fills gaps if needed."""
    base_name = os.path.splitext(os.path.basename(las_file))[0]
    
    # Generate WKT footprint from the LAS file
    target_wkt = get_las_footprint_wkt(las_file)

    avg_spacing, is_resolution_ok = check_resolution(las_file, resolution, method)
    if not is_resolution_ok:
        print(f"Warning: DTM resolution ({resolution}m) is finer than average point spacing ({avg_spacing:.3f}m).")
        print("   Consider increasing the resolution to avoid interpolation gaps.")
    
    # Create a temporary directory for DTM chunks
    temp_dtm_dir = os.path.join(temp_folder, base_name)
    os.makedirs(temp_dtm_dir, exist_ok=True)

    final_dtm_path = os.path.join(final_output_folder, f"{base_name}.tif")
    
    # Generate overlapping chunks from WKT
    large_chunks, small_chunks = create_chunks_from_wkt(target_wkt, chunk_size=chunk_size, overlap=0.2)
    
    dtm_chunks = []
    
    # Process each chunk
    for large_chunk, small_chunk in tqdm(zip(large_chunks, small_chunks), desc=f"Processing Chunks ({base_name})", unit="chunk", leave=False):
        chunk_dtm_path = process_chunk_to_dem(las_file, large_chunk, small_chunk, temp_dtm_dir, rigidness, iterations, resolution, fill_gaps)
        if chunk_dtm_path:
            dtm_chunks.append(chunk_dtm_path)
    
    # Merge chunks into a single DTM
    # Merge chunks into a single DSM
    if temp_dtm_dir:
        chunk_files = sorted(glob.glob(os.path.join(temp_dtm_dir, "*.tif")))

        merged_dsm = merge_chunks(chunk_files, final_dtm_path)
        
        # Fill gaps if needed
        if fill_gaps and merged_dsm:
            filled_dsm_path = os.path.join(temp_dtm_dir, f"{base_name}_filled.tif")
            subprocess.run([
                "gdal_fillnodata.py",
                "-md", "10",
                "-si", "2",
                merged_dsm,
                filled_dsm_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.replace(filled_dsm_path, final_dtm_path)

    else:
        print(f"No chunks found for {base_name}. Skipping merge.")

    # Cleanup temporary directory
    temp_root_dir = os.path.join(final_output_folder, "temp")

    if os.path.exists(temp_root_dir):
        shutil.rmtree(temp_root_dir, ignore_errors=True)

    # Increment the counter (safe in multiprocessing)
    counter.value += 1

    return final_dtm_path

def generate_dtm(input_folder, output_folder, run_name, method, rigidness, iterations, resolution, chunk_size, num_workers, fill_gaps=True):
    """Parallelized DTM generation for all LAS files in a folder."""
    
    # Ensure final output folders exist
    final_output_folder = os.path.join(output_folder, run_name, 'DTM')
    os.makedirs(final_output_folder, exist_ok=True)
    
    temp_folder = os.path.join(final_output_folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)
    
    start_time = time.time()
    
    # Find all LAS/LAZ files
    las_files = glob.glob(os.path.join(input_folder, run_name, "*.las")) + \
                glob.glob(os.path.join(input_folder, run_name, "*.laz"))
    
    if not las_files:
        print("No LAS/LAZ files found. Exiting DTM generation.")
        return
    
    # Use multiprocessing Manager to create a shared counter
    with multiprocessing.Manager() as manager:
        counter = manager.Value('i', 0)  # Shared integer counter
        
        # Progress bar in the main process
        with tqdm(total=len(las_files), desc="Processing LAS Files", unit="file") as progress_bar:
            with multiprocessing.Pool(processes=num_workers) as pool:
                async_results = [
                    pool.apply_async(
                        process_las_file_dem, 
                        (las_file, temp_folder, final_output_folder, resolution, method, rigidness, iterations, fill_gaps, counter, chunk_size)
                    ) for las_file in las_files
                ]
                
                # Update progress bar dynamically
                while counter.value < len(las_files):
                    progress_bar.n = counter.value
                    progress_bar.refresh()
                    time.sleep(1)  # Small delay to prevent excessive updates
                
                # Wait for all processes to finish
                for result in async_results:
                    result.get()
    
    elapsed_time = timedelta(seconds=int(time.time() - start_time))
    print(f"\nDTM generation completed in {elapsed_time}.")


def generate_chm(input_folder, output_folder, run_name):
    """
    Generates Canopy Height Models (CHM) for all corresponding DSM and DTM files in the given folders.

    Parameters:
        input_folder (str): Base folder containing DSM and DTM subfolders.
        output_folder (str): Base folder where CHM outputs will be saved.
        run_name (str): Name of the subfolder (same for DSM, DTM, and CHM).

    Returns:
        None
    """
    dsm_folder = os.path.join(input_folder, run_name, "DSM")
    DTM_folder = os.path.join(input_folder, run_name, "DTM")
    chm_folder = os.path.join(output_folder, run_name, "CHM")

    # Ensure output folder exists
    os.makedirs(chm_folder, exist_ok=True)

    # Find all DSM files
    dsm_files = glob.glob(os.path.join(dsm_folder, "*.tif"))

    print("\nStarting CHM generation")
    start_time = time.time()
    

    if not dsm_files:
        print(f"No DSM files found in {dsm_folder}. Exiting CHM generation.")
        return

    for dsm_path in tqdm(dsm_files, desc="Processing CHMs", unit="file"):
        try:
            # Extract base name (without extension)
            base_name = os.path.splitext(os.path.basename(dsm_path))[0]
            
            # Find the corresponding DTM file
            base_name = os.path.splitext(os.path.basename(dsm_path))[0].replace("_DSM", "")
            DTM_path = os.path.join(DTM_folder, f"{base_name}_DTM.tif")
            chm_output_path = os.path.join(chm_folder, f"{base_name}_CHM.tif")

            if not os.path.exists(DTM_path):
                print(f"Skipping {base_name}: Corresponding DTM not found.")
                continue

            # Open DSM and DTM rasters
            with rasterio.open(dsm_path) as dsm_src, rasterio.open(DTM_path) as DTM_src:
                # Read the raster data
                dsm = dsm_src.read(1)
                dtm = DTM_src.read(1)
                
                # Ensure they have the same shape
                if dsm.shape != dtm.shape:
                    print(f"Skipping {base_name}: DSM and DTM raster sizes do not match.")
                    continue

                # Calculate CHM by subtracting DTM from DSM
                chm = dsm - dtm

                # Handle NoData values
                chm[dsm == dsm_src.nodata] = dsm_src.nodata
                chm[dtm == DTM_src.nodata] = DTM_src.nodata

                # Define output metadata
                chm_meta = dsm_src.meta.copy()
                chm_meta.update(dtype=rasterio.float32)

                # Save the CHM raster
                with rasterio.open(chm_output_path, "w", **chm_meta) as chm_dst:
                    chm_dst.write(chm.astype(rasterio.float32), 1)

                

        except Exception as e:
            print(f"Error processing {base_name}: {e}")

    elapsed_time = timedelta(seconds=int(time.time() - start_time))
    print(f"\n CHM generation completed in {elapsed_time}.")


def process_all(config):
    """
    Runs DSM generation using cleaned LAS files.

    Reads from: `config.preprocessed_dir`
    Saves to: `config.results_dir / run_name / DSM/`
    """
    print('Starting Processing ...')

    start_time = time.time()

    if config.create_DSM:
        print("\n========== Starting DSM Generation ==========")
        generate_dsm(
            input_folder=config.preprocessed_dir,
            output_folder=config.results_dir,
            run_name=config.run_name,
            resolution=config.resolution,
            chunk_size=config.chunk_size,
            fill_gaps=config.fill_gaps,
            num_workers=config.num_workers, 
            method=config.point_density_method
        )
    
    

    if config.create_DEM:
        print("\n========== Starting DEM Generation ==========")
        generate_dtm(
            input_folder=config.preprocessed_dir,
            output_folder=config.results_dir,
            run_name=config.run_name,
            resolution=config.resolution,
            chunk_size=config.chunk_size,
            fill_gaps=config.fill_gaps,
            method=config.point_density_method, 
            rigidness = config.rigidness,
            iterations = config.iterations,
            num_workers= config.num_workers
        )

    if config.create_CHM:
        print("\n========== Starting CHM Generation ==========")
        generate_chm(
            input_folder=config.results_dir,
            output_folder=config.results_dir,
            run_name=config.run_name
        )

    elapsed_time = timedelta(seconds=int(time.time() - start_time))
    print(f"\n DEM generation completed in {elapsed_time}.\n")