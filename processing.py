import os
from pathlib import Path
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
from shapely.geometry import box
import multiprocessing
from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps
from typing import Tuple, List, Optional
from core.processing_windowed import create_chunks_from_wkt, process_chunk_to_dsm, process_chunk_to_dem, merge_chunks

class PointCloudProcessor:
    def __init__(self, config):
        """Initialize processor with configuration settings."""
        self.config = config
        self.temp_dirs = set()
        
    def cleanup(self):
        """Clean up temporary directories."""
        for dir_path in self.temp_dirs:
            shutil.rmtree(dir_path, ignore_errors=True)

    def check_resolution(self, las_file: str, resolution: float, method: str = "sampling", 
                        num_samples: int = 10000) -> Tuple[float, bool]:
        """Check DSM resolution against point cloud density."""
        try:
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
                    avg_distance = np.mean(distances[:, 1])
                elif method == "density":
                    bbox_volume = np.prod(points.max(axis=0) - points.min(axis=0))
                    density = len(points) / bbox_volume if bbox_volume > 0 else float('inf')
                    avg_distance = (1 / density) ** (1 / 3)
                else:
                    raise ValueError("Invalid method. Choose 'sampling' or 'density'.")
                
                return avg_distance, avg_distance <= resolution
                
        except Exception as e:
            print(f"Error checking resolution for {las_file}: {e}")
            return float('inf'), False

    def get_las_footprint_wkt(self, las_file: str) -> Optional[str]:
        """Extract WKT footprint from LAS file."""
        try:
            with laspy.open(las_file) as las:
                header = las.header
                min_x, min_y, max_x, max_y = header.min[0], header.min[1], header.max[0], header.max[1]
                footprint = box(min_x, min_y, max_x, max_y)
                return footprint.wkt
        except Exception as e:
            print(f"Error getting WKT footprint for {las_file}: {e}")
            return None

    def _process_chunk(self, chunk_data):
        """Process a single chunk."""
        las_file, temp_dir, output_folder, resolution, method, fill_gaps, counter, chunk_size = chunk_data
        
        base_name = os.path.splitext(os.path.basename(las_file))[0]
        target_wkt = self.get_las_footprint_wkt(las_file)
        
        if not target_wkt:
            return None
            
        avg_spacing, is_resolution_ok = self.check_resolution(las_file, resolution, method)
        if not is_resolution_ok:
            print(f"Warning: Resolution ({resolution}m) is finer than average spacing ({avg_spacing:.3f}m)")
            
        temp_dsm_dir = Path(temp_dir) / base_name
        os.makedirs(temp_dsm_dir, exist_ok=True)
        self.temp_dirs.add(temp_dsm_dir)
        
        final_dsm_path = Path(output_folder) / f"{base_name}.tif"
        large_chunks, small_chunks = create_chunks_from_wkt(target_wkt, chunk_size=chunk_size, overlap=0.2)
        
        dsm_chunks = []
        for large_chunk, small_chunk in tqdm(zip(large_chunks, small_chunks), 
                                           desc=f"Processing Chunks ({base_name})", 
                                           unit="chunk", leave=False):
            chunk_dsm_path = process_chunk_to_dsm(las_file, large_chunk, small_chunk, temp_dsm_dir, resolution)
            if chunk_dsm_path:
                dsm_chunks.append(chunk_dsm_path)
                
        if dsm_chunks:
            merged_dsm = merge_chunks(dsm_chunks, final_dsm_path)
            
            if fill_gaps and merged_dsm:
                filled_dsm_path = temp_dsm_dir / f"{base_name}_filled.tif"
                subprocess.run([
                    "gdal_fillnodata.py",
                    "-md", "10",
                    "-si", "2",
                    str(merged_dsm),
                    str(filled_dsm_path)
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.replace(filled_dsm_path, final_dsm_path)
                
        counter.value += 1
        return final_dsm_path

    def generate_dsm(self, input_folder: str, output_folder: str, run_name: str, 
                     resolution: float, chunk_size: int, num_workers: int, 
                     fill_gaps: bool = True, method: str = "sampling"):
        """Generate DSM for all LAS files in folder."""
        final_output_folder = Path(output_folder) / run_name / "DSM"
        temp_folder = final_output_folder / "temp"
        final_output_folder.mkdir(parents=True, exist_ok=True)
        temp_folder.mkdir(exist_ok=True)
        
        las_files = glob.glob(str(Path(input_folder) / run_name / "*.las")) + \
                   glob.glob(str(Path(input_folder) / run_name / "*.laz"))
        
        if not las_files:
            print("No LAS/LAZ files found. Exiting DSM generation.")
            return
            
        counter = multiprocessing.Value('i', 0)
        chunk_data = [(las_file, str(temp_folder), str(final_output_folder), resolution, 
                      method, fill_gaps, counter, chunk_size) 
                      for las_file in las_files]
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            with tqdm(total=len(las_files), desc="Processing LAS Files", unit="file") as progress_bar:
                async_results = [pool.apply_async(self._process_chunk, args=(data,))
                               for data in chunk_data]
                
                while counter.value < len(las_files):
                    progress_bar.n = counter.value
                    progress_bar.refresh()
                    time.sleep(1)
                    
                for result in async_results:
                    result.get()

    def generate_dtm(self, input_folder: str, output_folder: str, run_name: str, 
                     resolution: float, chunk_size: int, num_workers: int, 
                     fill_gaps: bool = True, method: str = "sampling", 
                     rigidness: float = 1.0, iterations: int = 3):
        """Generate DTM for all LAS files in folder."""
        final_output_folder = Path(output_folder) / run_name / "DTM"
        temp_folder = final_output_folder / "temp"
        final_output_folder.mkdir(parents=True, exist_ok=True)
        temp_folder.mkdir(exist_ok=True)
        
        las_files = glob.glob(str(Path(input_folder) / run_name / "*.las")) + \
                   glob.glob(str(Path(input_folder) / run_name / "*.laz"))
        
        if not las_files:
            print("No LAS/LAZ files found. Exiting DTM generation.")
            return
            
        counter = multiprocessing.Value('i', 0)
        chunk_data = [(las_file, str(temp_folder), str(final_output_folder), resolution, 
                      method, rigidness, iterations, fill_gaps, counter, chunk_size) 
                      for las_file in las_files]
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            with tqdm(total=len(las_files), desc="Processing LAS Files", unit="file") as progress_bar:
                async_results = [pool.apply_async(self._process_chunk_dtm, args=(data,))
                               for data in chunk_data]
                
                while counter.value < len(las_files):
                    progress_bar.n = counter.value
                    progress_bar.refresh()
                    time.sleep(1)
                    
                for result in async_results:
                    result.get()

    def _process_chunk_dtm(self, chunk_data):
        """Process a single chunk for DTM generation."""
        las_file, temp_dir, output_folder, resolution, method, rigidness, iterations, \
            fill_gaps, counter, chunk_size = chunk_data
        
        base_name = os.path.splitext(os.path.basename(las_file))[0]
        target_wkt = self.get_las_footprint_wkt(las_file)
        
        if not target_wkt:
            return None
            
        avg_spacing, is_resolution_ok = self.check_resolution(las_file, resolution, method)
        if not is_resolution_ok:
            print(f"Warning: DTM resolution ({resolution}m) is finer than average spacing ({avg_spacing:.3f}m)")
            
        temp_dtm_dir = Path(temp_dir) / base_name
        os.makedirs(temp_dtm_dir, exist_ok=True)
        self.temp_dirs.add(temp_dtm_dir)
        
        final_dtm_path = Path(output_folder) / f"{base_name}.tif"
        large_chunks, small_chunks = create_chunks_from_wkt(target_wkt, chunk_size=chunk_size, overlap=0.2)
        
        dtm_chunks = []
        for large_chunk, small_chunk in tqdm(zip(large_chunks, small_chunks), 
                                           desc=f"Processing Chunks ({base_name})", 
                                           unit="chunk", leave=False):
            chunk_dtm_path = process_chunk_to_dem(las_file, large_chunk, small_chunk, 
                                                temp_dtm_dir, rigidness, iterations, 
                                                resolution, fill_gaps)
            if chunk_dtm_path:
                dtm_chunks.append(chunk_dtm_path)
                
        if dtm_chunks:
            merged_dtm = merge_chunks(dtm_chunks, final_dtm_path)
            
            if fill_gaps and merged_dtm:
                filled_dtm_path = temp_dtm_dir / f"{base_name}_filled.tif"
                subprocess.run([
                    "gdal_fillnodata.py",
                    "-md", "10",
                    "-si", "2",
                    str(merged_dtm),
                    str(filled_dtm_path)
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.replace(filled_dtm_path, final_dtm_path)
                
        counter.value += 1
        return final_dtm_path

def generate_chm(input_folder: str, output_folder: str, run_name: str):
    """Generate Canopy Height Models (CHM) from DSM and DTM files."""
    dsm_folder = Path(input_folder) / run_name / "DSM"
    dtm_folder = Path(input_folder) / run_name / "DTM"
    chm_folder = Path(output_folder) / run_name / "CHM"
    
    chm_folder.mkdir(parents=True, exist_ok=True)
    dsm_files = glob.glob(str(dsm_folder / "*.tif"))
    
    if not dsm_files:
        print(f"No DSM files found in {dsm_folder}. Exiting CHM generation.")
        return
    
    for dsm_path in tqdm(dsm_files, desc="Processing CHMs", unit="file"):
        base_name = os.path.splitext(os.path.basename(dsm_path))[0]
        base_name = base_name.replace("_DSM", "")
        dtm_path = dtm_folder / f"{base_name}_DTM.tif"
        chm_output_path = chm_folder / f"{base_name}_CHM.tif"
        
        if not dtm_path.exists():
            print(f"Skipping {base_name}: Corresponding DTM not found.")
            continue
            
        try:
            with rasterio.open(dsm_path) as dsm_src, rasterio.open(dtm_path) as dtm_src:
                dsm = dsm_src.read(1)
                dtm = dtm_src.read(1)
                
                if dsm.shape != dtm.shape:
                    print(f"Skipping {base_name}: DSM and DTM raster sizes do not match.")
                    continue
                    
                chm = dsm - dtm
                chm[dsm == dsm_src.nodata] = dsm_src.nodata
                
                chm_meta = dsm_src.meta.copy()
                chm_meta.update(dtype=rasterio.float32)
                
                with rasterio.open(chm_output_path, "w", **chm_meta) as chm_dst:
                    chm_dst.write(chm.astype(rasterio.float32), 1)
                    
        except Exception as e:
            print(f"Error processing {base_name}: {e}")

def process_all(config):
    """Main processing function."""
    processor = PointCloudProcessor(config)
    start_time = time.time()
    
    if config.create_DSM:
        print("\n========== Starting DSM Generation ==========")
        processor.generate_dsm(
            input_folder=str(config.preprocessed_dir),
            output_folder=str(config.results_dir),
            run_name=config.run_name,
            resolution=config.resolution,
            chunk_size=config.chunk_size,
            num_workers=config.num_workers,
            fill_gaps=config.fill_gaps,
            method=config.point_density_method
        )
    
    if config.create_DEM:
        print("\n========== Starting DEM Generation ==========")
        processor.generate_dtm(
            input_folder=str(config.preprocessed_dir),
            output_folder=str(config.results_dir),
            run_name=config.run_name,
            resolution=config.resolution,
            chunk_size=config.chunk_size,
            num_workers=config.num_workers,
            fill_gaps=config.fill_gaps,
            method=config.point_density_method,
            rigidness=config.rigidness,
            iterations=config.iterations
        )
    
    if config.create_CHM:
        print("\n========== Starting CHM Generation ==========")
        generate_chm(
            input_folder=str(config.results_dir),
            output_folder=str(config.results_dir),
            run_name=config.run_name
        )
    
    processor.cleanup()
    elapsed_time = timedelta(seconds=int(time.time() - start_time))
    print(f"\nProcessing completed in {elapsed_time}")