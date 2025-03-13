import pdal
import json
import laspy
from multiprocessing import Pool
from shapely.geometry import box, shape
import os
from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps
from tqdm import tqdm

from core.reprojection import get_utm_epsg, reproject_las, is_utm_crs

def create_chunks_from_wkt(target_geom_wkt, chunk_size=100):
    """Create grid chunks based on the bounding box of the target geometry."""
    target_geom = wkt_loads(target_geom_wkt)
    min_x, min_y, max_x, max_y = target_geom.bounds
    
    chunks = []
    for x in range(int(min_x), int(max_x), chunk_size):
        for y in range(int(min_y), int(max_y), chunk_size):
            chunk_bbox = box(x, y, x + chunk_size, y + chunk_size)
            if target_geom.intersects(chunk_bbox):
                chunks.append(chunk_bbox)
    
    return chunks

def process_chunk(input_file, chunk_bbox, temp_dir, sor_knn=8, sor_multiplier=2.0, ref_scale=None, ref_offset=None, ref_crs=None):
    """Process a chunk with SOR filtering and save it to a temp folder."""
    chunk_file = os.path.join(temp_dir, f"{os.path.basename(input_file).replace('.las', '')}_chunk_{int(chunk_bbox.bounds[0])}_{int(chunk_bbox.bounds[1])}.las")
    
    pipeline = [
        {"type": "readers.las", "filename": input_file},
        {"type": "filters.crop", "polygon": wkt_dumps(chunk_bbox)},
        {"type": "filters.outlier", "method": "statistical", "mean_k": sor_knn, "multiplier": sor_multiplier},
        {"type": "filters.range", "limits": "Classification![7:7]"},
        {"type": "writers.las",
            "filename": chunk_file,
            "scale_x": str(ref_scale[0]), "scale_y": str(ref_scale[1]), "scale_z": str(ref_scale[2]),
            "offset_x": str(ref_offset[0]), "offset_y": str(ref_offset[1]), "offset_z": str(ref_offset[2]),
            "a_srs": f"EPSG:{ref_crs}" if ref_crs else None
        }
    ]
    
    try:
        pdal.pipeline.Pipeline(json.dumps(pipeline)).execute()
        return chunk_file
    except Exception as e:
        print(f"Error processing chunk {chunk_file}: {e}")
        return None

def merge_and_crop_chunks(chunk_files, target_geom_wkt, output_file):
    """Merge processed chunks and crop them to the target geometry."""
    target_geom = wkt_loads(target_geom_wkt)
    
    pipeline = [{"type": "readers.las", "filename": f} for f in chunk_files]
    pipeline.append({"type": "filters.merge"})
    pipeline.append({"type": "filters.crop", "polygon": wkt_dumps(target_geom)})
    pipeline.append({"type": "writers.las", "filename": output_file})
    
    try:
        pdal.pipeline.Pipeline(json.dumps(pipeline)).execute()
        return output_file
    except Exception as e:
        print(f"Error merging and cropping: {e}")
        return None

def process_las_files(las_dict, preprocessed_dir, num_workers=4, chunk_size=100, sor_knn=8, sor_multiplier=2.0):
    """Process multiple LAS files for different target areas."""
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    for target_area, las_files in las_dict.items():
        temp_dir = os.path.join(preprocessed_dir, target_area, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        gdf, input_files = las_files['gdf'], las_files['files']
        target_geom_wkt = wkt_dumps(shape(gdf.geometry.iloc[0]))
        chunks = create_chunks_from_wkt(target_geom_wkt, chunk_size)
        
        process_args = []
        for input_file in input_files:
            ref_scale, ref_offset, ref_crs = get_las_header(input_file)
            for chunk in chunks:
                process_args.append((input_file, chunk, temp_dir, sor_knn, sor_multiplier, ref_scale, ref_offset, ref_crs))
        
        processed_chunks = []
        with tqdm(total=len(process_args), desc=f"Processing {target_area}", unit="chunk") as pbar:
            with Pool(num_workers) as pool:
                for processed_chunk in pool.imap_unordered(lambda args: process_chunk(*args), process_args):
                    if processed_chunk:
                        processed_chunks.append(processed_chunk)
                    pbar.update(1)
        
        if processed_chunks:
            final_output_file = os.path.join(preprocessed_dir, f"{target_area}_processed.las")
            merge_and_crop_chunks(processed_chunks, target_geom_wkt, final_output_file)
        else:
            print(f"No processed chunks available for {target_area}.")

def get_las_header(las_file):
    """Extracts scale, offset, and CRS from an input LAS file."""
    with laspy.open(las_file) as las:
        header = las.header
        scale = header.scales
        offset = header.offsets
        crs = header.parse_crs()
        crs_epsg = crs.to_epsg() if crs else 4979
    return scale, offset, crs_epsg
