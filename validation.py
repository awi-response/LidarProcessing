import rasterio
import numpy as np
import os
import geopandas as gpd
from typing import Union, Optional

def raster_to_points(val_dir: str, val_band: int, num_points: int) -> gpd.GeoDataFrame:
    """
    Convert raster files to points with associated raster values.
    Handles multiple UTM zones by transforming all geometries to WGS84.
    
    Args:
        val_dir (str): Directory containing raster files
        val_band (int): Band number to sample from
        num_points (int): Total number of points to generate across all rasters
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing points and their raster values
    """
    final_gdf = gpd.GeoDataFrame()
    raster_files = [f for f in os.listdir(val_dir) if f.endswith(('.tif', '.jp2'))]
    points_per_raster = num_points // len(raster_files)
    
    for raster_file in raster_files:
        full_path = os.path.join(val_dir, raster_file)
        print(f'Reading raster: {raster_file}')
        try:
            with rasterio.open(full_path) as src:
                val_band_array = src.read(val_band)
                raster_bounds = src.bounds
                raster_crs = src.crs
                no_data = src.nodata
                
                # Generate points within bounds
                x_coords = np.random.uniform(raster_bounds[0], raster_bounds[2], points_per_raster)
                y_coords = np.random.uniform(raster_bounds[1], raster_bounds[3], points_per_raster)
                points = np.column_stack((x_coords, y_coords))
                
                # Create GeoDataFrame with source CRS
                gdf = gpd.GeoDataFrame(
                    geometry=gpd.points_from_xy(points[:, 0], points[:, 1]),
                    crs=raster_crs
                )
                
                # Sample raster values
                gdf['val_value'] = _sample_raster_at_points(gdf, val_band_array, src.transform)
                gdf = gdf[gdf['val_value'] != no_data]
                
                # Transform to WGS84 for consistent CRS
                gdf = gdf.to_crs("EPSG:4326")
                final_gdf = gpd.pd.concat([final_gdf, gdf], ignore_index=True)
                
                # Save intermediate results
                final_gdf.to_file(os.path.join(val_dir, 'validation_points.shp'))
                
        except Exception as e:
            print(f'Error processing {raster_file}: {str(e)}')
            continue
            
    return final_gdf

def _sample_raster_at_points(gdf: gpd.GeoDataFrame,
                           raster_array: np.ndarray,
                           transform: rasterio.Affine) -> np.ndarray:
    """
    Helper function to sample raster values at point locations.
    
    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing points
        raster_array (np.ndarray): Raster array to sample from
        transform (rasterio.Affine): Raster transformation matrix
        
    Returns:
        np.ndarray: Array of sampled values
    """
    # Convert points to raster coordinates
    rows, cols = rasterio.transform.rowcol(transform,
                                         gdf.geometry.x,
                                         gdf.geometry.y)
    
    # Ensure coordinates are within bounds
    valid_mask = (rows >= 0) & (rows < raster_array.shape[0]) & \
                 (cols >= 0) & (cols < raster_array.shape[1])
    
    # Sample values
    values = np.full(len(gdf), np.nan)
    values[valid_mask] = raster_array[rows[valid_mask], cols[valid_mask]]
    return values

def get_dem_value(processed_dir: str, val_gdf: gpd.GeoDataFrame,
                 val_column: str, run_name: str) -> Optional[gpd.GeoDataFrame]:
    """
    Sample DEM values at point locations and add them to the GeoDataFrame.
    
    Args:
        processed_dir (str): Directory containing DEM files
        val_gdf (gpd.GeoDataFrame): GeoDataFrame with validation points
        val_column (str): Name of the column to store DEM values
        run_name (str): Name of the run directory
        
    Returns:
        gpd.GeoDataFrame: Updated GeoDataFrame with DEM values
    """
    dem_files_path = os.path.join(processed_dir, run_name)
    
    if not os.path.exists(dem_files_path):
        print(f"Error: Directory does not exist: {dem_files_path}")
        return None
    
    dem_files = [f for f in os.listdir(dem_files_path) if f.endswith('.tif')]
    
    if not dem_files:
        print(f"No DEM files found in {dem_files_path}")
        return None
    
    # Create dem_values column upfront
    val_gdf['dem_values'] = np.nan
    
    # Track processed files
    processed_files = []
    
    for dem in dem_files:
        full_path = os.path.join(dem_files_path, dem)
        print(f'Reading DEM: {full_path}')
        
        try:
            with rasterio.open(full_path) as src:
                dem_band = src.read(1)
                dem_bounds = src.bounds
                dem_crs = src.crs
                no_data = src.nodata
                
                # Sample values
                temp_values = _sample_raster_at_points(val_gdf, dem_band, src.transform)
                
                # Replace NaN values in dem_values with valid DEM values
                mask = np.isnan(val_gdf['dem_values'])
                val_gdf.loc[mask, 'dem_values'] = temp_values[mask]
                
                processed_files.append(dem)
                
        except Exception as e:
            print(f'Error processing {dem}: {str(e)}')
            continue
            
    if not processed_files:
        print(f"No valid DEM files processed in {run_name}")
        return None
        
    print("Processed DEM files:", ", ".join(processed_files))
    return val_gdf

def validate_all(conf):
    """
    Main validation function that coordinates raster sampling and DEM value extraction.
    
    Args:
        conf: Configuration object containing validation parameters
    """
    global config
    config = conf
    
    if config.data_type == 'raster':
        output = raster_to_points(config.validation_dir, config.val_band_raster, config.sample_size)
        print(output['val_value'].describe())
        
        result = get_dem_value(config.results_dir, output, config.val_column_point, config.run_name)
        if result is None:
            print("Warning: DEM value processing failed")
            return
        
        print(result['dem_values'].describe())  # Using correct column name
        
    elif config.data_type == 'vector':
        pass

def check_columns(gdf):
    """
    Helper function to check available columns in a GeoDataFrame.
    
    Args:
        gdf: GeoDataFrame to check
    """
    print("\nAvailable columns:")
    for col in gdf.columns:
        print(f"- {col}")
    print(f"\nTotal columns: {len(gdf.columns)}")