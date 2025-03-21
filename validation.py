import rasterio
import numpy as np
import os
import geopandas as gpd
from typing import Union, Optional
import numpy as np
import seaborn as sns
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

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

                final_gdf.to_file(os.path.join(val_dir, 'validation_points.shp'))
                
        except Exception as e:
            print(f'Error processing {raster_file}: {str(e)}')
            continue
            
    return final_gdf

def _sample_raster_at_points(gdf: gpd.GeoDataFrame, raster_array: np.ndarray, transform: rasterio.Affine) -> np.ndarray:
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

def get_dem_value(preprocessed_dir: str, val_gdf: gpd.GeoDataFrame, run_name: str) -> gpd.GeoDataFrame:
    """
    Sample DEM values from raster files and add 'dem_value' and 'raster_name' to the GeoDataFrame.
    """
    combined_gdf = gpd.GeoDataFrame()
    base_dir = os.path.join(preprocessed_dir, run_name)
    raster_files = [f for f in os.listdir(base_dir) if f.endswith(('.tif', '.jp2'))]

    for raster_file in raster_files:
        full_path = os.path.join(base_dir, raster_file)
        print(f'Reading raster: {raster_file}')
        
        try:
            with rasterio.open(full_path) as src:
                val_band_array = src.read(1)
                raster_crs = src.crs
                no_data = src.nodata

                # Copy input and transform CRS
                gdf = val_gdf.copy()
                gdf = gdf.to_crs(raster_crs)

                # Sample values and filter
                gdf['dem_value'] = _sample_raster_at_points(gdf, val_band_array, src.transform)
                gdf = gdf[gdf['dem_value'] != no_data]

                # Add raster name column BEFORE concat
                gdf['raster_name'] = raster_file

                # Reproject back to WGS84
                gdf = gdf.to_crs("EPSG:4326")

                # Append to final combined result
                combined_gdf = pd.concat([combined_gdf, gdf], ignore_index=True)

        except Exception as e:
            print(f'Error processing {raster_file}: {str(e)}')
            continue

    # Save output if needed
    combined_gdf.to_file(os.path.join(base_dir, 'validation_points.shp'))
    return combined_gdf

def compute_error_metrics(
    gdf: gpd.GeoDataFrame,
    reference_col: str,
    prediction_col: str,
    plot: bool = True,
    save_path: Optional[str] = None
) -> dict:
    """
    Compute RMSE, NMAD, MR, and STDE between two columns in a GeoDataFrame.
    Optionally show a residual plot.

    Args:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame.
        reference_col (str): Ground truth column.
        prediction_col (str): Predicted/estimated column.
        plot (bool): Whether to display a residual plot.

    Returns:
        dict: Dictionary with RMSE, NMAD, MR, STDE.
    """
    # Clean data
    df = gdf[[reference_col, prediction_col, 'raster_name']].dropna()
    residuals = df[prediction_col] - df[reference_col]

    # Compute metrics
    rmse = np.sqrt(np.mean(residuals ** 2))
    nmad = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
    mr = np.mean(residuals)
    stde = np.std(residuals)

    if plot:
        plt.figure(figsize=(6, 6))

        # Get unique colors for each raster
        unique_rasters = df['raster_name'].unique()
        palette = sns.color_palette("tab10", len(unique_rasters))  # or any other palette

        for i, raster_name in enumerate(unique_rasters):
            subset = df[df['raster_name'] == raster_name]
            plt.scatter(
                subset[reference_col],
                subset[prediction_col],
                alpha=0.5,
                edgecolor='k',
                linewidth=0.3,
                label=raster_name,
                color=palette[i]
            )

        # 1:1 Line
        min_val = min(df[reference_col].min(), df[prediction_col].min())
        max_val = max(df[reference_col].max(), df[prediction_col].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')

        plt.xlabel(f'{reference_col} (Reference)')
        plt.ylabel(f'{prediction_col} (Predicted)')
        plt.title('Predicted vs. Reference (Colored by Raster)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.axis('equal')
        plt.legend(loc='best', fontsize='small')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)

    return {
        "RMSE": rmse,
        "NMAD": nmad,
        "MR": mr,
        "STDE": stde
    }




def validate_all(conf):
    global config
    config = conf
    
    # Initialize output variable
    output = None
    
    if config.data_type == 'raster':
        output = raster_to_points(config.validation_dir, config.val_band_raster, config.sample_size)
        config.val_column_point = 'val_value'
        print('finished conversion')
    
    # Only proceed if output exists and has the expected column
    combined = get_dem_value(config.results_dir, output, config.run_name)

    metrics = compute_error_metrics(combined, config.val_column_point, 'dem_value', save_path='/isipd/projects/p_planetdw/data/lidar/validation/val_plot.png')
    print(metrics)
    
    return output