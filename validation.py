import rasterio
import numpy as np
import os
import geopandas as gpd
from typing import Union, Optional
import numpy as np
import seaborn as sns
import pandas as pd
import geopandas as gpd
from scipy import stats
import shapely
import matplotlib.pyplot as plt
from rasterio.features import shapes
from shapely.geometry import shape
from geopandas import GeoDataFrame
import time
from tqdm import tqdm
from datetime import timedelta

from core.utils import save_validation_report



def raster_to_points(val_dir: str, val_band: int, num_points: int, target_dir: str) -> gpd.GeoDataFrame:

    final_gdf = gpd.GeoDataFrame()
    raster_files = [f for f in os.listdir(val_dir) if f.endswith(('.tif', '.jp2'))]
    target_areas = [f for f in os.listdir(target_dir) if f.endswith(('.shp', '.gpkg'))]
    number_of_targets = len(target_areas)

    for target_area in target_areas:
        full_path = os.path.join(target_dir, target_area)

        target_area_bounds = gpd.read_file(full_path).total_bounds
        target_area_crs = gpd.read_file(full_path).crs
        x_coords = np.random.uniform(target_area_bounds[0], target_area_bounds[2], num_points*10)
        y_coords = np.random.uniform(target_area_bounds[1], target_area_bounds[3], num_points*10)
        points = np.column_stack((x_coords, y_coords))

        gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(points[:, 0], points[:, 1]),
            crs= target_area_crs
        )
    
        for raster_file in raster_files:
            full_path = os.path.join(val_dir, raster_file)
            
            try:
                with rasterio.open(full_path) as src:
                    val_band_array = src.read(val_band)
                    no_data = src.nodata
                    
                    # Sample raster values
                    gdf['val_value'] = _sample_raster_at_points(gdf, val_band_array, src.transform)
                    gdf = gdf[gdf['val_value'] != no_data]
                    gdf.reset_index(drop=True, inplace=True)
                    
            except Exception as e:
                print(f'Error processing {raster_file}: {str(e)}')
                continue
            


        gdf = gdf.to_crs("EPSG:4326")
        final_gdf = gpd.pd.concat([final_gdf, gdf], ignore_index=True)

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

def get_dem_value(preprocessed_dir: str, validation_target, val_gdf: gpd.GeoDataFrame, run_name: str) -> gpd.GeoDataFrame:
    """
    Sample DEM values from raster files and add 'dem_value' and 'raster_name' to the GeoDataFrame.
    """
    combined_gdf = gpd.GeoDataFrame()
    base_dir = os.path.join(preprocessed_dir, run_name, validation_target)
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
                gdf.reset_index(drop=True)

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
    Optionally show residual plots for each raster.

    Returns:
        dict: {
            'global': {...},
            'per_raster': {raster_name: {...}, ...}
        }
    """
    df = gdf[[reference_col, prediction_col, 'raster_name']].dropna()
    print(f"Total points: {len(df)}")
    print(f"df: {df}")
    residuals = df[prediction_col] - df[reference_col]
    abs_residuals = np.abs(residuals)

    def compute_stats(subset):
        res = subset[prediction_col] - subset[reference_col]
        abs_res = np.abs(res)

        rmse = np.sqrt(np.mean(res ** 2))
        mae = np.mean(abs_res)
        mr = stats.tmean(res)
        stde = stats.tstd(res)
        median_error = np.median(res)
        nmad = 1.4826 * stats.median_abs_deviation(res, scale=1.0)
        le90 = np.percentile(abs_res, 90)
        le95 = np.percentile(abs_res, 95)
        max_over = np.max(res)
        max_under = np.min(res)
        slope, intercept, r_value, _, _ = stats.linregress(subset[reference_col], subset[prediction_col])
        r2 = r_value ** 2

        return {
            "RMSE": rmse,
            "MAE": mae,
            "NMAD": nmad,
            "MR": mr,
            "STDE": stde,
            "Median Error": median_error,
            "LE90": le90,
            "LE95": le95,
            "Max Over": max_over,
            "Max Under": max_under,
            "R2": r2
        }

    # Global stats
    global_stats = compute_stats(df)

    print("\nGlobal Error Metrics:")
    for k, v in global_stats.items():
        print(f"{k}: {v:.3f}")

    # Per-raster stats
    per_raster_stats = {}
    for raster_name in df['raster_name'].unique():
        subset = df[df['raster_name'] == raster_name]
        per_raster_stats[raster_name] = compute_stats(subset)

    # Plotting
    if plot:
        unique_rasters = df['raster_name'].unique()
        n = len(unique_rasters)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)

        for idx, raster_name in enumerate(unique_rasters):
            ax = axes[idx // cols][idx % cols]
            subset = df[df['raster_name'] == raster_name]
            ax.scatter(
                subset[reference_col],
                subset[prediction_col],
                alpha=0.6,
                edgecolor='k',
                linewidth=0.3
            )
            min_val = min(subset[reference_col].min(), subset[prediction_col].min())
            max_val = max(subset[reference_col].max(), subset[prediction_col].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')

            ax.set_title(raster_name, fontsize=10)
            ax.set_xlabel('reference data')
            ax.set_ylabel('modelled data')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.axis('equal')

        # Hide unused subplots
        for i in range(n, rows * cols):
            fig.delaxes(axes[i // cols][i % cols])

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=500)
            print(f"\nPlot saved to: {save_path}")

        plt.show()

    return {
        "global": global_stats,
        "per_raster": per_raster_stats
    }


def validate_all(conf):
    global config
    config = conf

    print("\n========== Starting Validation ==========")
    time_start = time.time()
    
    output = None

    if config.data_type == 'raster':
        print("\n--- Converting Raster to Points ---")
        output = raster_to_points(config.validation_dir, config.val_band_raster, config.sample_size, config.target_area_dir)
        config.val_column_point = 'val_value'
        print(f"Raster-to-point conversion complete. Sampled {len(output)} points.")

    elif config.data_type == 'vector':
        print("\n--- Reading Validation Vectors ---")
        vector_files = [f for f in os.listdir(config.validation_dir) if f.endswith(('.shp', '.gpkg'))]
        output = gpd.GeoDataFrame()
        
        for vector_file in tqdm(vector_files, desc="Loading vector files"):
            full_path = os.path.join(config.validation_dir, vector_file)
            try:
                gdf = gpd.read_file(full_path)
                output = pd.concat([output, gdf], ignore_index=True)
            except Exception as e:
                print(f"Error processing {vector_file}: {str(e)}")
                continue
        
        print(f"Loaded {len(vector_files)} vector files with total {len(output)} points.")

    if output is None or output.empty:
        print("No valid validation data found. Exiting.")
        return

    print("\n--- Sampling DEM Values ---")
    combined = get_dem_value(config.results_dir, config.validation_target, output, config.run_name)
    print(f"DEM sampling complete. Total sampled points: {len(combined)}")

    # Save validation GeoDataFrame
    val_run_dir = os.path.join(config.validation_dir, config.run_name)
    os.makedirs(val_run_dir, exist_ok=True)
    output_path = os.path.join(val_run_dir, f'{int(time_start)}_validation_points_{config.run_name}.gpkg')
    combined.to_file(output_path)
    print(f"Saved validation points to: {output_path}")

    print("\n--- Computing Error Metrics ---")
    metrics = compute_error_metrics(
        combined,
        config.val_column_point,
        'dem_value',
        save_path=os.path.join(val_run_dir, f'{int(time_start)}_val_plot.png')
    )

    report_path = os.path.join(val_run_dir, f'{int(time_start)}_validation_report_{config.run_name}.pdf')
    save_validation_report(metrics, 
                           plot_path=os.path.join(val_run_dir, f'{int(time_start)}_val_plot.png',), 
                           save_path=report_path)
    print(f"Validation report saved to: {report_path}")

    print("\nValidation complete.")
    print("Total validation time:", str(timedelta(seconds=int(time.time() - time_start))))
    