import os
import time
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm

from core.utils import save_validation_report


def raster_to_points(val_dir: str, val_band: int, num_points: int, target_dir: str) -> gpd.GeoDataFrame:
    final_gdf = gpd.GeoDataFrame()
    raster_files = [f for f in os.listdir(val_dir) if f.endswith(('.tif', '.jp2'))]
    target_areas = [f for f in os.listdir(target_dir) if f.endswith(('.shp', '.gpkg'))]

    for target_area in target_areas:
        full_path = os.path.join(target_dir, target_area)
        print(f"\nReading target area: {target_area}")

        try:
            target_gdf = gpd.read_file(full_path)
        except Exception as e:
            print(f"Error reading {target_area}: {e}")
            continue

        bounds = target_gdf.total_bounds
        crs = target_gdf.crs

        x_coords = np.random.uniform(bounds[0], bounds[2], num_points * 10)
        y_coords = np.random.uniform(bounds[1], bounds[3], num_points * 10)
        points = np.column_stack((x_coords, y_coords))

        point_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(points[:, 0], points[:, 1]),
            crs=crs
        )

        for raster_file in raster_files:
            raster_path = os.path.join(val_dir, raster_file)
            print(f"Sampling raster: {raster_file} for {target_area}")

            try:
                with rasterio.open(raster_path) as src:
                    val_band_array = src.read(val_band)
                    no_data = src.nodata

                    sampled_gdf = point_gdf.copy().to_crs(src.crs)
                    sampled_gdf['val_value'] = _sample_raster_at_points(sampled_gdf, val_band_array, src.transform)

                    sampled_gdf = sampled_gdf[
                        (sampled_gdf['val_value'] != no_data) &
                        (sampled_gdf['val_value'].notna())
                    ]

                    sampled_gdf = sampled_gdf.to_crs("EPSG:4326").reset_index(drop=True)
                    sampled_gdf['val_raster_name'] = raster_file
                    sampled_gdf['target_area'] = target_area.replace('.gpkg', '').replace('.shp', '')

                    final_gdf = pd.concat([final_gdf, sampled_gdf], ignore_index=True)

            except Exception as e:
                print(f"Error sampling raster {raster_file}: {e}")
                continue

    print(f"\nFinal number of validation points: {len(final_gdf)}")
    return final_gdf


def _sample_raster_at_points(gdf: gpd.GeoDataFrame, raster_array: np.ndarray, transform: rasterio.Affine) -> np.ndarray:
    print("Sampling raster at point locations...")
    rows, cols = rowcol(transform, gdf.geometry.x, gdf.geometry.y)

    valid_mask = (rows >= 0) & (rows < raster_array.shape[0]) & \
                 (cols >= 0) & (cols < raster_array.shape[1])

    values = np.full(len(gdf), np.nan)
    values[valid_mask] = raster_array[rows[valid_mask], cols[valid_mask]]

    return values


def get_dem_value(preprocessed_dir: str, validation_target, val_gdf: gpd.GeoDataFrame, run_name: str) -> gpd.GeoDataFrame:
    combined_gdf = gpd.GeoDataFrame()
    base_dir = os.path.join(preprocessed_dir, run_name, validation_target)
    raster_files = [f for f in os.listdir(base_dir) if f.endswith(('.tif', '.jp2'))]

    print(f"\nFound DEM rasters in {base_dir}")

    for raster_file in raster_files:
        full_path = os.path.join(base_dir, raster_file)
        print(f"Reading raster: {raster_file}")

        try:
            with rasterio.open(full_path) as src:
                val_band_array = src.read(1)
                no_data = src.nodata

                gdf = val_gdf.copy().to_crs(src.crs)
                gdf['dem_value'] = _sample_raster_at_points(gdf, val_band_array, src.transform)

                gdf = gdf[
                    (gdf['dem_value'] != no_data) &
                    (gdf['dem_value'].notna())
                ]

                gdf['raster_name'] = raster_file
                gdf = gdf.to_crs("EPSG:4326")
                combined_gdf = pd.concat([combined_gdf, gdf], ignore_index=True)

        except Exception as e:
            print(f"Error processing {raster_file}: {str(e)}")
            continue

    print(f"\nTotal combined DEM-sampled points: {len(combined_gdf)}")

    return combined_gdf


def compute_error_metrics(
    gdf: gpd.GeoDataFrame,
    reference_col: str,
    prediction_col: str,
    plot: bool = True,
    save_path: Optional[str] = None
) -> tuple[dict, Optional[plt.Figure]]:
    df = gdf[[reference_col, prediction_col, 'raster_name']].dropna()
    print(f"\nComputing error metrics on {len(df)} points")

    if df.empty:
        print("No valid points for computing error metrics.")
        return {"global": {}, "per_raster": {}}, None

    def compute_stats(subset):
        res = subset[prediction_col] - subset[reference_col]
        abs_res = np.abs(res)

        return {
            "RMSE": np.sqrt(np.mean(res ** 2)),
            "MAE": np.mean(abs_res),
            "NMAD": 1.4826 * stats.median_abs_deviation(res, scale=1.0),
            "MR": stats.tmean(res),
            "STDE": stats.tstd(res),
            "Median Error": np.median(res),
            "LE90": np.percentile(abs_res, 90),
            "LE95": np.percentile(abs_res, 95),
            "Max Over": np.max(res),
            "Max Under": np.min(res),
            "R2": stats.linregress(subset[reference_col], subset[prediction_col])[2] ** 2
        }

    global_stats = compute_stats(df)

    per_raster_stats = {
        raster_name: compute_stats(df[df['raster_name'] == raster_name])
        for raster_name in df['raster_name'].unique()
    }

    fig = None
    if plot:
        unique_rasters = df['raster_name'].unique()
        cols = 3
        rows = (len(unique_rasters) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)

        for idx, raster_name in enumerate(unique_rasters):
            ax = axes[idx // cols][idx % cols]
            subset = df[df['raster_name'] == raster_name]
            ax.scatter(subset[reference_col], subset[prediction_col], alpha=0.6, edgecolor='k', linewidth=0.3)

            min_val = min(subset[reference_col].min(), subset[prediction_col].min())
            max_val = max(subset[reference_col].max(), subset[prediction_col].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')

            ax.set_title(raster_name, fontsize=10)
            ax.set_xlabel('Reference Data')
            ax.set_ylabel('Modelled Data')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.axis('equal')

        for i in range(len(unique_rasters), rows * cols):
            fig.delaxes(axes[i // cols][i % cols])

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=500)

    return {
        "global": global_stats,
        "per_raster": per_raster_stats
    }, fig



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

    if output is None or output.empty:
        print("No valid validation data found. Exiting.")
        return

    print("\n--- Sampling DEM Values ---")
    combined = get_dem_value(config.results_dir, config.validation_target, output, config.run_name)

    combined = combined.groupby('raster_name', group_keys=False).sample(n=config.sample_size, random_state=42)


    val_run_dir = os.path.join(config.validation_dir, config.run_name)
    os.makedirs(val_run_dir, exist_ok=True)
    output_path = os.path.join(val_run_dir, f'{int(time_start)}_validation_points_{config.run_name}.gpkg')
    combined.to_file(output_path)

    print(f"Saved validation points to: {output_path}")

    print("\n--- Computing Error Metrics ---")
    metrics, plot_fig = compute_error_metrics(
        combined,
        config.val_column_point,
        'dem_value',
        plot=True,
        save_path=None
    )

    report_path = os.path.join(val_run_dir, f'{int(time_start)}_validation_report_{config.run_name}.pdf')
    save_validation_report(metrics, plot_fig, report_path)
    print(f"Validation report saved to: {report_path}")
    print("Validation complete.")
    print("Total validation time:", str(timedelta(seconds=int(time.time() - time_start))))