import os
import shutil
import geopandas as gpd
import pandas as pd

import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from weasyprint import HTML
from typing import Optional, Dict, Tuple
from io import BytesIO
import base64


def cleanup_temp_dir(temp_dir):
    """Delete the temporary directory after processing."""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Failed to remove temp directory {temp_dir}: {e}")

def split_gpkg(gdf_path, out_dir, field_name='name'):
    """Splits a GeoPackage into separate files based on a field name.
    
    - If the field value is missing, it assigns a unique number as the filename.
    - Deletes the input GPKG after splitting.
    - Returns the path to the output directory.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Read the input GPKG
    gdf = gpd.read_file(gdf_path)

    # Ensure the field exists
    if field_name not in gdf.columns:
        raise ValueError(f"Field '{field_name}' not found in the GeoPackage.")

    # Get unique values, replacing missing ones with an empty string
    unique_values = gdf[field_name].fillna("").astype(str).unique()

    for i, value in enumerate(unique_values):
        # Use a number if the field value is empty
        if value.strip() == "":
            safe_value = f"unnamed_{i}"
        else:
            # Make sure filename is safe
            safe_value = value.replace(" ", "_").replace("/", "_").replace("\\", "_")

        # Define output path
        out_path = os.path.join(out_dir, f"{safe_value}.gpkg")

        # Save subset of the data
        gdf[gdf[field_name] == value].to_file(out_path, driver="GPKG")

        print(f"Saved: {out_path}")

    # Delete the original input file
    os.remove(gdf_path)
    print(f"Deleted input file: {gdf_path}")

    # Return the output folder path
    return os.path.abspath(out_dir)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
})

def compute_error_metrics(
    gdf: gpd.GeoDataFrame,
    reference_col: str,
    prediction_col: str,
    plot: bool = True,
    save_path: Optional[str] = None
) -> Tuple[Dict, Optional[plt.Figure]]:
    """
    Compute validation metrics for reference vs predicted values.
    
    Args:
        gdf: GeoDataFrame containing reference and prediction data
        reference_col: Name of column containing reference values
        prediction_col: Name of column containing predicted values
        plot: Whether to generate plots (False for markdown generation)
        save_path: Path to save figures (None for markdown generation)
        
    Returns:
        Tuple of (metrics dictionary, figure object)
    """
    df = gdf[[reference_col, prediction_col, 'raster_name']].dropna()
    
    
    if df.empty:
        print("No valid validation data found.")
        return {"global": {}, "per_raster": {}}, None
    
    def compute_stats(subset):
        """Compute statistical metrics for a dataset subset."""
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
    
    # Compute global statistics
    global_stats = compute_stats(df)
    
    # Compute per-raster statistics
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
            
            # Create scatter plot
            ax.scatter(subset[reference_col], subset[prediction_col], alpha=0.6, edgecolor='k', linewidth=0.3)
            
            # Add 1:1 line
            min_val = min(subset[reference_col].min(), subset[prediction_col].min())
            max_val = max(subset[reference_col].max(), subset[prediction_col].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
            
            # Customize plot
            ax.set_title(raster_name, fontsize=10)
            ax.set_xlabel('Reference Data')
            ax.set_ylabel('Modelled Data')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.axis('equal')
        
        # Remove empty subplots
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

def generate_validation_report(gdf, reference_col, prediction_col, output_path):
    """
    Generate a validation report using WeasyPrint.
    
    Args:
        gdf: GeoDataFrame containing validation data
        reference_col: Name of column with reference values
        prediction_col: Name of column with predicted values
        output_path: Path where the PDF report will be saved
    """
    # Compute metrics
    metrics, _ = compute_error_metrics(gdf, reference_col, prediction_col, plot=False)
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 2cm; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; }}
            table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .plot {{ text-align: center; margin: 1em 0; }}
            .plot img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>Validation Report</h1>
        <h2>Summary Statistics</h2>
    """
    
    # Add global metrics table
    global_df = pd.DataFrame.from_dict(metrics['global'], orient='index', columns=['Value'])
    html_content += global_df.to_html(classes='metric-table')
    
    # Add global plot
    fig, ax = plt.subplots(figsize=(10, 6))
    global_data = gdf[[reference_col, prediction_col]].dropna()
    ax.scatter(global_data[reference_col], global_data[prediction_col], alpha=0.6)
    ax.plot([global_data[reference_col].min(), global_data[reference_col].max()],
            [global_data[reference_col].min(), global_data[reference_col].max()], 'r--')
    ax.set_title('Global Validation Results')
    ax.set_xlabel('Reference Data')
    ax.set_ylabel('Modelled Data')
    
    # Save plot to bytes buffer
    plot_buffer = BytesIO()
    fig.savefig(plot_buffer, format='png', bbox_inches='tight', dpi=300)
    plot_buffer.seek(0)
    encoded_image = base64.b64encode(plot_buffer.getvalue()).decode('utf-8')
    html_content += f'<div class="plot"><img src="data:image/png;base64,{encoded_image}"></div>'
    
    # Add per-raster sections
    for raster_name in gdf['raster_name'].unique():
        html_content += f'<h2>{raster_name}</h2>'
        
        # Add raster-specific metrics
        raster_df = pd.DataFrame.from_dict(metrics['per_raster'][raster_name], 
                                         orient='index', columns=['Value'])
        html_content += raster_df.to_html(classes='metric-table')
        
        # Add scatter plot
        raster_data = gdf[gdf['raster_name'] == raster_name]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(raster_data[reference_col], raster_data[prediction_col], alpha=0.6)
        ax.plot([raster_data[reference_col].min(), raster_data[reference_col].max()],
                [raster_data[reference_col].min(), raster_data[reference_col].max()], 'r--')
        ax.set_title(f'{raster_name} Validation Results')
        ax.set_xlabel('Reference Data')
        ax.set_ylabel('Modelled Data')
        
        # Save plot to bytes buffer
        plot_buffer = BytesIO()
        fig.savefig(plot_buffer, format='png', bbox_inches='tight', dpi=300)
        plot_buffer.seek(0)
        encoded_image = base64.b64encode(plot_buffer.getvalue()).decode('utf-8')
        html_content += f'<div class="plot"><img src="data:image/png;base64,{encoded_image}"></div>'
    
    html_content += """
    </body>
    </html>
    """
    
    # Generate PDF
    HTML(string=html_content).write_pdf(output_path)