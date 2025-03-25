import os
import shutil
import geopandas as gpd
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import table as tbl


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

def save_validation_report(metrics: dict, plot_path: str, save_path: str):
    """
    Save error metrics and plots into a single PDF report.
    
    Args:
        metrics (dict): Output from compute_error_metrics
        plot_path (str): Path to the saved comparison plot
        save_path (str): Path to the PDF report
    """
    with PdfPages(save_path) as pdf:
        # Page 1: Title and Global Metrics
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.set_title('Validation Report', fontsize=16, weight='bold', pad=20)

        y_start = 0.9
        line_space = 0.05
        ax.text(0.05, y_start, "Global Error Metrics", fontsize=12, weight='bold')

        for idx, (key, val) in enumerate(metrics["global"].items()):
            ax.text(0.07, y_start - (idx+1)*line_space, f"{key}: {val:.3f}", fontsize=10)

        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Per-Raster Table
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.set_title('Per-Raster Error Metrics', fontsize=14, weight='bold')

        table_data = []
        headers = ["Raster"] + list(next(iter(metrics["per_raster"].values())).keys())

        for raster_name, stat_dict in metrics["per_raster"].items():
            row = [raster_name] + [f"{v:.3f}" for v in stat_dict.values()]
            table_data.append(row)

        table = tbl.table(ax, cellText=table_data, colLabels=headers, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Scatter Plot Comparisons
        if os.path.exists(plot_path):
            img = plt.imread(plot_path)
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            ax.imshow(img)
            pdf.savefig(fig)
            plt.close(fig)