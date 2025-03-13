import os
import shutil
import geopandas as gpd
import pandas as pd

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