import os
import warnings
import numpy as np
from osgeo import gdal


class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass


class Configuration:
    """ Configuration of all parameters used for DEM/DSM creation """

    def __init__(self):

        # --------- RUN NAME ---------
        self.run_name = 'Lidar_test'  # Custom name for this run

        # ---------- PATHS -----------
        # Input data paths
        self.target_area_dir = '/isipd/projects/p_planetdw/data/lidar/target_areas'  # Path to vector footprints of target areas
        self.las_footprints_dir = '/isipd/projects/p_planetdw/data/lidar/las_footprints'  # Path to footprints of flight paths
        self.las_files_dir = '/isipd/projects/p_planetdw/data/lidar/las_pointclouds'  # Path to lidar point clouds (*.las/*.laz)
        self.validation_dir = '/isipd/projects/p_planetdw/data/lidar/validation'  # Path to validation data

        # Output directories
        self.preprocessed_dir = '/isipd/projects/p_planetdw/data/lidar/preprocessed'  # Path for preprocessed lidar data
        self.results_dir = '/isipd/projects/p_planetdw/data/lidar/results'  # Path for final DEM/DSM results

        # ------ PREPROCESSING ------

        self.multiple_targets = False  # If target areas are saved in one gdf set to True
        self.target_name_field = 'name'  # Field in target area gdf to use as target name

        # SOR parameters
        self.overlap = 0.2  # minimum overlap between pointcloud and AOI, 0.5 means 50% overlap
        self.knn = 100  # number of k nearest neighbors, the higher the more stable
        self.multiplier = 2.2 # Threshold for outlier removal: points beyond (global_mean + multiplier * stddev) are removed.

        # ------- PROCESSING --------

        self.create_DSM = True
        self.create_DEM = True
        self.create_CHM = False

        self.fill_gaps = True # use IDW to close gaps in rasters
        self.resolution = 1 # resoltion of generated rasters in meter, can be 'Auto' or number

        self.point_density_method = 'sampling' # method to determine point density, can be 'sampling' (exact) or 'density' (fast)
        self.rigidness = 2 # rigidness of the simulated cloth, the lower the more flexible
        self.iterations = 1000 # number of simulation steps, the higher, the more adapted to the point cloud

        # ------ VALIDATION ------

        self.data_type = 'raster'   # Type of validation data, can be 'raster' or 'vector'
        self.validation_target = 'DSM' # product to validate, can be 'DSM', 'DEM' or 'CHM', select validation data accordingly! (DSM: higest point, DEM: ground level, CHM: height of vegetation)
        self.val_column_point = 'val_value' # column in point validation data to use for comparison
        self.val_band_raster = 1
        self.sample_size = 100 # number of points to sample for validation


        # ------ ADVANCED SETTINGS ------
        self.chunk_size = 100 # Number of points to process in each chunk
        self.num_workers = 4  # Number of parallel workers for processing

        # Set overall GDAL settings
        gdal.UseExceptions()  # Enable exceptions instead of silent failures
        gdal.SetCacheMax(32000000000)  # Set cache size in KB for GDAL operations
        warnings.filterwarnings('ignore')  # Suppress warnings

    def validate(self):
        """Validate config to catch errors early, and not during or at the end of processing"""

        # Check that required input paths exist
        for path_attr in ["target_area_dir", "las_footprints_dir", "las_files_dir"]:
            path = getattr(self, path_attr)
            if not os.path.exists(path):
                raise ConfigError(f"Invalid path: {path_attr} = {path}")

        # Create required output directories if they don't exist
        for path_attr in ["results_dir", "preprocessed_dir"]:
            path = getattr(self, path_attr)
            try:
                os.makedirs(path, exist_ok=True)
            except OSError:
                raise ConfigError(f"Unable to create folder: {path_attr} = {path}")

        return self
