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
        self.run_name = 'newtestbiiig'  # Custom name for this run

        # ---------- PATHS -----------
        # Input data paths
        self.target_area_dir = '/isipd/projects/p_planetdw/data/lidar/01_target_areas'  # Path to vector footprints of target areas
        self.las_files_dir = '/isipd/projects/p_planetdw/data/lidar/02_pointclouds'  # Path to lidar point clouds (*.las/*.laz)
        self.las_footprints_dir = '/isipd/projects/p_planetdw/data/lidar/03_las_footprints'  # Path to footprints of flight paths, if not available will be generated

        # Output directories
        self.preprocessed_dir = '/isipd/projects/p_planetdw/data/lidar/04_preprocessed'  # Path for preprocessed lidar data
        self.results_dir = '/isipd/projects/p_planetdw/data/lidar/05_results'  # Path for final DEM/DSM results
        self.validation_dir = '/isipd/projects/p_planetdw/data/lidar/06_validation'  # Path to validation data

        # ------ PREPROCESSING ------

        self.multiple_targets = False  # If target areas are saved in one gdf set to True
        self.target_name_field = 'target_area'  # Field in target area gdf to use as target name

        self.max_elevation_threshold = 200 # threshold to remove outliers from MTA/atmosphere, when outside of median elevation +/- threshold, the point is removed. 

        # SOR parameters
        self.knn = 100  # number of k nearest neighbors, the higher the more stable
        self.multiplier = 2.2 # Threshold for outlier removal: points beyond (global_mean + multiplier * stddev) are removed.

        # ------- PROCESSING --------

        self.create_DSM = True
        self.create_DEM = True
        self.create_CHM = True

        self.fill_gaps = True # use IDW to close gaps in rasters
        self.resolution = 0.2 # resoltion of generated rasters in meter, can be 'Auto' or number

        self.point_density_method = 'sampling' # method to determine point density, can be 'sampling' (exact) or 'density' (fast)

        self.rigidness = 2 # rigidness of the simulated cloth, the lower the more flexible
        self.iterations = 500 # number of simulation steps, the higher, the more adapted to the point cloud
        self.time_step = 1 # time step of the simulation, the lower the more accurate, but slower
        self.cloth_resolution = 1 # resolution of the cloth (m), the lower the more accurate, but slower

        # ------ VALIDATION ------

        self.data_type = 'raster'   # Type of validation data, can be 'raster' or 'vector' (points)
        self.validation_target = 'DSM' # product to validate, can be 'DSM', 'DEM' or 'CHM', select validation data accordingly! (DSM: higest point, DEM: ground level, CHM: height of vegetation)
        self.val_column_point = 'val_value' # column in point validation data to use for comparison
        self.val_band_raster = 1
        self.sample_size = 100 # number of points to sample for validation


        # ------ ADVANCED SETTINGS ------

        # _______ Preprocessing _______
        self.overlap = 0.2  # minimum overlap between pointcloud and AOI, 0.5 means 50% overlap

        self.filter_date = False  # Filter las files by date
        self.start_date = '2023-07-01'  # Start date for filtering las files
        self.end_date = '2023-07-30'  # End date for filtering las files

        # _______ Processing _______
        self.chunk_size = 1000 # chunk in meters
        self.chunk_overlap = 0.2 # overlap between chunks in percentage, 0.2 means 20% overlap
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
