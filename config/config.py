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
        self.run_name = 'FireData'  # Custom name for this run

        # ---------- PATHS -----------
        # Input data paths
        self.target_area_dir = '/isipd/projects/p_planetdw/data/lidar/01_target_areas'  # Path to vector footprints of target areas
        self.las_files_dir = '/isipd/projects/p_planetdw/data/lidar/02_pointclouds/2024'  # Path to lidar point clouds (*.las/*.laz)
        self.las_footprints_dir = '/isipd/projects/p_planetdw/data/lidar/03_las_footprints/2024'  # Path to footprints of flight paths, if not available will be generated

        # Output directories
        self.preprocessed_dir = '/isipd/projects/p_planetdw/data/lidar/04_preprocessed'  # Path for preprocessed lidar data
        self.results_dir = '/isipd/projects/p_planetdw/data/lidar/05_results'  # Path for final DEM/DSM results
        self.validation_dir = '/isipd/projects/p_planetdw/data/lidar/06_validation'  # Path to validation data

        # ------ PREPROCESSING ------

        self.multiple_targets = True  # If target areas are saved in one gdf set to True
        self.target_name_field = 'id'  # Field in target area gdf to use as target name

        self.max_elevation_threshold = 0.99 # quantile to disgard atmospheric noise etc. Data outside the quantile is disgarded. 

        # SOR parameters
        self.knn = 100  # number of k nearest neighbors, the higher the more stable
        self.multiplier = 2 # Threshold for outlier removal: points beyond (global_mean + multiplier * stddev) are removed.

        # ------- PROCESSING --------

        self.create_DSM = True
        self.create_DEM = False
        self.create_CHM = False

        self.fill_gaps = True # use IDW to close gaps in rasters
        self.resolution = 1 # resoltion of generated rasters in meter, can be 'Auto' or number

        self.point_density_method = 'sampling' # method to determine point density, can be 'sampling' (exact) or 'density' (fast)

    # ______ GROUND FILTERING ______

        self.smrf_filter = True # use SMRF filter 
        self.csf_filter = True # use cloth simulation method
        self.threshold = 2 # threshold for filters

        self.smrf_window_size = 20 # window size for SMRF filter, the higher the more vegetation is removed
        self.smrf_slope = 0.2 # slope for SMRF filter, the higher the more vegetation is removed
        self.smrf_scalar = 2 # scalar for SMRF filter, the higher the more vegetation is removed

        self.csf_rigidness = 3 # rigidness of the simulated cloth, the lower the more flexible, use low values for steep and high for flat terrain
        self.csf_iterations = 500 # number of simulation steps, the higher, the more adapted to the point cloud
        self.csf_time_step = 0.5 # time step of the simulation, the lower the more accurate, but slower
        self.csf_cloth_resolution = 1 # resolution of the cloth (m), the lower the more accurate, but slower

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
        self.start_date = '2023-07-22'  # Start date for filtering las files
        self.end_date = '2023-07-20'  # End date for filtering las files

        # _______ Processing _______
        self.chunk_size = 500 # chunk in meters
        self.chunk_overlap = 0.1 # overlap between chunks in percentage, 0.2 means 20% overlap
        self.num_workers = 16  # Number of parallel workers for processing

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
