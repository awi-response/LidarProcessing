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
        self.run_name = 'Stud_OTBTU_Lakes'
        self.year = 2025

        # ---------- PATHS -----------
        # Input data paths
        self.target_area_dir = r'N:\isipd\projects\p_planetdw\data\lidar\01_target_areas\Stud_OTBTU_Lakes'
        self.las_files_dir = r'N:\isipd\projects\p_planetdw\data\lidar\02_pointclouds\2025\GII'
        self.las_footprints_dir = rf'N:\isipd\projects\p_planetdw\data\lidar\03_las_footprints\{self.year}\GII'

        # Output directories
        self.preprocessed_dir = r'N:\isipd\projects\p_planetdw\data\lidar\04_preprocessed'
        self.results_dir = r'N:\isipd\projects\p_planetdw\data\lidar\05_results'
        self.validation_dir = r'N:\isipd\projects\p_planetdw\data\lidar\06_validation'

        # ------ PREPROCESSING ------

        self.multiple_targets = False
        self.target_name_field = 'id'

        self.max_elevation_threshold = 1.0

        # SOR parameters (Statistical Outlier Removal)
        self.knn = 100
        self.multiplier = 1
        self.sor_passes = 3

        # ELM filter (Extended Local Minimum)
        self.elm_filter = True
        self.elm_cell = 10.0
        self.elm_threshold = 1.0

        # Radius outlier filter
        self.radius_filter = True
        self.radius_filter_radius = 1.0
        self.radius_filter_min_count = 4

        # ------- PROCESSING --------

        self.create_DSM = True
        self.create_DEM = True
        self.create_CHM = False

        self.fill_gaps = True
        self.resolution = 1

        self.point_density_method = 'density'

        # ______ GROUND FILTERING ______

        self.smrf_filter = False
        self.csf_filter = True
        self.threshold = 0.1

        # SMRF
        self.smrf_window_size = 20
        self.smrf_slope = 0.2
        self.smrf_scalar = 2

        # CSF (Cloth Simulation)
        self.csf_rigidness = 5
        self.csf_iterations = 200
        self.csf_time_step = 1
        self.csf_cloth_resolution = 2

        # ------ VALIDATION ------

        self.data_type = 'raster'
        self.validation_target = 'DSM'
        self.val_column_point = 'val_value'
        self.val_band_raster = 1
        self.sample_size = 100

        # ------ ADVANCED SETTINGS ------

        # _______ Preprocessing _______
        self.overlap = 0.1

        self.filter_date = False
        self.automatic_date_parser = True

        self.start_date = '2023-07-10'
        self.end_date = '2023-07-10'

        # _______ Processing _______
        self.chunk_size = 5000
        self.chunk_overlap = 0.1
        self.num_workers = 8

        # _______ Preprocessing _______
        self.preprocess_chunk_size = 500
        self.preprocess_chunk_overlap = 0.1
        self.preprocess_reproject_vertical = False
        self.preprocess_vertical_target_epsg = 3855

        # _______ Run Control _______
        self.run_preprocessing = True
        self.run_processing = False
        self.run_validation = False

        # Set overall GDAL settings
        gdal.UseExceptions()
        gdal.SetCacheMax(32000000000)
        warnings.filterwarnings('ignore')

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

        if self.preprocess_reproject_vertical:
            if self.preprocess_vertical_target_epsg is None:
                raise ConfigError("preprocess_vertical_target_epsg must be set when preprocess_reproject_vertical is True")
            try:
                self.preprocess_vertical_target_epsg = int(self.preprocess_vertical_target_epsg)
            except (TypeError, ValueError):
                raise ConfigError("preprocess_vertical_target_epsg must be an integer EPSG code")

        return self