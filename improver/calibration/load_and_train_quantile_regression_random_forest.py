# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to load inputs and train a model using Quantile Regression Random Forest (QRF)."""

import pathlib
import pyarrow as pa
from improver import PostProcessingPlugin
from improver.utilities.load import load_cube
import pandas as pd
import numpy as np
import iris
import pyarrow.parquet as pq
from improver.calibration.__init__ import FORECAST_SCHEMA, TRUTH_SCHEMA
from improver.calibration.quantile_regression_random_forest import TrainQuantileRegressionRandomForests
from improver.calibration.dataframe_utilities import (
        forecast_and_truth_dataframes_to_cubes,
)

class LoadAndTrainQRF(PostProcessingPlugin):

    def process(
        self,
        file_paths: pathlib.Path,
        feature_config: dict,
        target_diagnostic_name: str,
        forecast_periods: str,
        cycletime: str,
        training_length: int,
        experiment: str=None,
        n_estimators: int=100,
        max_depth: int=None,
        random_state: int=None,
        transformation: str=None,
        pre_transform_addition: float=0,
        compression: int=5,
        model_output: str=None,
        ):


        """ Loading input files and training a model using Quantile Regression Random Forest.
    
        Loads in arguments for training a Quantile Regression Random Forest (QRF)
        model which can later be applied to calibrate the forecast.
        Two sources of input data must be provided: historical forecasts and
        historical truth data (to use in calibration). The model is output as a pickle file.

        Args:
            file_paths (cli.inputpaths):
                A list of input paths containing:
                - The path to a Parquet file containing the truths to be used
                for calibration. The expected columns within the
                Parquet file are: ob_value, time, wmo_id, diagnostic, latitude,
                longitude and altitude.
                - The path to a Parquet file containing the forecasts to be used
                for calibration.
                - Optionally, paths to NetCDF files containing additional preictors.
            feature_config (dict):
                Feature configuration defining the features to be used for quantile regression.
                    The configuration is a dictionary of strings, where the keys are the names of
                    the input cube(s) supplied, and the values are a list. This list can contain both
                    computed features, such as the mean or standard deviation (std), or static
                    features, such as the altitude. The computed features will be computed using
                    the cube defined in the dictionary key. If the key is the feature itself e.g.
                    a distance to water cube, then the value should state "static". This will ensure
                    the cube's data is used as the feature.                
                    The config will have the structure: 
                    "DYNAMIC_VARIABLE_NAME": ["FEATURE1", "FEATURE2"] e.g: 
                    {
                    "air_temperature": ["mean", "std", "altitude"],
                    "visibility_at_screen_level": ["mean", "std"]
                    "distance_to_water": ["static"],
                    }
            target_diagnostic_name (str):
                A string containing the diagnostic name of the forecast to be
                calibrated. This will be used to filter the target forecast and truth
                dataframes.
            forecast_period (int):
                Range of forecast periods to be calibrated in hours in the form:
                "start:end:interval" e.g. "6:18:6".
            cycletime (str):
                Cycletime of the forecast to be calibrated in a format similar to 
                20170109T0000Z. This is used to filter the correct blendtimes from
                the dataframe on load.
            training_length (int):
                The length of the training period in days.
            experiment (str):
                The name of the experiment (step) that calibration is applied to.
            n_estimators (int):
                Number of trees in the forest.
            max_depth (int):
                Maximum depth of the tree.
            random_state (int):
                Random seed for reproducibility.
            transformation (str):
                Transformation to be applied to the data before fitting.
            pre_transform_addition (float):
                Value to be added before transformation.
            compression (int):
                Compression level for saving the model.
            model_output (str):
                Full path including model file name that will store the pickled model.
        Returns:
            None:
                The function creates a pickle file.
        """

        forecast_table_path = None
        truth_table_path = None
        cube_inputs = iris.cube.CubeList([])
        
        # file extraction loop:
        for file_path in file_paths:
            try:
                cube = load_cube(str(file_path))
                cube_inputs.append(cube)
            except IsADirectoryError:
                # For loop here because the read_schema must read a .parquet file rather than a directory.
                for file in file_path.glob("**/*.parquet"):
                    try:
                        pq.read_schema(file).field("forecast_period")
                        forecast_table_path = file_path
                    except KeyError:
                        truth_table_path = file_path
                    if forecast_table_path and truth_table_path:
                        break
        
        forecast_periods = list(range(*map(int, forecast_periods.split(":"))))
        forecast_periods = [fp * 3600 for fp in forecast_periods] 
        cycletimes = []

        for forecast_period in forecast_periods:
            # Load forecasts from parquet file filtering by diagnostic and blend_time.
            forecast_period_td = pd.Timedelta(int(forecast_period), unit="seconds")

            cycletimes.extend(pd.date_range(
               end=pd.Timestamp(cycletime)
               - pd.Timedelta(1, unit="days")
               - forecast_period_td.floor("D"),
               periods=int(training_length),
               freq="D",
           ))
        cycletimes=list(set(cycletimes))
        
        filters = [[("diagnostic", "==", target_diagnostic_name), ("blend_time", "in", cycletimes), ("experiment", "==", experiment)]]
        forecast_df = pd.read_parquet(forecast_table_path, filters=filters, schema=FORECAST_SCHEMA, engine="pyarrow")

        # Convert df columns from ms to pandas timestamp object to work with existing code
        for column in ["time", "forecast_reference_time", "blend_time"]:
            forecast_df[column] = pd.to_datetime(forecast_df[column], unit="ns", utc=True)
        forecast_df["forecast_period"] = pd.to_timedelta(forecast_df["forecast_period"], unit="us")
        forecast_df["period"] = pd.to_timedelta(forecast_df["period"], unit="us")
        

        # Load truths from parquet file filtering by diagnostic.
        filters = [[("diagnostic", "==", target_diagnostic_name)]]
        truth_df = pd.read_parquet(truth_table_path, filters=filters, schema=TRUTH_SCHEMA)
        truth_df["time"] = pd.to_datetime(truth_df["time"], unit="ns", utc=True)

        if truth_df.empty:
            msg = (
                f"The requested filepath {truth_table_path} does not contain the "
                f"requested contents: {filters}"
            )
            raise IOError(msg)
        
        forecast_cubes = iris.cube.CubeList([])
        truth_cubes = iris.cube.CubeList([])

        for forecast_period in forecast_periods:
            forecast_cube, truth_cube = forecast_and_truth_dataframes_to_cubes(
                forecast_df,
                truth_df,
                cycletime,
                forecast_period,
                training_length,
                experiment=experiment,
            )

            forecast_cube = iris.util.new_axis(forecast_cube, "forecast_period")
            forecast_cube.remove_coord("time")

            for forecast_slice in forecast_cube.slices_over("forecast_reference_time"):
                forecast_slice = iris.util.new_axis(forecast_slice, "forecast_reference_time")
                forecast_cubes.append(forecast_slice)
                

            for truth_slice in truth_cube.slices_over("time"):
                truth_slice = iris.util.new_axis(truth_slice, "time")
                truth_cubes.append(truth_slice)
        truth_cube = truth_cubes.concatenate_cube()
        forecast_cube = forecast_cubes.concatenate()

        # concatenate_cube() can fail for the forecast_cube, even though calling concatenate()
        # results in a single cube. This check ensures the concatenation was successful.
        if len(forecast_cube) == 1:
            forecast_cube = forecast_cube[0]
        else:
            msg = "Concatenating the forecast has failed to create a single cube."
            raise ValueError(msg)

        if len(cube_inputs) + 2 != len(file_paths):
            raise ValueError("Unable to identify the correct number of inputs")
        
        # If target_forecast is also a dynamic feature in the feature config then add it to cube_inputs
        for feature_name in feature_config.keys():
            if feature_name == forecast_cube[0].name():
                cube_inputs.append(forecast_cube)

        # Remove sites that have nans in the data
        bad_site_ids=[]
        nan_mask = np.any(np.isnan(truth_cube.data), axis=truth_cube.coord_dims("time"))
        all_site_ids = truth_cube.coord("wmo_id").points
        bad_site_ids = all_site_ids[nan_mask]
        wmo_ids = set(all_site_ids) - set(bad_site_ids)
        constr = iris.Constraint(wmo_id=lambda cell: cell in wmo_ids)
        truth_cube = truth_cube.extract(constr)
        forecast_cube = forecast_cube.extract(constr)

        feature_cube_inputs = iris.cube.CubeList([])
        for cube in cube_inputs:
            cube = cube.extract(constr)
            feature_cube_inputs.append(cube)

        result = TrainQuantileRegressionRandomForests(
                experiment=experiment,
                feature_config=feature_config,
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                transformation=transformation,
                pre_transform_addition=pre_transform_addition,
                compression=compression,
                model_output=model_output,
                )(forecast_cube, truth_cube, feature_cube_inputs)
        
        return result
