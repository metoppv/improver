# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to load inputs and train a model using Quantile Regression Random Forest (QRF)."""

import pathlib
from pathlib import Path
from typing import Optional

import iris
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from improver import PostProcessingPlugin
from improver.calibration import FORECAST_SCHEMA, TRUTH_SCHEMA
from improver.calibration.dataframe_utilities import (
    forecast_and_truth_dataframes_to_cubes,
)
from improver.calibration.quantile_regression_random_forest import (
    TrainQuantileRegressionRandomForests,
)
from improver.utilities.load import load_cube


class LoadAndTrainQRF(PostProcessingPlugin):
    """Plugin to load and train a Quantile Regression Random Forest (QRF) model."""

    def __init__(
        self,
        feature_config: dict[str, list[str]],
        target_diagnostic_name: str,
        forecast_periods: str,
        cycletime: str,
        training_length: int,
        experiment: str = None,
        n_estimators: int = 100,
        max_depth: int = None,
        random_state: int = None,
        transformation: str = None,
        pre_transform_addition: float = 0,
        compression: int = 5,
    ):
        """Initialise the LoadAndTrainQRF plugin."""
        self.feature_config = feature_config
        self.target_diagnostic_name = target_diagnostic_name
        self.forecast_periods = forecast_periods
        self.cycletime = cycletime
        self.training_length = training_length
        self.experiment = experiment
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.transformation = transformation
        self.pre_transform_addition = pre_transform_addition
        self.compression = compression

    def _split_cubes_and_parquet_files(
        self, file_paths: list[pathlib.Path | str]
    ) -> tuple[Optional[pathlib.Path], Optional[pathlib.Path], iris.cube.CubeList]:
        """Split the input file paths into cubes and parquet files.

        Args:
            file_paths: List of file paths.

        Returns:
            Tuple containing the items below if found:
                - Path to the forecast parquet file.
                - Path to the truth parquet file.
                - List of cubes loaded from the NetCDF files.

        Raises:
            ValueError: If the number of cubes loaded does not match the number of
                features expected.
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
                for file in Path(file_path).glob("**/*.parquet"):
                    try:
                        pq.read_schema(file).field("forecast_period")
                        forecast_table_path = file_path
                    except KeyError:
                        truth_table_path = file_path
                    if forecast_table_path and truth_table_path:
                        break
            except OSError:
                # This will occur when the filepath does not exist. In this case,
                # return None.
                return None, None, None

        if len(self.feature_config.keys()) not in [
            len(cube_inputs),
            len(cube_inputs) + 1,
        ]:
            msg = (
                "The number of cubes loaded does not match the number of features "
                "expected. These can mismatch if some features are coming from the "
                "historic forecast table. The number of cubes loaded was: "
                f"{len(cube_inputs)}. The number of features expected was: "
                f"{len(self.feature_config.keys())}."
            )
            raise ValueError(msg)

        return forecast_table_path, truth_table_path, cube_inputs

    def _read_parquet_files(
        self,
        forecast_table_path: pathlib.Path | str,
        truth_table_path: pathlib.Path | str,
        forecast_periods: list[int],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Read the forecast and truth data from parquet files.

        Args:
            forecast_table_path: Path to the forecast parquet file.
            truth_table_path: Path to the truth parquet file.
            forecast_periods: List of forecast periods in seconds.

        Returns:
            Tuple containing:
                - DataFrame containing the forecast data.
                - DataFrame containing the truth data.

        Raises:
            ValueError: If the forecast parquet file does not contain the expected
                fields.
            ValueError: If the truth parquet file does not contain the expected
                fields.
        """
        cycletimes = []

        for forecast_period in forecast_periods:
            # Load forecasts from parquet file filtering by diagnostic and blend_time.
            forecast_period_td = pd.Timedelta(int(forecast_period), unit="seconds")

            cycletimes.extend(
                pd.date_range(
                    end=pd.Timestamp(self.cycletime)
                    - pd.Timedelta(1, unit="days")
                    - forecast_period_td.floor("D"),
                    periods=int(self.training_length),
                    freq="D",
                )
            )
        cycletimes = list(set(cycletimes))

        filters = [
            [
                ("diagnostic", "==", self.target_diagnostic_name),
                ("blend_time", "in", cycletimes),
                ("experiment", "==", self.experiment),
            ]
        ]

        for file in Path(forecast_table_path).glob("**/*.parquet"):
            if pq.read_schema(file).get_all_field_indices("percentile"):
                altered_schema = FORECAST_SCHEMA
            elif pq.read_schema(file).get_all_field_indices("realization"):
                altered_schema = FORECAST_SCHEMA.remove(
                    FORECAST_SCHEMA.get_field_index("percentile")
                )
                altered_schema = altered_schema.append(
                    pa.field("realization", pa.int64())
                )
            else:
                msg = (
                    "The forecast parquet file is expected to contain either a "
                    "'percentile' or 'realization' field. Neither was found."
                )
                raise ValueError(msg)
            break

        forecast_df = pd.read_parquet(
            forecast_table_path,
            filters=filters,
            schema=altered_schema,
            engine="pyarrow",
        )

        # Convert df columns from ms to pandas timestamp object to work with existing code
        for column in ["time", "forecast_reference_time", "blend_time"]:
            forecast_df[column] = pd.to_datetime(
                forecast_df[column], unit="ns", utc=True
            )
        forecast_df["forecast_period"] = pd.to_timedelta(
            forecast_df["forecast_period"], unit="us"
        )
        forecast_df["period"] = pd.to_timedelta(forecast_df["period"], unit="us")

        # Load truths from parquet file filtering by diagnostic.
        filters = [[("diagnostic", "==", self.target_diagnostic_name)]]
        truth_df = pd.read_parquet(
            truth_table_path, filters=filters, schema=TRUTH_SCHEMA, engine="pyarrow"
        )

        truth_df["time"] = pd.to_datetime(truth_df["time"], unit="ns", utc=True)

        if truth_df.empty:
            msg = (
                f"The requested filepath {truth_table_path} does not contain the "
                f"requested contents: {filters}"
            )
            raise IOError(msg)
        return forecast_df, truth_df

    def _dataframe_to_cubes(
        self,
        forecast_df: pd.DataFrame,
        truth_df: pd.DataFrame,
        forecast_periods: list[int],
    ) -> tuple[iris.cube.Cube, iris.cube.Cube]:
        """Convert the forecast and truth dataframes to cubes at each forecast period
        required.

        Args:
            forecast_df: DataFrame containing the forecast data.
            truth_df: DataFrame containing the truth data.
            forecast_periods: List of forecast periods in seconds.

        Returns:
            Tuple containing:
                - Cube containing the forecast data.
                - Cube containing the truth data.

        Raises:
            ValueError: The forecast has failed to concatenate into a single cube.
        """
        forecast_cubes = iris.cube.CubeList([])
        truth_cubes = iris.cube.CubeList([])

        for forecast_period in forecast_periods:
            forecast_cube, truth_cube = forecast_and_truth_dataframes_to_cubes(
                forecast_df,
                truth_df,
                self.cycletime,
                forecast_period,
                self.training_length,
                experiment=self.experiment,
            )

            if forecast_cube is None or truth_cube is None:
                continue

            if not forecast_cube.coords("realization", dim_coords=True):
                forecast_cube = iris.util.new_axis(forecast_cube, "realization")
            forecast_cube = iris.util.new_axis(forecast_cube, "forecast_period")
            forecast_cube.remove_coord("time")

            for forecast_slice in forecast_cube.slices_over("forecast_reference_time"):
                forecast_slice = iris.util.new_axis(
                    forecast_slice, "forecast_reference_time"
                )
                forecast_cubes.append(forecast_slice)

            for truth_slice in truth_cube.slices_over("time"):
                truth_slice = iris.util.new_axis(truth_slice, "time")
                truth_cubes.append(truth_slice)

        truth_cube = truth_cubes.concatenate_cube()
        forecast_cube = forecast_cubes.concatenate()

        # concatenate_cube() can fail for the forecast_cube, even though calling
        # concatenate() results in a single cube. This check ensures the concatenation
        # was successful.
        if len(forecast_cube) == 1:
            forecast_cube = forecast_cube[0]
        else:
            msg = "Concatenating the forecast has failed to create a single cube."
            raise ValueError(msg)

        return forecast_cube, truth_cube

    @staticmethod
    def filter_bad_sites(
        forecast_cube: iris.cube.Cube,
        truth_cube: iris.cube.Cube,
        cube_inputs: iris.cube.CubeList,
    ) -> tuple[iris.cube.Cube, iris.cube.Cube, iris.cube.CubeList]:
        """Remove sites that have NaNs in the data.

        Args:
            forecast_cube: Cube containing the forecast data.
            truth_cube: Cube containing the truth data.
            cube_inputs: List of additional feature cubes.

        Returns:
            Tuple containing:
                - Cube containing the forecast data with bad sites removed.
                - Cube containing the truth data with bad sites removed.
                - List of additional feature cubes with bad sites removed.
        """
        nan_mask = np.any(np.isnan(truth_cube.data), axis=truth_cube.coord_dims("time"))
        all_site_ids = truth_cube.coord("wmo_id").points
        bad_site_ids = all_site_ids[nan_mask]
        constr = iris.Constraint(wmo_id=lambda cell: cell not in bad_site_ids.tolist())
        truth_cube = truth_cube.extract(constr)
        forecast_cube = forecast_cube.extract(constr)
        feature_cube_inputs = iris.cube.CubeList([])
        for cube in cube_inputs:
            cube = cube.extract(constr)
            feature_cube_inputs.append(cube)

        return forecast_cube, truth_cube, feature_cube_inputs

    def process(
        self,
        file_paths: list[pathlib.Path | str],
        model_output: str = None,
    ) -> None:
        """Load input files and training a Quantile Regression Random Forest (QRF)
        model. This model can be applied later to calibrate the forecast. Two sources
        of input data must be provided: historical forecasts and historical truth data
        (to use in calibration). The model is output as a pickle file.

        Args:
            file_paths (cli.inputpaths):
                A list of input paths containing:
                - The path to a Parquet file containing the truths to be used
                for calibration. The expected columns within the
                Parquet file are: ob_value, time, wmo_id, diagnostic, latitude,
                longitude and altitude.
                - The path to a Parquet file containing the forecasts to be used
                for calibration.
                - Optionally, paths to NetCDF files containing additional predictors.
            feature_config (dict):
                Feature configuration defining the features to be used for quantile
                regression. The configuration is a dictionary of strings, where the
                keys are the names of the input cube(s) supplied, and the values are
                a list. This list can contain both computed features, such as the mean
                or standard deviation (std), or static features, such as the altitude.
                The computed features will be computed using the cube defined in the
                dictionary key. If the key is the feature itself e.g. a distance to
                water cube, then the value should state "static". This will ensure
                the cube's data is used as the feature. The config will have the
                structure:
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
                Range of forecast periods to be included in training in hours in the
                form: "start:end:interval" e.g. "6:18:6".
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
                Supported transformations are "log", "log10", "sqrt", and "cbrt".
            pre_transform_addition (float):
                Value to be added before transformation. This is useful for ensuring
                that the data is positive before applying a transformation such as log.
            compression (int):
                Compression level for saving the model. Please see the joblib
                documentation for more information on compression levels.
            model_output (str):
                Full path including model file name that will store the pickled model.

        """
        forecast_table_path, truth_table_path, cube_inputs = (
            self._split_cubes_and_parquet_files(file_paths)
        )
        if not forecast_table_path or not truth_table_path:
            return None

        forecast_periods = list(range(*map(int, self.forecast_periods.split(":"))))
        forecast_periods = [fp * 3600 for fp in forecast_periods]

        forecast_df, truth_df = self._read_parquet_files(
            forecast_table_path, truth_table_path, forecast_periods
        )

        forecast_cube, truth_cube = self._dataframe_to_cubes(
            forecast_df, truth_df, forecast_periods
        )

        # If target_forecast is also a dynamic feature in the feature config then
        # add it to cube_inputs
        for feature_name in self.feature_config.keys():
            if feature_name == forecast_cube[0].name():
                cube_inputs.append(forecast_cube)

        forecast_cube, truth_cube, feature_cube_inputs = self.filter_bad_sites(
            forecast_cube, truth_cube, cube_inputs
        )

        TrainQuantileRegressionRandomForests(
            feature_config=self.feature_config,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            transformation=self.transformation,
            pre_transform_addition=self.pre_transform_addition,
            compression=self.compression,
            model_output=model_output,
        )(forecast_cube, truth_cube, feature_cube_inputs)
