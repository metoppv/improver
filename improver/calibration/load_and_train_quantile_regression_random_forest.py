# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to load inputs and train a model using Quantile Regression Random Forest
(QRF)."""

import pathlib
from pathlib import Path
from typing import Optional

import iris
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from iris.pandas import as_data_frame

from improver import PostProcessingPlugin
from improver.calibration import CalibrationSchemas
from improver.calibration.quantile_regression_random_forest import (
    TrainQuantileRegressionRandomForests,
)
from improver.utilities.load import load_cube

iris.FUTURE.pandas_ndim = True


class LoadAndTrainQRF(PostProcessingPlugin):
    """Plugin to load and train a Quantile Regression Random Forest (QRF) model."""

    def __init__(
        self,
        feature_config: dict[str, list[str]],
        target_diagnostic_name: str,
        target_cf_name: str,
        forecast_periods: str,
        cycletime: str,
        training_length: int,
        experiment: Optional[str] = None,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        max_samples: Optional[float] = None,
        random_state: Optional[int] = None,
        transformation: Optional[str] = None,
        pre_transform_addition: float = 0,
        compression: int = 5,
    ):
        """Initialise the LoadAndTrainQRF plugin."""
        self.feature_config = feature_config
        self.target_diagnostic_name = target_diagnostic_name
        self.target_cf_name = target_cf_name
        self.forecast_periods = forecast_periods
        self.cycletime = cycletime
        self.training_length = training_length
        self.experiment = experiment
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_samples = max_samples
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
            if not Path(file_path).exists():
                # This will occur when the filepath does not exist. In this case,
                # return None.
                return None, None, None
            elif Path(file_path).is_dir():
                try:
                    example_file_path = next(Path(file_path).glob("**/*.parquet"))
                except StopIteration:
                    # If no parquet files are found, return None.
                    return None, None, None
                try:
                    pq.read_schema(example_file_path).field("forecast_period")
                    forecast_table_path = file_path
                except KeyError:
                    truth_table_path = file_path
            else:
                cube = load_cube(str(file_path))
                cube_inputs.append(cube)

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

        example_file_path = next(Path(forecast_table_path).glob("**/*.parquet"))
        if pq.read_schema(example_file_path).get_all_field_indices("percentile"):
            altered_schema = CalibrationSchemas.FORECAST_SCHEMA
        elif pq.read_schema(example_file_path).get_all_field_indices("realization"):
            altered_schema = CalibrationSchemas.FORECAST_SCHEMA.remove(
                CalibrationSchemas.FORECAST_SCHEMA.get_field_index("percentile")
            )
            altered_schema = altered_schema.append(pa.field("realization", pa.int64()))
        else:
            msg = (
                "The forecast parquet file is expected to contain either a "
                "'percentile' or 'realization' field. Neither was found."
            )
            raise ValueError(msg)

        forecast_df = pd.read_parquet(
            forecast_table_path,
            filters=filters,
            schema=altered_schema,
            engine="pyarrow",
        )

        forecast_df = forecast_df[
            forecast_df["forecast_period"].isin(np.array(forecast_periods) * 1e9)
        ]
        # Convert df columns from ns to pandas timestamp object.
        for column in ["time", "forecast_reference_time", "blend_time"]:
            forecast_df[column] = pd.to_datetime(
                forecast_df[column], unit="ns", utc=True
            )
        for column in ["forecast_period", "period"]:
            forecast_df[column] = pd.to_timedelta(forecast_df[column], unit="ns")

        forecast_df = forecast_df.rename(columns={"forecast": self.target_cf_name})

        # Load truths from parquet file filtering by diagnostic.
        filters = [[("diagnostic", "==", self.target_diagnostic_name)]]
        truth_df = pd.read_parquet(
            truth_table_path,
            filters=filters,
            schema=CalibrationSchemas.TRUTH_SCHEMA,
            engine="pyarrow",
        )

        truth_df["time"] = pd.to_datetime(truth_df["time"], unit="ns", utc=True)

        if truth_df.empty:
            msg = (
                f"The requested filepath {truth_table_path} does not contain the "
                f"requested contents: {filters}"
            )
            raise IOError(msg)
        return forecast_df, truth_df

    @staticmethod
    def _check_matching_times(
        forecast_df: pd.DataFrame, truth_df: pd.DataFrame
    ) -> list[pd.Timestamp]:
        """Find the intersecting times available within the forecast and truth
        DataFrames.

        Args:
            forecast_df: DataFrame containing the forecast data.
            truth_df: DataFrame containing the truth data.
        Returns:
            List of intersecting times as pandas Timestamp objects.
        """
        # Calling unique() on the time column is quicker than relying upon set() to
        # find the unique times.
        return list(
            set(forecast_df["time"].unique()).intersection(
                set(truth_df["time"].unique())
            )
        )

    def _add_features_to_df(
        self, forecast_df: pd.DataFrame, cube_inputs: iris.cube.CubeList
    ) -> pd.DataFrame:
        """Add features to the forecast DataFrame based on the feature configuration.

        Args:
            forecast_df: DataFrame containing the forecast data.
            cube_inputs: List of cubes containing additional features.

        Returns:
            DataFrame with additional features added.
        """
        for feature_name, feature_list in self.feature_config.items():
            for feature in feature_list:
                if feature == "static":
                    # Use the cube's data directly as a feature.
                    constr = iris.Constraint(name=feature_name)
                    feature_cube = cube_inputs.extract_cube(constr)
                    feature_df = as_data_frame(feature_cube, add_aux_coords=True)
                    forecast_df = forecast_df.merge(
                        feature_df[["wmo_id", feature_name]], on=["wmo_id"], how="left"
                    )
        return forecast_df

    @staticmethod
    def filter_bad_sites(
        forecast_df: pd.DataFrame,
        truth_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Remove sites that have NaNs in the data.

        Args:
            feature_df: DataFrame containing the forecast data with features.
            truth_df: DataFrame containing the truth data.

        Returns:
            Tuple containing:
                - DataFrame containing the forecast data with bad sites removed.
                - DataFrame containing the truth data with bad sites removed.
        """
        truth_df.dropna(
            subset=["latitude", "longitude", "altitude", "ob_value"], inplace=True
        )

        wmo_ids = set(forecast_df["wmo_id"]).intersection(set(truth_df["wmo_id"]))

        forecast_df = forecast_df[forecast_df["wmo_id"].isin(wmo_ids)]
        truth_df = truth_df[truth_df["wmo_id"].isin(wmo_ids)]
        return forecast_df, truth_df

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
            model_output (str):
                Full path including model file name that will store the pickled model.

        """
        forecast_table_path, truth_table_path, cube_inputs = (
            self._split_cubes_and_parquet_files(file_paths)
        )

        if not forecast_table_path or not truth_table_path:
            return None

        if ":" in self.forecast_periods:
            forecast_periods = list(range(*map(int, self.forecast_periods.split(":"))))
            forecast_periods = [fp * 3600 for fp in forecast_periods]
        else:
            try:
                forecast_periods = [int(self.forecast_periods) * 3600]
            except ValueError:
                msg = (
                    "The forecast_periods argument must be a single integer or "
                    "a range in the form 'start:end:interval'. The forecast period"
                    f"provided was: {self.forecast_periods}."
                )
                raise ValueError(msg)

        forecast_df, truth_df = self._read_parquet_files(
            forecast_table_path, truth_table_path, forecast_periods
        )
        intersecting_times = self._check_matching_times(forecast_df, truth_df)
        if len(intersecting_times) == 0:
            return None

        forecast_df = self._add_features_to_df(forecast_df, cube_inputs)
        forecast_df, truth_df = self.filter_bad_sites(forecast_df, truth_df)

        TrainQuantileRegressionRandomForests(
            target_name=self.target_cf_name,
            feature_config=self.feature_config,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_samples=self.max_samples,
            random_state=self.random_state,
            transformation=self.transformation,
            pre_transform_addition=self.pre_transform_addition,
            compression=self.compression,
            model_output=model_output,
        )(forecast_df, truth_df)
