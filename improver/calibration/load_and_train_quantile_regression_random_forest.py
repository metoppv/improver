# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to load inputs and train a model using Quantile Regression Random Forest
(QRF)."""

import pathlib
from pathlib import Path
from typing import Optional, Union

import iris
import numpy as np
import pandas as pd
from iris.pandas import as_data_frame

from improver import PostProcessingPlugin
from improver.calibration import (
    CalibrationSchemas,
    get_training_period_cycles,
    identify_parquet_type,
    split_pickle_parquet_and_netcdf,
)
from improver.calibration.quantile_regression_random_forest import (
    TrainQuantileRegressionRandomForests,
    quantile_forest_package_available,
)

try:
    from quantile_forest import RandomForestQuantileRegressor
except ModuleNotFoundError:
    # Define empty class to avoid type hint errors.
    class RandomForestQuantileRegressor:
        pass


iris.FUTURE.pandas_ndim = True


class LoadForTrainQRF(PostProcessingPlugin):
    """Plugin to load input files for training a Quantile Regression Random Forest
    (QRF) model."""

    def __init__(
        self,
        feature_config: dict[str, list[str]],
        parquet_diagnostic_names: Union[list[str], str],
        target_cf_name: str,
        forecast_periods: str,
        cycletime: str,
        training_length: int,
        experiment: Optional[str] = None,
        unique_site_id_keys: Union[list[str], str] = "wmo_id",
    ):
        """Initialise the LoadForTrainQRF plugin.

        Args:
            feature_config: Feature configuration defining the features to be used for
                Quantile Regression Random Forests.
            parquet_diagnostic_names:
                A string containing the diagnostic name that will be used for filtering
                the target diagnostic from the forecast and truth DataFrames read in
                from the parquet files. This could be different from the CF name e.g.
                'temperature_at_screen_level'.
            target_cf_name: A string containing the CF name of the forecast to be
                calibrated e.g. air_temperature.
            forecast_periods: Range of forecast periods to be calibrated in hours in
                the form: "start:end:interval" e.g. "6:18:6" or a single forecast period
                e.g. "6".
            cycletime: The time at which the forecast is valid in the form:
                YYYYMMDDTHHMMZ.
            training_length: The number of days of training data to use.
                experiment: The name of the experiment (step) that calibration is applied to.
            unique_site_id_key: The names of the coordinates that uniquely identify
                each site, e.g. "wmo_id" or "latitude,longitude".
        """
        self.quantile_forest_installed = quantile_forest_package_available()
        self.feature_config = feature_config
        self.parquet_diagnostic_names = parquet_diagnostic_names
        self.target_cf_name = target_cf_name
        self.forecast_periods = forecast_periods
        self.cycletime = cycletime
        self.training_length = training_length
        self.experiment = experiment
        if isinstance(unique_site_id_keys, str):
            unique_site_id_keys = [unique_site_id_keys]
        self.unique_site_id_keys = unique_site_id_keys

    def _parse_forecast_periods(self) -> list[int]:
        """Parse the forecast periods argument to produce a list of forecast periods
        in seconds.

        Returns:
            List of forecast periods in seconds.
        Raises:
            ValueError: If the forecast_periods argument is not a single integer or
                a range in the form 'start:end:interval'.
        """
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
        return forecast_periods

    def _read_parquet_files(
        self,
        forecast_table_path: pathlib.Path | str,
        truth_table_path: pathlib.Path | str,
        forecast_periods: list[int],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Read the forecast and truth data from parquet files.
        self.quantile_forest_installed = quantile_forest_package_available()
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
        import pyarrow as pa
        import pyarrow.parquet as pq

        cycletimes = []
        for forecast_period in forecast_periods:
            # Load forecasts from parquet file filtering by diagnostic and blend_time.
            cycletimes.extend(
                get_training_period_cycles(
                    self.cycletime, forecast_period, self.training_length
                )
            )
        cycletimes = list(set(cycletimes))

        filters = [
            [
                ("diagnostic", "in", self.parquet_diagnostic_names),
                ("blend_time", "in", cycletimes),
                ("experiment", "==", self.experiment),
            ]
        ]

        example_file_path = next(Path(forecast_table_path).glob("**/*.parquet"))
        if pq.read_schema(example_file_path).get_all_field_indices("percentile"):
            altered_schema = CalibrationSchemas().FORECAST_SCHEMA
        elif pq.read_schema(example_file_path).get_all_field_indices("realization"):
            altered_schema = CalibrationSchemas().FORECAST_SCHEMA.remove(
                CalibrationSchemas().FORECAST_SCHEMA.get_field_index("percentile")
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
        ].reset_index(drop=True)

        # Convert df columns from ns to pandas timestamp object.
        for column in ["time", "forecast_reference_time", "blend_time"]:
            forecast_df[column] = pd.to_datetime(
                forecast_df[column], unit="ns", utc=True
            )
            forecast_df[column] = forecast_df[column].astype("datetime64[ns, UTC]")
        for column in ["forecast_period", "period"]:
            forecast_df[column] = pd.to_timedelta(forecast_df[column], unit="ns")
            forecast_df[column] = forecast_df[column].astype("timedelta64[ns]")

        # Convert additional features from rows to columns.
        representation = (
            "percentile" if "percentile" in forecast_df.columns else "realization"
        )
        base_df = forecast_df[
            forecast_df["diagnostic"] == self.parquet_diagnostic_names[0]
        ]
        for parquet_diagnostic_name in self.parquet_diagnostic_names[1:]:
            additional_df = forecast_df[
                forecast_df["diagnostic"] == parquet_diagnostic_name
            ]
            base_df = pd.merge(
                base_df,
                additional_df[
                    [
                        *self.unique_site_id_keys,
                        "forecast_reference_time",
                        "forecast_period",
                        representation,
                        "forecast",
                    ]
                ].rename(columns={"forecast": parquet_diagnostic_name}),
                on=[
                    *self.unique_site_id_keys,
                    "forecast_reference_time",
                    "forecast_period",
                    representation,
                ],
                how="left",
            )
        forecast_df = base_df
        forecast_df.rename(columns={"forecast": self.target_cf_name}, inplace=True)

        # Load truths from parquet file filtering by diagnostic.
        filters = [[("diagnostic", "==", self.parquet_diagnostic_names[0])]]
        truth_df = pd.read_parquet(
            truth_table_path,
            filters=filters,
            schema=CalibrationSchemas().TRUTH_SCHEMA,
            engine="pyarrow",
        )

        truth_df["time"] = pd.to_datetime(truth_df["time"], unit="ns", utc=True)
        truth_df["time"] = truth_df["time"].astype("datetime64[ns, UTC]")

        if truth_df.empty:
            msg = (
                f"The requested filepath {truth_table_path} does not contain the "
                f"requested contents: {filters}"
            )
            raise IOError(msg)
        return forecast_df, truth_df

    def process(
        self,
        file_paths: list[pathlib.Path | str],
    ) -> Optional[tuple[iris.cube.CubeList, pathlib.Path | str, pathlib.Path | str]]:
        """Load input files for training a Quantile Regression Random Forest (QRF)
        model. Two sources of input data must be provided: historical forecasts and
        historical truth data (to use in calibration).

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
        """
        if not self.quantile_forest_installed:
            return None
        cube_inputs, parquets, _ = split_pickle_parquet_and_netcdf(file_paths)

        # If there are no parquet files, return None.
        if not parquets:
            return None, None, None

        forecast_table_path, truth_table_path = identify_parquet_type(parquets)

        # If either the forecast or truth parquet files are missing, return None.
        if not forecast_table_path or not truth_table_path:
            return None, None, None

        forecast_periods = self._parse_forecast_periods()
        forecast_df, truth_df = self._read_parquet_files(
            forecast_table_path, truth_table_path, forecast_periods
        )

        if cube_inputs is None:
            cube_inputs = iris.cube.CubeList()
        missing_features = (
            set(self.feature_config.keys())
            - set(forecast_df.columns)
            - set([c.name() for c in cube_inputs])
        )

        if len(missing_features) > 0:
            msg = (
                "The features requested in the feature_config are absent from "
                "the forecast parquet file and the input cubes. The missing fields are: "
                f"{missing_features}."
            )
            raise ValueError(msg)
        return forecast_df, truth_df, cube_inputs


class PrepareAndTrainQRF(PostProcessingPlugin):
    """Plugin to prepare and train a Quantile Regression Random Forest (QRF) model."""

    def __init__(
        self,
        feature_config: dict[str, list[str]],
        target_cf_name: str,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        max_samples: Optional[float] = None,
        random_state: Optional[int] = None,
        transformation: Optional[str] = None,
        pre_transform_addition: float = 0,
        unique_site_id_keys: Union[list[str], str] = "wmo_id",
    ):
        """Initialise the PrepareAndTrainQRF plugin.

        Args:
            feature_config: Feature configuration defining the features to be used for
                Quantile Regression Random Forests.
            target_cf_name: A string containing the CF name of the forecast to be
                calibrated e.g. air_temperature.
            n_estimators: The number of trees in the forest.
            max_depth: The maximum depth of the trees.
            max_samples: The maximum number of samples to draw from the total number of
                samples to train each tree.
            random_state: Seed used by the random number generator.
            transformation: Transformation to be applied to the data before fitting.
            pre_transform_addition: Value to be added before transformation.
            unique_site_id_key: The names of the coordinates that uniquely identify
                each site, e.g. "wmo_id" or "latitude,longitude".
        """
        self.feature_config = feature_config
        self.target_cf_name = target_cf_name
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.random_state = random_state
        self.transformation = transformation
        self.pre_transform_addition = pre_transform_addition
        if isinstance(unique_site_id_keys, str):
            unique_site_id_keys = [unique_site_id_keys]
        self.unique_site_id_keys = unique_site_id_keys
        self.quantile_forest_installed = quantile_forest_package_available()

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

    def _add_static_features_from_cubes_to_df(
        self, forecast_df: pd.DataFrame, cube_inputs: iris.cube.CubeList
    ) -> pd.DataFrame:
        """Add features to the forecast DataFrame from cubes based on the feature
        configuration. Other features are expected to already be present in the
        forecast DataFrame.

        Args:
            forecast_df: DataFrame containing the forecast data.
            cube_inputs: List of cubes containing additional features.

        Returns:
            DataFrame with additional features added from the input cubes.
        """
        if cube_inputs is None or len(cube_inputs) == 0:
            return forecast_df
        for feature_name, feature_list in self.feature_config.items():
            for feature in feature_list:
                if feature == "static":
                    # Use the cube's data directly as a feature.
                    constr = iris.Constraint(name=feature_name)
                    # Static features can be provided either as a cube or as a column
                    # in the forecast DataFrame.
                    if not cube_inputs.extract(constr):
                        continue
                    feature_cube = cube_inputs.extract_cube(constr)
                    feature_df = as_data_frame(feature_cube, add_aux_coords=True)
                    forecast_df = forecast_df.merge(
                        feature_df[[*self.unique_site_id_keys, feature_name]],
                        on=[*self.unique_site_id_keys],
                        how="left",
                    )
        return forecast_df

    def filter_bad_sites(
        self,
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
        truth_df.dropna(subset=["ob_value"], inplace=True)

        if truth_df.empty:
            msg = "Empty truth DataFrame after removing NaNs."
            raise ValueError(msg)

        forecast_index = forecast_df.set_index([*self.unique_site_id_keys]).index
        truth_index = truth_df.set_index([*self.unique_site_id_keys]).index
        forecast_df = forecast_df[forecast_index.isin(truth_index)]
        truth_df = truth_df[truth_index.isin(forecast_index)]

        return forecast_df, truth_df

    def process(
        self,
        forecast_df: pd.DataFrame,
        truth_df: pd.DataFrame,
        cube_inputs: Optional[iris.cube.CubeList] = None,
    ) -> Optional[tuple[RandomForestQuantileRegressor, str, float]]:
        """Load input files and train a Quantile Regression Random Forest (QRF)
        model. This model can be applied later to calibrate the forecast. Two sources
        of input data must be provided: historical forecasts and historical truth data
        (to use in calibration). The model is output as a pickle file.

        Args:
            forecast_df: DataFrame containing the forecast data.
            truth_df: DataFrame containing the truth data.
            cube_inputs: List of cubes containing additional features.

        Raises:
            ValueError: If the number of cubes loaded does not match the number of
                features expected.
        """
        if not self.quantile_forest_installed:
            return None

        intersecting_times = self._check_matching_times(forecast_df, truth_df)
        if len(intersecting_times) == 0:
            return None

        forecast_df = self._add_static_features_from_cubes_to_df(
            forecast_df, cube_inputs
        )
        forecast_df, truth_df = self.filter_bad_sites(forecast_df, truth_df)

        result = TrainQuantileRegressionRandomForests(
            target_name=self.target_cf_name,
            feature_config=self.feature_config,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_samples=self.max_samples,
            random_state=self.random_state,
            transformation=self.transformation,
            pre_transform_addition=self.pre_transform_addition,
            unique_site_id_keys=self.unique_site_id_keys,
        )(forecast_df, truth_df)

        # Create a tuple that returns the model, transformation and
        # pre_transform_addition to allow these to be saved together.
        return (result, self.transformation, self.pre_transform_addition)
