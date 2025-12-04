# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""init for calibration that contains functionality to split forecast, truth
and coefficient inputs.
"""

import glob
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import iris
import joblib
import numpy as np
import pandas as pd
from iris.cube import Cube, CubeList
from iris.pandas import as_data_frame

from improver.metadata.probabilistic import (
    get_diagnostic_cube_name_from_probability_name,
)
from improver.utilities.cube_manipulation import MergeCubes
from improver.utilities.flatten import flatten
from improver.utilities.load import load_cubelist

iris.FUTURE.pandas_ndim = True


class CalibrationSchemas:
    def __init__(self):
        """Define the pyarrow schemas for forecast and truth parquet files."""
        import pyarrow as pa

        self.FORECAST_SCHEMA = pa.schema(
            [
                ("percentile", pa.float64()),
                ("forecast", pa.float32()),
                ("altitude", pa.float32()),
                ("blend_time", pa.timestamp("s", "utc")),
                ("forecast_period", pa.int64()),
                ("forecast_reference_time", pa.timestamp("s", "utc")),
                ("latitude", pa.float32()),
                ("longitude", pa.float32()),
                ("time", pa.timestamp("s", "utc")),
                ("wmo_id", pa.string()),
                ("station_id", pa.string()),
                ("cf_name", pa.string()),
                ("units", pa.string()),
                ("experiment", pa.string()),
                ("period", pa.int64()),
                ("height", pa.float32()),
                ("diagnostic", pa.string()),
            ]
        )
        self.TRUTH_SCHEMA = pa.schema(
            [
                ("diagnostic", pa.string()),
                ("latitude", pa.float32()),
                ("longitude", pa.float32()),
                ("altitude", pa.float32()),
                ("time", pa.timestamp("s", "utc")),
                ("wmo_id", pa.string()),
                ("station_id", pa.string()),
                ("ob_value", pa.float32()),
            ]
        )


def split_forecasts_and_truth(
    cubes: List[Cube], truth_attribute: str
) -> Tuple[Cube, Cube, Optional[Cube]]:
    """
    A common utility for splitting the various inputs cubes required for
    calibration CLIs. These are generally the forecast cubes, historic truths,
    and in some instances a land-sea mask is also required.

    Args:
        cubes:
            A list of input cubes which will be split into relevant groups.
            These include the historical forecasts, in the format supported by
            the calibration CLIs, and the truth cubes.
        truth_attribute:
            An attribute and its value in the format of "attribute=value",
            which must be present on truth cubes.

    Returns:
        - A cube containing all the historic forecasts.
        - A cube containing all the truth data.
        - If found within the input cubes list a land-sea mask will be
          returned, else None is returned.

    Raises:
        ValueError:
            An unexpected number of distinct cube names were passed in.
        IOError:
            More than one cube was identified as a land-sea mask.
        IOError:
            Missing truth or historical forecast in input cubes.
    """
    grouped_cubes = {}
    for cube in cubes:
        try:
            cube_name = get_diagnostic_cube_name_from_probability_name(cube.name())
        except ValueError:
            cube_name = cube.name()
        grouped_cubes.setdefault(cube_name, []).append(cube)
    if len(grouped_cubes) == 1:
        # Only one group - all forecast/truth cubes
        land_sea_mask = None
        diag_name = list(grouped_cubes.keys())[0]
    elif len(grouped_cubes) == 2:
        # Two groups - the one with exactly one cube matching a name should
        # be the land_sea_mask, since we require more than 2 cubes in
        # the forecast/truth group
        grouped_cubes = OrderedDict(
            sorted(grouped_cubes.items(), key=lambda kv: len(kv[1]))
        )
        # landsea name should be the key with the lowest number of cubes (1)
        landsea_name, diag_name = list(grouped_cubes.keys())
        land_sea_mask = grouped_cubes[landsea_name][0]
        if len(grouped_cubes[landsea_name]) != 1:
            raise IOError("Expected one cube for land-sea mask.")
    else:
        raise ValueError("Must have cubes with 1 or 2 distinct names.")

    # split non-land_sea_mask cubes on forecast vs truth
    truth_key, truth_value = truth_attribute.split("=")
    input_cubes = grouped_cubes[diag_name]
    grouped_cubes = {"truth": [], "historical forecast": []}
    for cube in input_cubes:
        if cube.attributes.get(truth_key) == truth_value:
            grouped_cubes["truth"].append(cube)
        else:
            grouped_cubes["historical forecast"].append(cube)

    missing_inputs = " and ".join(k for k, v in grouped_cubes.items() if not v)
    if missing_inputs:
        raise IOError(f"Missing {missing_inputs} input.")

    truth = MergeCubes()(grouped_cubes["truth"])
    forecast = MergeCubes()(grouped_cubes["historical forecast"])

    return forecast, truth, land_sea_mask


def split_forecasts_and_coeffs(
    cubes: Union[List[CubeList[Cube]], List[List[Cube]]],
    land_sea_mask_name: Optional[str] = None,
):
    """Split the input forecast, coefficients, static additional predictors,
    land sea-mask and probability template, if provided. The coefficients
    cubes and land-sea mask are identified based on their name. The
    static additional predictors are identified as not have a time
    coordinate. The current forecast and probability template are then split.

    Args:
        cubes:
            A list either containing a CubeList or containing a list of input cubes
            which will be split into relevant groups. This includes the forecast,
            coefficients, static additional predictors, land-sea mask and probability
            template.
        land_sea_mask_name:
            Name of the land-sea mask cube to help identification.

    Returns:
        - A cube containing the current forecast.
        - If found, a cubelist containing the coefficients else None.
        - If found, a cubelist containing the static additional predictor else None.
        - If found, a land-sea mask will be returned, else None.
        - If found, a probability template will be returned, else None.

    Raises:
        ValueError: If multiple items provided, when only one is expected.
        ValueError: If no forecast is found.
    """
    coefficients = CubeList()
    land_sea_mask = None
    grouped_cubes: Dict[str, List[Cube]] = {}
    static_additional_predictors = CubeList()

    for cubelist in cubes:
        for cube in cubelist:
            if "emos_coefficient" in cube.name():
                coefficients.append(cube)
            elif land_sea_mask_name and cube.name() == land_sea_mask_name:
                land_sea_mask = cube
            elif "time" not in [c.name() for c in cube.coords()]:
                static_additional_predictors.append(cube)
            else:
                if "probability" in cube.name() and any(
                    "probability" in k for k in grouped_cubes
                ):
                    msg = (
                        "Providing multiple probability cubes is "
                        "not supported. A probability cube can "
                        "either be provided as the forecast or "
                        "the probability template, but not both. "
                        f"Cubes provided: {grouped_cubes.keys()} "
                        f"and {cube.name()}."
                    )
                    raise ValueError(msg)
                elif cube.name() in grouped_cubes:
                    msg = (
                        "Multiple items have been provided with the "
                        f"name {cube.name()}. Only one item is expected."
                    )
                    raise ValueError(msg)
                grouped_cubes.setdefault(cube.name(), []).append(cube)

    prob_template = None
    # Split the forecast and the probability template.
    if len(grouped_cubes) == 0:
        msg = "No forecast is present. A forecast cube is required."
        raise ValueError(msg)
    elif len(grouped_cubes) == 1:
        (current_forecast,) = list(grouped_cubes.values())[0]
    elif len(grouped_cubes) == 2:
        for key in grouped_cubes.keys():
            if "probability" in key:
                (prob_template,) = grouped_cubes[key]
            else:
                (current_forecast,) = grouped_cubes[key]

    coefficients = coefficients if coefficients else None
    static_additional_predictors = (
        static_additional_predictors if static_additional_predictors else None
    )
    return (
        current_forecast,
        coefficients,
        static_additional_predictors,
        land_sea_mask,
        prob_template,
    )


def split_forecasts_and_bias_files(cubes: CubeList) -> Tuple[Cube, Optional[CubeList]]:
    """Split the input forecast from the forecast error files used for bias-correction.

    Args:
        cubes:
            A list of input cubes which will be split into forecast and forecast errors.

    Returns:
        - A cube containing the current forecast.
        - If found, a cube or cubelist containing the bias correction files.

    Raises:
        ValueError: If multiple forecast cubes provided, when only one is expected.
        ValueError: If no forecast is found.
    """
    forecast_cube = None
    bias_cubes = CubeList()

    for cube in cubes:
        if "forecast_error" in cube.name():
            bias_cubes.append(cube)
        else:
            if forecast_cube is None:
                forecast_cube = cube
            else:
                msg = (
                    "Multiple forecast inputs have been provided. Only one is expected."
                )
                raise ValueError(msg)

    if forecast_cube is None:
        msg = "No forecast is present. A forecast cube is required."
        raise ValueError(msg)

    bias_cubes = bias_cubes if bias_cubes else None

    return forecast_cube, bias_cubes


def split_netcdf_parquet_pickle(files):
    """Split the input files into netcdf, parquet, and pickle files.
    Only a single pickle file is expected.

    Args:
        files:
            A list of input file paths which will be split into pickle,
            parquet, and netcdf files.

    Returns:
        - A flattened cube list containing all the cubes contained within the
          provided paths to NetCDF files.
        - A list of paths to Parquet files.
        - A loaded pickle file.

    Raises:
        ValueError: If multiple pickle files provided, as only one is ever expected.
    """
    cubes = CubeList([])
    loaded_pickles = []
    parquets = []

    for file in files:
        file_paths = glob.glob(str(file))
        for file_path_str in file_paths:
            file_path = Path(file_path_str)
            if not file_path.exists():
                continue

            # Directories indicate we are working with parquet files.
            if file_path.is_dir():
                parquets.append(file_path)
                continue

            try:
                cube = load_cubelist(str(file_path))
                cubes.extend(cube)
            except ValueError:
                try:
                    loaded_pickles.append(joblib.load(file_path))
                except Exception as e:
                    msg = f"Failed to load {file_path}: {e}"
                    raise ValueError(msg)

    if len(loaded_pickles) > 1:
        msg = "Multiple pickle inputs have been provided. Only one is expected."
        raise ValueError(msg)

    return (
        cubes if cubes else None,
        parquets if parquets else None,
        loaded_pickles[0] if loaded_pickles else None,
    )


def identify_parquet_type(parquet_paths: List[Path]):
    """Determine whether the provided parquet paths contain forecast or truth data.
    This is done by checking the columns within the parquet files for the presence
    of a forecast_period column which is only present for forecast data.

    Args:
        parquet_paths:
            A list of paths to Parquet files.


    Returns:
        - The path to the Parquet file containing the historical forecasts.
        - The path to the Parquet file containing the truths.
    """
    # import here to avoid dependency on pyarrow for all of improver
    import pyarrow.parquet as pq

    forecast_table_path = None
    truth_table_path = None
    for file_path in parquet_paths:
        try:
            example_file_path = next(file_path.glob("**/*.parquet"))
        except StopIteration:
            continue
        try:
            pq.read_schema(example_file_path).field("forecast_period")
            forecast_table_path = file_path
        except KeyError:
            truth_table_path = file_path

    return forecast_table_path, truth_table_path


def split_cubes_for_samos(
    cubes: CubeList,
    gam_features: List[str],
    truth_attribute: Optional[str] = None,
    expect_emos_coeffs: bool = False,
    expect_emos_fields: bool = False,
):
    """Function to split the forecast, truth, gam additional predictors and emos
    additional predictor cubes.

    Args:
        cubes:
            A list of input cubes which will be split into relevant groups.
        gam_features:
            A list of strings containing the names of the additional fields
            required for the SAMOS GAMs.
        truth_attribute:
            An attribute and its value in the format of "attribute=value",
            which must be present on truth cubes. If None, no truth cubes are
            expected or returned.
        expect_emos_coeffs:
            If True, EMOS coefficient cubes are expected to be found in the input
            cubes. If False, an error will be raised if any such cubes are found.
        expect_emos_fields:
            If True, additional EMOS fields are expected to be found in the input
            cubes. If False, an error will be raised if any such cubes are found.

    Raises:
        IOError:
            If no forecast cube is found and/or no truth cube is found when a
            truth_attribute has been provided.
        IOError:
            If EMOS coefficients cubes are found when they are not expected.
        IOError:
            If additional fields cubes are found which do not match the features in
            gam_features.
        IOError:
            If probability cubes are provided with more than one name.

    Returns:
        - A cube containing all the historic forecasts, or None if no such cubes
          were found.
        - A cube containing all the truth data, or None if no such cubes were found
          or no truth_attribute was provided.
        - A cubelist containing all the additional fields required for the GAMs,
          or None if no such cubes were found.
        - A cubelist containing all the EMOS coefficient cubes, or None if no such
          cubes were found.
        - A cubelist containing all the additional fields required for EMOS,
          or None if no such cubes were found.
        - A cube containing a probability template, or None if no such cube is found.
    """
    forecast = iris.cube.CubeList([])
    truth = iris.cube.CubeList([])
    gam_additional_fields = iris.cube.CubeList([])
    emos_coefficients = iris.cube.CubeList([])
    emos_additional_fields = iris.cube.CubeList([])
    prob_template = None

    # Prepare variables used to split forecast and truth.
    truth_key, truth_value = None, None
    if truth_attribute:
        truth_key, truth_value = truth_attribute.split("=")

    for cube in flatten(cubes):
        if "time" in [c.name() for c in cube.coords()]:
            if truth_key and cube.attributes.get(truth_key) == truth_value:
                truth.append(cube.copy())
            else:
                forecast.append(cube.copy())
        elif "emos_coefficient" in cube.name():
            emos_coefficients.append(cube.copy())
        elif cube.name() in gam_features:
            gam_additional_fields.append(cube.copy())
        else:
            emos_additional_fields.append(cube.copy())

    # Check that all required inputs are present and no unexpected cubes have been
    # found.
    missing_inputs = []
    if len(forecast) == 0:
        missing_inputs.append("forecast")
    if truth_key and len(truth) == 0:
        missing_inputs.append("truth")
    if missing_inputs:
        raise IOError(f"Missing {' and '.join(missing_inputs)} input.")

    if not expect_emos_coeffs and len(emos_coefficients) > 0:
        msg = (
            f"Found EMOS coefficients cubes when they were not expected. The following "
            f"such cubes were found: {[c.name() for c in emos_coefficients]}."
        )
        raise IOError(msg)

    if not expect_emos_fields and len(emos_additional_fields) > 0:
        msg = (
            f"Found additional fields cubes which do not match the features in "
            f"gam_features. The following cubes were found: "
            f"{[c.name() for c in emos_additional_fields]}."
        )
        raise IOError(msg)

    # Split out prob_template cube if required.
    forecast_names = [c.name() for c in forecast]
    prob_forecast_names = [name for name in forecast_names if "probability" in name]
    if len(set(prob_forecast_names)) > 1:
        msg = (
            "Providing multiple probability cubes is not supported. A probability cube "
            "can either be provided as the forecast or the probability template, but "
            f"not both. Cubes provided: {prob_forecast_names}."
        )
        raise IOError(msg)
    else:
        if len(set(forecast_names)) > 1:
            prob_template = forecast.extract(prob_forecast_names[0])[0]
            forecast.remove(prob_template)

    forecast = MergeCubes()(forecast)
    if truth_key:
        truth = MergeCubes()(truth)

    return (
        None if not forecast else forecast,
        None if not truth else truth,
        None if not gam_additional_fields else gam_additional_fields,
        None if not emos_coefficients else emos_coefficients,
        None if not emos_additional_fields else emos_additional_fields,
        prob_template,
    )


def validity_time_check(forecast: Cube, validity_times: List[str]) -> bool:
    """Check the validity time of the forecast matches the accepted validity times
    within the validity times list.

    Args:
        forecast:
            Cube containing the forecast to be calibrated.
        validity_times:
            Times at which the forecast must be valid. This must be provided
            as a four digit string (HHMM) where the first two digits represent the hour
            and the last two digits represent the minutes e.g. 0300 or 0315. If the
            forecast provided is at a different validity time then no coefficients
            will be applied.

    Returns:
        If the validity time within the cube matches a validity time within the
        validity time list, then True is returned. Otherwise, False is returned.
    """
    point = forecast.coord("time").cell(0).point
    if f"{point.hour:02}{point.minute:02}" not in validity_times:
        return False
    return True


def add_warning_comment(forecast: Cube) -> Cube:
    """Add a comment to warn that calibration has not been applied.

    Args:
        forecast: The forecast to which a comment will be added.

    Returns:
        Forecast with an additional comment.
    """
    if forecast.attributes.get("comment", None):
        forecast.attributes["comment"] = forecast.attributes["comment"] + (
            "\nWarning: Calibration of this forecast has been attempted, "
            "however, no calibration has been applied."
        )
    else:
        forecast.attributes["comment"] = (
            "Warning: Calibration of this forecast has been attempted, "
            "however, no calibration has been applied."
        )
    return forecast


def get_common_wmo_ids(
    forecast_cube: Cube,
    truth_cube: Cube,
    additional_predictors: Optional[CubeList] = None,
) -> Tuple[Cube, Cube, CubeList]:
    """Extracts the common WMO IDs from the forecast, truth and any additional
    predictor cubes.

    Args:
        forecast_cube: Cube containing the forecast data.
        truth_cube: Cube containing the truth data.
        additional_predictors: CubeList containing any additional predictors.

    Raises:
        IOError: If no common WMO IDs are found in the input cubes.

    Returns:
        The forecast, truth and additional predictor cubes with only the common
        WMO IDs retained.
    """
    wmo_ids = []
    wmo_ids.append(forecast_cube.coord("wmo_id").points)
    wmo_ids.append(truth_cube.coord("wmo_id").points)
    if additional_predictors is not None:
        for ap in additional_predictors:
            wmo_ids.append(ap.coord("wmo_id").points)
    common_wmo_ids = list(set.intersection(*map(set, wmo_ids)))
    if not common_wmo_ids:
        raise IOError("No common WMO IDs found in the input cubes.")
    constr = iris.Constraint(wmo_id=common_wmo_ids)
    truth_cube = truth_cube.extract(constr)
    forecast_cube = forecast_cube.extract(constr)
    if additional_predictors is not None:
        additional_predictors = additional_predictors.extract(constr)
    return forecast_cube, truth_cube, additional_predictors


def get_training_period_cycles(
    cycletime: str, forecast_period: Union[int, str], training_length: int
):
    """Generate a list of forecast reference times for the training period.

    Args:
        cycletime: The time at which the forecast is issued in a format understood by
            pandas.Timestamp e.g. 20170109T0000Z.
        forecast_period: The forecast period in seconds.
        training_length: The number of days in the training period.
    """
    forecast_period_td = pd.Timedelta(int(forecast_period), unit="seconds")

    return pd.date_range(
        end=pd.Timestamp(cycletime)
        - pd.Timedelta(1, unit="days")
        - forecast_period_td.floor("D"),
        periods=int(training_length),
        freq="D",
    )


def add_static_feature_from_cube_to_df(
    forecast_df: pd.DataFrame,
    feature_cube: iris.cube.Cube,
    feature_name: str,
    possible_merge_columns: list[str],
    float_decimals: int = 4,
) -> pd.DataFrame:
    """Add a static feature to the forecast DataFrame from a cube based on the
    feature configuration. Other features are expected to already be present in the
    forecast DataFrame. Columns within possible_merge_columns that are float after
    converting from a Cube to a DataFrame, are rounded to a specified number of
    decimal places before merging to avoid precision issues.

    Args:
        forecast_df: DataFrame containing the forecast data.
        cube_inputs: List of cubes containing additional features.
        feature_name: Name of the feature to be added.
        possible_merge_columns: List of column names that can be used to merge
            the feature DataFrame to the forecast DataFrame.
        float_decimals: Number of decimal places to round float columns to
            before merging. Default is 4, which corresponds to rounding to
            0.0001.
    Returns:
        DataFrame with additional feature added from the input cubes.
    """
    feature_df = as_data_frame(feature_cube, add_aux_coords=True).reset_index()

    forecast_df = add_feature_from_df_to_df(
        forecast_df, feature_df, feature_name, possible_merge_columns, float_decimals
    )
    return forecast_df


def add_feature_from_df_to_df(
    forecast_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_name: str,
    possible_merge_columns: list[str],
    float_decimals: int = 4,
):
    """Add a feature to the forecast DataFrame from a second DataFrame based on the
    feature configuration. Columns within possible_merge_columns that are float are
    rounded to a specified number of decimal places before merging to avoid
    precision issues.

    Args:
        forecast_df: DataFrame containing the forecast data.
        feature_df: DataFrame containing the feature data.
        feature_name: Name of the feature to be added.
        possible_merge_columns: List of column names that can be used to merge
            the feature DataFrame to the forecast DataFrame.
        float_decimals: Number of decimal places to round float columns to
            before merging. Default is 4, which corresponds to rounding to
            0.0001.
    Returns:
        DataFrame with additional feature added.
    """
    merge_columns = [col for col in possible_merge_columns if col in feature_df.columns]

    # Select the required DataFrame subset using the merge_columns and dtypes.
    # Columns with any NaNs can not be converted to integers, and therefore are left
    # unmodified.
    float_subset = feature_df[merge_columns].select_dtypes(
        include=[np.float32, np.float64]
    )
    float_subset = float_subset[float_subset.columns[~float_subset.isnull().any()]]

    float_cols = list(set(float_subset.columns).intersection(set(forecast_df.columns)))
    original_dtypes = forecast_df[float_cols].dtypes

    # Scale float columns to integers for both DataFrames to avoid precision issues
    multiplier = 10**float_decimals
    feature_df[float_cols] = np.round(feature_df[float_cols] * multiplier).astype(int)
    forecast_df[float_cols] = np.round(forecast_df[float_cols] * multiplier).astype(int)

    forecast_df = forecast_df.merge(
        feature_df[merge_columns + [feature_name]],
        on=merge_columns,
        how="left",
    )
    # Revert float columns to original dtypes and scale back to original values.
    for col, dtype in zip(float_cols, original_dtypes):
        forecast_df[col] = forecast_df[col].astype(dtype) / multiplier
    return forecast_df
