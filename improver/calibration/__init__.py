# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""init for calibration that contains functionality to split forecast and truth
inputs, and functionality to convert a pandas DataFrame in the expected format
into an iris cube.

.. Further information is available in:
.. include:: extended_documentation/calibration/calibration_data_ingestion.rst

"""

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import iris
import numpy as np
import pandas as pd
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList
from numpy import timedelta64
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import DatetimeIndex

from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    RebadgePercentilesAsRealizations,
)
from improver.metadata.constants.time_types import TIME_COORDS
from improver.metadata.probabilistic import (
    get_diagnostic_cube_name_from_probability_name,
)
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.utilities.cube_manipulation import MergeCubes

FORECAST_DATAFRAME_COLUMNS = [
    "forecast",
    "blend_time",
    "forecast_period",
    "forecast_reference_time",
    "time",
    "wmo_id",
    "percentile",
    "diagnostic",
    "latitude",
    "longitude",
    "period",
    "height",
    "cf_name",
    "units",
]

TRUTH_DATAFRAME_COLUMNS = [
    "ob_value",
    "time",
    "wmo_id",
    "diagnostic",
    "latitude",
    "longitude",
    "altitude",
]


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
    cubes: CubeList, land_sea_mask_name: Optional[str] = None,
):
    """Split the input forecast, coefficients, land sea-mask and
    probability template, if provided. The coefficients cubes and
    land-sea mask are identified based on their name. The current
    forecast and probability template are then split.

    Args:
        cubes:
            A list of input cubes which will be split into relevant groups.
            This includes the forecast, coefficients, land-sea mask and
            probability template.
        land_sea_mask_name:
            Name of the land-sea mask cube to help identification.

    Returns:
        - A cube containing the current forecast.
        - If found, a cubelist containing the coefficients else None.
        - If found, a land-sea mask will be returned, else None.
        - If found, a probability template will be returned, else None.

    Raises:
        ValueError: If multiple items provided, when only one is expected.
        ValueError: If no forecast is found.
    """
    coefficients = CubeList()
    land_sea_mask = None
    grouped_cubes: Dict[str, List[Cube]] = {}

    for cubelist in cubes:
        for cube in cubelist:
            if "emos_coefficient" in cube.name():
                coefficients.append(cube)
            elif land_sea_mask_name and cube.name() == land_sea_mask_name:
                land_sea_mask = cube
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
    return (
        current_forecast,
        coefficients,
        land_sea_mask,
        prob_template,
    )


def _dataframe_column_check(df: DataFrame, compulsory_columns: Sequence) -> None:
    """Check that the compulsory columns are present on the DataFrame.

    Args:
        df:
            Dataframe expected to contain the compulsory columns.
        compulsory_columns:
            The names of the compulsory columns.

    Raises:
        ValueError: Raise an error if a compulsory column is missing.
    """
    if not set(compulsory_columns).issubset(df.columns):
        diff = set(compulsory_columns).symmetric_difference(df.columns)
        msg = (
            "The following compulsory column(s) are missing from the "
            f"dataframe: {diff}"
        )
        raise ValueError(msg)


def _preprocess_temporal_columns(df: DataFrame) -> DataFrame:
    """Pre-process the columns with temporal dtype to convert
    from numpy datetime objects to pandas datetime objects.

    Args:
        df:
            A DataFrame with temporal columns with numpy
            datetime dtypes.

    Returns:
        A DataFrame without numpy datetime dtypes.
    """
    for col in [c for c in df.columns if df[c].dtype == "datetime64[ns]"]:
        df[col] = df[col].dt.tz_localize("UTC").astype("O")
    for col in [c for c in df.columns if df[c].dtype == "timedelta64[ns]"]:
        df[col] = df[col].astype("O")
    return df


def _unique_check(df: DataFrame, column: str) -> None:
    """Check whether the values in the column are unique.

    Args:
        df:
            The DataFrame to be checked.
        column:
            Name of a column in the DataFrame.

    Raises:
        ValueError: Only one unique value within the specifed column
            is expected.
    """
    if df[column].nunique(dropna=False) > 1:
        msg = (
            f"Multiple values provided for the {column}: "
            f"{df[column].unique()}. "
            f"Only one value for the {column} is expected."
        )
        raise ValueError(msg)


def _define_time_coord(
    adate: pd.Timestamp, time_bounds: Optional[Sequence[pd.Timestamp]] = None,
) -> DimCoord:
    """Define a time coordinate with bounds is provided.

    Args:
        adate:
            The point for the time coordinate.
        time_bounds:
            The values defining the bounds for the time coordinate.

    Returns:
        A time
    """
    return DimCoord(
        np.array(adate.timestamp(), dtype=TIME_COORDS["time"].dtype),
        "time",
        bounds=time_bounds
        if time_bounds is None
        else [
            np.array(t.timestamp(), dtype=TIME_COORDS["time"].dtype)
            for t in time_bounds
        ],
        units=TIME_COORDS["time"].units,
    )


def _define_height_coord(height) -> AuxCoord:
    """Define a height coordinate. A unit of metres is assumed.

    Args:
        height:
            The value for the height coordinate in metres.

    Returns:
        The height coordinate.
    """
    return AuxCoord(np.array(height, dtype=np.float32), "height", units="m",)


def _training_dates_for_calibration(
    cycletime: str, forecast_period: int, training_length: int
) -> DatetimeIndex:
    """Compute the date range required for extracting the required training
    dataset.

    Args:
        cycletime:
            Cycletime of a format similar to 20170109T0000Z.
        forecast_period:
            Forecast period in hours as an integer.
        training_length:
            Training length in days as an integer.

    Returns:
        Datetimes ending one day prior to the computed validity time. The number
        of datetimes is equal to the training length.
    """
    forecast_period = pd.Timedelta(int(forecast_period), unit="hours")
    validity_time = pd.Timestamp(cycletime) + forecast_period
    return pd.date_range(
        end=validity_time - pd.Timedelta(1, unit="days") - forecast_period.floor("D"),
        periods=int(training_length),
        freq="D",
        tz="UTC",
    )


def forecast_dataframe_to_cube(
    df: DataFrame, training_dates: DatetimeIndex, forecast_period: int
) -> Cube:
    """Convert a forecast DataFrame into an iris Cube. The percentiles
    within the forecast DataFrame are rebadged as realizations.

    Args:
        df:
            DataFrame expected to contain the following columns: forecast,
            blend_time, forecast_period, forecast_reference_time, time,
            wmo_id, percentile, diagnostic, latitude, longitude, period,
            height, cf_name, units.
        training_dates:
            Datetimes spanning the training period.
        forecast_period:
            Forecast period in hours as an integer.

    Returns:
        Cube containing the forecasts from the training period.
    """
    _dataframe_column_check(df, FORECAST_DATAFRAME_COLUMNS)
    df = _preprocess_temporal_columns(df)

    fp_point = pd.Timedelta(int(forecast_period), unit="hours")

    cubelist = CubeList()

    for adate in training_dates:
        time_df = df.loc[(df["time"] == adate) & (df["forecast_period"] == fp_point)]

        if time_df.empty:
            continue

        # The following columns are expected to contain one unique value
        # per column.
        for col in ["period", "height", "cf_name", "units", "diagnostic"]:
            _unique_check(time_df, col)

        if time_df["period"].isna().all():
            time_bounds = None
            fp_bounds = None
        else:
            period = time_df["period"].values[0]
            time_bounds = [adate - period, adate]
            fp_bounds = [fp_point - period, fp_point]

        time_coord = _define_time_coord(adate, time_bounds)
        height_coord = _define_height_coord(time_df["height"].values[0])

        fp_coord = AuxCoord(
            np.array(
                fp_point.total_seconds(), dtype=TIME_COORDS["forecast_period"].dtype
            ),
            "forecast_period",
            bounds=fp_bounds
            if fp_bounds is None
            else [
                np.array(f.total_seconds(), dtype=TIME_COORDS["forecast_period"].dtype)
                for f in fp_bounds
            ],
            units=TIME_COORDS["forecast_period"].units,
        )
        frt_coord = AuxCoord(
            np.array(
                time_df["forecast_reference_time"].values[0].timestamp(),
                dtype=TIME_COORDS["forecast_reference_time"].dtype,
            ),
            "forecast_reference_time",
            units=TIME_COORDS["forecast_reference_time"].units,
        )

        for percentile in sorted(df["percentile"].unique()):
            perc_coord = DimCoord(
                np.float32(percentile), long_name="percentile", units="%"
            )
            perc_df = time_df.loc[time_df["percentile"] == percentile]

            cube = build_spotdata_cube(
                perc_df["forecast"].astype(np.float32),
                perc_df["cf_name"].values[0],
                perc_df["units"].values[0],
                perc_df["altitude"].astype(np.float32),
                perc_df["latitude"].astype(np.float32),
                perc_df["longitude"].astype(np.float32),
                perc_df["wmo_id"].values.astype("U5"),
                scalar_coords=[
                    time_coord,
                    frt_coord,
                    fp_coord,
                    perc_coord,
                    height_coord,
                ],
            )
            cubelist.append(cube)

    if not cubelist:
        return
    cube = cubelist.merge_cube()

    return RebadgePercentilesAsRealizations()(cube)


def truth_dataframe_to_cube(
    df: DataFrame,
    training_dates: DatetimeIndex,
    period: timedelta64,
    height: float,
    cf_name: str,
    units: str,
) -> Cube:
    """Convert a truth DataFrame into an iris Cube.

    Args:
        df:
            DataFrame expected to contain the following columns: ob_value,
            time, wmo_id, diagnostic, latitude, longitude and altitude.
        training_dates:
            Datetimes spanning the training period.
        period:
            The period for defining the bounds on the time coordinate.
        height:
            The value of the height for defining a height coordinate.
        cf_name:
            The name of the resulting truth cube.
        units:
            The units of the resulting truth cube.

    Returns:
        Cube containing the truths from the training period.
    """
    _dataframe_column_check(df, TRUTH_DATAFRAME_COLUMNS)
    df = _preprocess_temporal_columns(df)

    cubelist = CubeList()
    for adate in training_dates:

        time_df = df.loc[df["time"] == adate]

        if time_df.empty:
            continue

        # The following columns are expected to contain one unique value
        # per column.
        _unique_check(time_df, "diagnostic")

        # Ensure that every WMO ID has an entry for a particular time.
        new_index = Index(df["wmo_id"].unique(), name="wmo_id")
        time_df = time_df.set_index("wmo_id").reindex(new_index)
        # Fill the alt/lat/lon with the mode to ensure consistent coordinates
        # to support merging. Also fill other columns known to contain one
        # unique value.
        for col in ["altitude", "latitude", "longitude", "diagnostic"]:
            time_df[col] = df.groupby(by="wmo_id", sort=False)[col].agg(
                lambda x: pd.Series.mode(x, dropna=not df[col].isna().all())
            )
        time_df = time_df.reset_index()

        if period is pd.Timedelta("NaT"):
            time_bounds = None
        else:
            time_bounds = [adate - period, adate]

        time_coord = _define_time_coord(adate, time_bounds)
        height_coord = _define_height_coord(height)

        cube = build_spotdata_cube(
            time_df["ob_value"].astype(np.float32),
            cf_name,
            units,
            time_df["altitude"].astype(np.float32),
            time_df["latitude"].astype(np.float32),
            time_df["longitude"].astype(np.float32),
            time_df["wmo_id"].values.astype("U5"),
            scalar_coords=[time_coord, height_coord],
        )
        cubelist.append(cube)

    if not cubelist:
        return
    return cubelist.merge_cube()


def _filter_forecasts_and_truths(forecast: Cube, truth: Cube) -> Tuple[Cube, Cube]:
    """Filter forecasts and truths to ensure consistent WMO IDs.

    Args:
        forecast:
            The training period forecasts.
        truth:
            The training period truths.

    Returns:
        Forecasts and truths for the training period that have been filtered to
        only include sites that are present in both the forecasts and truths.

    """
    forecast_constr = iris.Constraint(wmo_id=forecast.coord("wmo_id").points)
    truth_constr = iris.Constraint(wmo_id=truth.coord("wmo_id").points)
    forecast = forecast.extract(truth_constr)
    truth = truth.extract(forecast_constr)

    # As the lat/lon/alt of the observation can change, ensure that the
    # lat/lon/alt are consistent over the training period.
    for coord in ["altitude", "latitude", "longitude"]:
        truth.coord(coord).points = forecast.coord(coord).points

    return forecast, truth


def forecast_and_truth_dataframes_to_cubes(
    forecast_df: DataFrame,
    truth_df: DataFrame,
    cycletime: str,
    forecast_period: int,
    training_length: int,
) -> Tuple[Cube, Cube]:
    """Convert a truth DataFrame into an iris Cube.

    Args:
        forecast_df:
            DataFrame expected to contain the following columns: .
        truth_df:
            DataFrame expected to contain the following columns: .
        cycletime:
            Cycletime of a format similar to 20170109T0000Z.
        forecast_period:
            Forecast period in hours as an integer.
        training_length:
            Training length in days as an integer.

    Returns:
        Forecasts and truths for the training period in Cube format.
    """
    training_dates = _training_dates_for_calibration(
        cycletime, forecast_period, training_length
    )
    forecast_cube = forecast_dataframe_to_cube(
        forecast_df, training_dates, forecast_period
    )
    truth_cube = truth_dataframe_to_cube(
        truth_df,
        training_dates,
        period=forecast_df["period"].values[0],
        height=forecast_df["height"].values[0],
        cf_name=forecast_df["cf_name"].values[0],
        units=forecast_df["units"].values[0],
    )
    forecast_cube, truth_cube = _filter_forecasts_and_truths(forecast_cube, truth_cube)
    return forecast_cube, truth_cube
