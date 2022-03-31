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
"""
Functionality to convert a pandas DataFrame in the expected format
into an iris cube.

.. Further information is available in:
.. include:: extended_documentation/calibration/calibration_data_ingestion.rst


"""
from heapq import merge
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList
from matplotlib.pyplot import get
from pandas.core.frame import DataFrame
from pandas.core.indexes.datetimes import DatetimeIndex

from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    RebadgePercentilesAsRealizations,
)
from improver.ensemble_copula_coupling.utilities import (
    choose_set_of_percentiles,
    create_cube_with_percentiles,
)
from improver.metadata.constants.time_types import TIME_COORDS
from improver.spotdata.build_spotdata_cube import build_spotdata_cube

ALT_PERCENTILE_COLUMNS = [
    "percentile",
    "realization",
    "threshold",
]

FORECAST_DATAFRAME_COLUMNS = [
    "altitude",
    "blend_time",
    "cf_name",
    "diagnostic",
    "experiment",
    "forecast",
    "forecast_period",
    "forecast_reference_time",
    "height",
    "latitude",
    "longitude",
    "period",
    "time",
    "units",
    "wmo_id",
]

TRUTH_DATAFRAME_COLUMNS = [
    "altitude",
    "diagnostic",
    "latitude",
    "longitude",
    "ob_value",
    "time",
    "wmo_id",
]


def _dataframe_column_check(df: DataFrame, compulsory_columns: Sequence) -> None:
    """Check that the compulsory columns are present on the DataFrame.
    Any other columns within the DataFrame are ignored.

    Args:
        df:
            Dataframe expected to contain the compulsory columns.
        compulsory_columns:
            The names of the compulsory columns.

    Raises:
        ValueError: Raise an error if a compulsory column is missing.
    """
    if not set(compulsory_columns).issubset(df.columns):
        diff = set(compulsory_columns).difference(df.columns)
        msg = (
            "The following compulsory column(s) are missing from the "
            f"dataframe: {diff}"
        )
        raise ValueError(msg)


def _preprocess_temporal_columns(df: DataFrame) -> DataFrame:
    """Pre-process the columns with temporal dtype to convert
    from numpy datetime objects to pandas datetime objects.
    Casting the dtype of the columns to object type results
    in columns of dtype "object" with the contents of the
    columns being pandas datetime objects, rather than numpy
    datetime objects.

    Args:
        df:
            A DataFrame with temporal columns with numpy
            datetime dtypes.

    Returns:
        A DataFrame without numpy datetime dtypes. The
        content of the columns with temporal dtypes are
        accessible as pandas datetime objects.
    """
    for col in df.select_dtypes(include=["datetime64[ns, UTC]"]):
        df = df.astype({col: "O"})
    for col in df.select_dtypes(include="timedelta64[ns]"):
        df = df.astype({col: "O"})
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


def _quantile_check(df: DataFrame) -> None:
    """Check that the percentiles provided can be considered to be
    quantiles with equal spacing spanning the percentile range.

    Args:
        df: DataFrame with a percentile column.

    Raises:
        ValueError: Percentiles are not equally spaced.
    """
    expected_percentiles = choose_set_of_percentiles(df["percentile"].nunique())

    if not np.allclose(expected_percentiles, df["percentile"].unique()):
        msg = (
            "The forecast percentiles can not be considered as quantiles. "
            f"The forecast percentiles are {df['percentile'].unique()}."
            "Based on the number of percentiles provided, the expected "
            f"percentiles would be {expected_percentiles}."
        )
        raise ValueError(msg)


def _define_time_coord(
    adate: pd.Timestamp, time_bounds: Optional[Sequence[pd.Timestamp]] = None,
) -> DimCoord:
    """Define a time coordinate. The coordinate will have bounds,
    if bounds are provided.

    Args:
        adate:
            The point for the time coordinate.
        time_bounds:
            The values defining the bounds for the time coordinate.

    Returns:
        A time coordinate. This coordinate will have bounds, if bounds
        are provided.
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
    dataset. The final validity time within the training dataset is
    at least one day prior to the cycletime. The final validity time
    within the training dataset is additionally offset by the number
    of days within the forecast period to ensure that the dates defined
    by the training dataset are in the past relative to the cycletime.
    For example, for a cycletime of 20170720T0000Z with a forecast period
    of T+30 and a training length of 3 days, the validity time is
    20170721T0600Z. Subtracting one day gives 20170720T0600Z. Note that
    this is in the future relative to the cycletime and we want the
    training dates to be in the past relative to the cycletime.
    Subtracting the forecast period rounded down to the nearest day for
    T+30 gives 1 day. Subtracting this additional day gives 20170719T0600Z.
    This is the final validity time within the training period. We then
    compute the validity times for a 3 day training period using 20170719T0600Z
    as the final validity time giving 20170719T0600Z, 20170718T0600Z
    and 20170717T0600Z.

    Args:
        cycletime:
            Cycletime of a format similar to 20170109T0000Z.
            The training dates will always be in the past, relative
            to the cycletime.
        forecast_period:
            Forecast period in seconds as an integer.
        training_length:
            Training length in days as an integer.

    Returns:
        Datetimes defining the training dataset. The number of datetimes
        is equal to the training length.
    """
    forecast_period = pd.Timedelta(int(forecast_period), unit="seconds")
    validity_time = pd.Timestamp(cycletime) + forecast_period
    return pd.date_range(
        end=validity_time - pd.Timedelta(1, unit="days") - forecast_period.floor("D"),
        periods=int(training_length),
        freq="D",
        tz="UTC",
    )


def get_var_type(df: DataFrame) -> str:
    """Check which of ALT_PERCENTILE_COLUMNS exists in the dataframe.
    
    Args:
        df:
            DataFrame expected to contain exactly one of ALT_PERCENTILE_COLUMNS.
    
    Returns:
        var_type:
            The member of ALT_PERCENTILE_COLUMNS found in the dataframe columns.

    Raises:
        ValueError:
            If none of the allowed columns are present, or more than one is present.
    """
    var_type = None
    for variable in ALT_PERCENTILE_COLUMNS:
        if variable in df.columns:
            if var_type is not None:
                msg = f"More than one column of {ALT_PERCENTILE_COLUMNS} exists in the input dataset"
                raise ValueError(msg)
            var_type = variable
        else:
            continue

    # check if one of the data-columns was found
    if var_type is None:
        msg = (
            f"None of the columns {ALT_PERCENTILE_COLUMNS} exists in the input dataset"
        )
        raise ValueError(msg)

    return var_type


def _prepare_dataframes(
    forecast_df: DataFrame,
    truth_df: DataFrame,
    percentiles: Optional[List[float]] = None,
    var_subset: Optional[List[float]] = None,
    experiment: Optional[str] = None,
) -> Tuple[DataFrame, DataFrame]:
    """Prepare dataframes for conversion to cubes by: 1) checking
    that the expected columns are present, 2) checking the percentiles
    are as expected, 3) removing duplicates from the forecast and truth,
    4) finding the sites common to both the forecast and truth dataframes
    and 5) replacing and supplementing the truth dataframe with
    information from the forecast dataframe. Note that this third
    step will also ensure that a row containing a NaN for the
    ob_value is inserted for any missing observations.

    Args:
        forecast_df:
            DataFrame expected to contain the following columns: forecast,
            blend_time, forecast_period, forecast_reference_time, time,
            wmo_id, percentile, diagnostic, latitude, longitude, altitude,
            period, height, cf_name, units and experiment. Any other
            columns are ignored.
        truth_df:
            DataFrame expected to contain the following columns: ob_value,
            time, wmo_id, diagnostic, latitude, longitude and altitude.
            Any other columns are ignored.
        percentiles:
            The set of percentiles to be used for estimating EMOS coefficients.
        var_subet:
            The set of realizations/probabilites to be used
        experiment:
            A value within the experiment column to select from the forecast
            table.

    Returns:
        A sanitised version of the forecasts and truth dataframes that
        are ready for conversion to cubes.
    """

    var_type = get_var_type(forecast_df)

    _dataframe_column_check(forecast_df, FORECAST_DATAFRAME_COLUMNS + [var_type])
    _dataframe_column_check(truth_df, TRUTH_DATAFRAME_COLUMNS)

    # Filter to select only one experiment
    if experiment:
        forecast_df = forecast_df.loc[forecast_df["experiment"] == experiment]

    if forecast_df["experiment"].nunique() > 1:
        unique_exps = forecast_df["experiment"].unique()
        msg = (
            "More than one value for the experiment column found in the "
            f"forecast dataframe. Values for experiment column {unique_exps}"
        )
        raise ValueError(msg)

    # Extract the required percentiles.
    if percentiles:
        indices = [np.isclose(forecast_df["percentile"], float(p)) for p in percentiles]
        forecast_df = forecast_df[np.logical_or.reduce(indices)]

    if var_subset:
        indices = [np.isclose(forecast_df[var_type], float(p)) for p in var_subset]
        forecast_df = forecast_df[np.logical_or.reduce(indices)]

    # Check the percentiles can be considered to be equally space quantiles.
    if var_type == "percentile":
        _quantile_check(forecast_df)

    # Remove forecast duplicates.
    forecast_cols = ["diagnostic", "forecast_period", var_type, "time", "wmo_id"]
    if "station_id" in forecast_df.columns:
        forecast_cols.append("station_id")
    forecast_df = forecast_df.drop_duplicates(subset=forecast_cols, keep="last",)
    # Sort to ensure a consistent ordering after removing duplicates.
    sort_cols = ["blend_time", var_type, "wmo_id"]
    if "station_id" in forecast_df:
        sort_cols.append("station_id")
    forecast_df.sort_values(
        by=sort_cols, inplace=True, ignore_index=True,
    )

    # Remove truth duplicates.
    truth_cols = ["diagnostic", "time", "wmo_id"]
    if "station_id" in truth_df.columns:
        truth_cols.append("station_id")
    truth_df = truth_df.drop_duplicates(subset=truth_cols, keep="last",)
    # Sort to ensure a consistent ordering after removing duplicates.
    truth_df = truth_df.sort_values(by=truth_cols, ignore_index=True)

    # Find the common set of WMO IDs.
    common_wmo_ids = sorted(
        set(forecast_df["wmo_id"].unique()).intersection(truth_df["wmo_id"].unique())
    )
    forecast_df = forecast_df[forecast_df["wmo_id"].isin(common_wmo_ids)]
    truth_df = truth_df[truth_df["wmo_id"].isin(common_wmo_ids)]

    if ("station_id" in forecast_df.columns) and ("station_id" in truth_df.columns):
        # Find the common set of WMO IDs.
        common_station_ids = sorted(
            set(forecast_df["station_id"].unique()).intersection(
                truth_df["station_id"].unique()
            )
        )
        forecast_df = forecast_df[forecast_df["station_id"].isin(common_station_ids)]
        truth_df = truth_df[truth_df["station_id"].isin(common_station_ids)]

    # Ensure time in forecasts is present in truths.
    forecast_df = forecast_df[forecast_df["time"].isin(truth_df["time"].unique())]

    # Ensure time in truths is present in forecasts.
    truth_df = truth_df[truth_df["time"].isin(forecast_df["time"].unique())]

    truth_df = truth_df.drop(columns=["altitude", "latitude", "longitude"])
    # Identify columns to copy onto the truth_df from the forecast_df
    subset_cols = [
        "wmo_id",
        "latitude",
        "longitude",
        "altitude",
        "period",
        "height",
        "cf_name",
        "units",
        "time",
        "forecast_reference_time",
        "diagnostic",
    ]
    if "station_id" in forecast_df.columns:
        subset_cols.append("station_id")
    if var_type == "threshold":
        subset_cols.append("threshold")
    forecast_subset = forecast_df[subset_cols].drop_duplicates()

    # Use "right" to fill in any missing observations in the truth dataframe
    # and retain the order from the forecast_subset.
    merge_cols = ["wmo_id", "time", "diagnostic"]
    if ("station_id" in forecast_df.columns) and ("station_id" in truth_df.columns):
        merge_cols.append("station_id")
    truth_df = truth_df.merge(forecast_subset, on=merge_cols, how="right")
    return forecast_df, truth_df


def forecast_dataframe_to_cube(
    df: DataFrame, training_dates: DatetimeIndex, forecast_period: int,
) -> Cube:
    """Convert a forecast DataFrame into an iris Cube. The percentiles
    within the forecast DataFrame are rebadged as realizations.

    Args:
        df:
            DataFrame expected to contain the following columns: forecast,
            blend_time, forecast_period, forecast_reference_time, time,
            wmo_id, percentile (or realization/probability), diagnostic, latitude, longitude, period,
            height, cf_name, units. Any other columns are ignored.
        training_dates:
            Datetimes spanning the training period.
        forecast_period:
            Forecast period in seconds as an integer.

    Returns:
        Cube containing the forecasts from the training period.
    """
    fp_point = pd.Timedelta(int(forecast_period), unit="seconds")

    cube_list_2d = []

    var_type = get_var_type(df)
    if var_type == "percentile":
        unit = "%"
    else:
        unit = "1"
    if var_type == "realization":
        datatype = np.int32
    else:
        datatype = np.float32

    for adate in training_dates:
        time_df = df.loc[(df["time"] == adate) & (df["forecast_period"] == fp_point)]

        time_df = _preprocess_temporal_columns(time_df)
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

        var_cubelist = []
        for var_val in sorted(time_df[var_type].unique()):
            var_coord = DimCoord(datatype(var_val), long_name=var_type, units=unit)
            # rename so that we populate the standard name if possible
            var_coord.rename(var_type)
            var_df = time_df.loc[time_df[var_type] == var_val]
            cube = build_spotdata_cube(
                var_df["forecast"].astype(np.float32),
                var_df["cf_name"].values[0],
                var_df["units"].values[0],
                var_df["altitude"].astype(np.float32),
                var_df["latitude"].astype(np.float32),
                var_df["longitude"].astype(np.float32),
                var_df["wmo_id"].values.astype("U5"),
                scalar_coords=[
                    time_coord,
                    frt_coord,
                    fp_coord,
                    var_coord,
                    height_coord,
                ],
            )
            var_cubelist.append(cube)
        cube_list_2d.append(var_cubelist)

    if not cube_list_2d:
        return
    time_list = CubeList()
    for var_list in zip(*cube_list_2d):
        var_cube = CubeList(list(var_list)).merge_cube()
        time_list.append(var_cube)
    cube = time_list.merge_cube()

    cube.attributes["cube_type"] = "forecast"
    if var_type == "percentile":
        return RebadgePercentilesAsRealizations()(cube)
    else:
        return cube


def truth_dataframe_to_cube(
    df: DataFrame, training_dates: DatetimeIndex, var_type: str = "percentile"
) -> Cube:
    """Convert a truth DataFrame into an iris Cube.

    Args:
        df:
            DataFrame expected to contain the following columns: ob_value,
            time, wmo_id, diagnostic, latitude, longitude, altitude, cf_name,
            height, period and units. Any other columns are ignored.
        training_dates:
            Datetimes spanning the training period.
        var_type:
            One of ALT_PERCENTILE_COLUMNS
    Returns:
        Cube containing the truths from the training period.
    """
    cubelist = CubeList()
    for adate in training_dates:
        time_df = df.loc[(df["time"] == adate)]

        time_df = _preprocess_temporal_columns(time_df)

        if time_df.empty:
            continue

        # The following columns are expected to contain one unique value
        # per column.
        _unique_check(time_df, "diagnostic")

        if time_df["period"].isna().all():
            time_bounds = None
        else:
            period = time_df["period"].values[0]
            time_bounds = [adate - period, adate]

        time_coord = _define_time_coord(adate, time_bounds)
        height_coord = _define_height_coord(time_df["height"].values[0])

        if var_type == "threshold":
            for var_val in sorted(df[var_type].unique()):
                var_coord = DimCoord(np.float32(var_val), long_name=var_type, units=1)
                var_df = time_df.loc[time_df[var_type] == var_val]
                cube = build_spotdata_cube(
                    (var_df["ob_value"] > var_val).astype(np.int32),
                    var_df["cf_name"].values[0],
                    var_df["units"].values[0],
                    var_df["altitude"].astype(np.float32),
                    var_df["latitude"].astype(np.float32),
                    var_df["longitude"].astype(np.float32),
                    var_df["wmo_id"].values.astype("U5"),
                    scalar_coords=[time_coord, height_coord, var_coord],
                )
                cubelist.append(cube)
        else:
            cube = build_spotdata_cube(
                time_df["ob_value"].astype(np.float32),
                time_df["cf_name"].values[0],
                time_df["units"].values[0],
                time_df["altitude"].astype(np.float32),
                time_df["latitude"].astype(np.float32),
                time_df["longitude"].astype(np.float32),
                time_df["wmo_id"].values.astype("U5"),
                scalar_coords=[time_coord, height_coord],
            )
            cubelist.append(cube)

    if not cubelist:
        return

    cube = cubelist.merge_cube()
    cube.attributes["cube_type"] = "observations"
    return cube


def forecast_and_truth_dataframes_to_cubes(
    forecast_df: DataFrame,
    truth_df: DataFrame,
    cycletime: str,
    forecast_period: int,
    training_length: int,
    percentiles: Optional[List[float]] = None,
    var_subset: Optional[List[float]] = None,
    experiment: Optional[str] = None,
) -> Tuple[Cube, Cube]:
    """Convert a forecast DataFrame into an iris Cube and a
    truth DataFrame into an iris Cube.

    Args:
        forecast_df:
            DataFrame expected to contain the following columns: forecast,
            blend_time, forecast_period, forecast_reference_time, time,
            wmo_id, percentile, diagnostic, latitude, longitude, period,
            height, cf_name, units. Any other columns are ignored.
        truth_df:
            DataFrame expected to contain the following columns: ob_value,
            time, wmo_id, diagnostic, latitude, longitude and altitude.
            Any other columns are ignored.
        cycletime:
            Cycletime of a format similar to 20170109T0000Z.
        forecast_period:
            Forecast period in seconds as an integer.
        training_length:
            Training length in days as an integer.
        percentiles:
            The set of percentiles to be used for estimating EMOS coefficients.
            These should be a set of equally spaced quantiles.
        var_subset:
            The set of realizations/probailities used 
        experiment:
            A value within the experiment column to select from the forecast
            table.


    Returns:
        Forecasts and truths for the training period in Cube format.
    """
    training_dates = _training_dates_for_calibration(
        cycletime, forecast_period, training_length
    )

    forecast_df, truth_df = _prepare_dataframes(
        forecast_df, truth_df, percentiles=percentiles, experiment=experiment
    )

    forecast_cube = forecast_dataframe_to_cube(
        forecast_df, training_dates, forecast_period
    )
    var_type = get_var_type(forecast_df)
    truth_cube = truth_dataframe_to_cube(truth_df, training_dates, var_type)
    return forecast_cube, truth_cube
