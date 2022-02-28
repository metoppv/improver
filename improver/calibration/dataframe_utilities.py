# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2022 Met Office.
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
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList
from pandas.core.frame import DataFrame
from pandas.core.indexes.datetimes import DatetimeIndex

from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    RebadgePercentilesAsRealizations,
)
from improver.ensemble_copula_coupling.utilities import choose_set_of_percentiles
from improver.metadata.constants.time_types import TIME_COORDS
from improver.spotdata.build_spotdata_cube import build_spotdata_cube

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
    "percentile",
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


def _prepare_dataframes(
    forecast_df: DataFrame,
    truth_df: DataFrame,
    percentiles: Optional[List[float]] = None,
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
        experiment:
            A value within the experiment column to select from the forecast
            table.

    Returns:
        A sanitised version of the forecasts and truth dataframes that
        are ready for conversion to cubes.
    """
    _dataframe_column_check(forecast_df, FORECAST_DATAFRAME_COLUMNS)
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

    # Check the percentiles can be considered to be equally space quantiles.
    _quantile_check(forecast_df)

    # Remove forecast duplicates.
    forecast_df = forecast_df.drop_duplicates(
        subset=["diagnostic", "forecast_period", "percentile", "time", "wmo_id"],
        keep="last",
    )
    # Sort to ensure a consistent ordering after removing duplicates.
    forecast_df = forecast_df.sort_values(
        by=["blend_time", "percentile", "wmo_id"], ignore_index=True,
    )

    # Remove truth duplicates.
    truth_cols = ["diagnostic", "time", "wmo_id"]
    truth_df = truth_df.drop_duplicates(subset=truth_cols, keep="last",)
    # Sort to ensure a consistent ordering after removing duplicates.
    truth_df = truth_df.sort_values(by=truth_cols, ignore_index=True)

    # Find the common set of WMO IDs.
    common_wmo_ids = sorted(
        set(forecast_df["wmo_id"].unique()).intersection(truth_df["wmo_id"].unique())
    )
    forecast_df = forecast_df[forecast_df["wmo_id"].isin(common_wmo_ids)]
    truth_df = truth_df[truth_df["wmo_id"].isin(common_wmo_ids)]

    # Ensure time in forecasts is present in truths.
    forecast_df = forecast_df[forecast_df["time"].isin(truth_df["time"].unique())]

    # Ensure time in truths is present in forecasts.
    truth_df = truth_df[truth_df["time"].isin(forecast_df["time"].unique())]

    truth_df = truth_df.drop(columns=["altitude", "latitude", "longitude"])
    # Identify columns to copy onto the truth_df from the forecast_df
    forecast_subset = forecast_df[
        [
            "wmo_id",
            "latitude",
            "longitude",
            "altitude",
            "period",
            "height",
            "cf_name",
            "units",
            "time",
            "diagnostic",
        ]
    ].drop_duplicates()

    # Use "right" to fill in any missing observations in the truth dataframe
    # and retain the order from the forecast_subset.
    truth_df = truth_df.merge(
        forecast_subset, on=["wmo_id", "time", "diagnostic"], how="right"
    )
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
            wmo_id, percentile, diagnostic, latitude, longitude, period,
            height, cf_name, units. Any other columns are ignored.
        training_dates:
            Datetimes spanning the training period.
        forecast_period:
            Forecast period in seconds as an integer.

    Returns:
        Cube containing the forecasts from the training period.
    """
    fp_point = pd.Timedelta(int(forecast_period), unit="seconds")

    cubelist = CubeList()

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


def truth_dataframe_to_cube(df: DataFrame, training_dates: DatetimeIndex,) -> Cube:
    """Convert a truth DataFrame into an iris Cube.

    Args:
        df:
            DataFrame expected to contain the following columns: ob_value,
            time, wmo_id, diagnostic, latitude, longitude, altitude, cf_name,
            height, period and units. Any other columns are ignored.
        training_dates:
            Datetimes spanning the training period.

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
    return cubelist.merge_cube()


def forecast_and_truth_dataframes_to_cubes(
    forecast_df: DataFrame,
    truth_df: DataFrame,
    cycletime: str,
    forecast_period: int,
    training_length: int,
    percentiles: Optional[List[float]] = None,
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
    truth_cube = truth_dataframe_to_cube(truth_df, training_dates)
    return forecast_cube, truth_cube
