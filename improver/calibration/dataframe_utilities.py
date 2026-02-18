# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Functionality to convert a pandas DataFrame in the expected format
into an iris cube.

.. Further information is available in:
.. include:: extended_documentation/calibration/calibration_data_ingestion.rst


"""

import warnings
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

REPRESENTATION_COLUMNS = ["percentile", "realization"]

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
            f"The following compulsory column(s) are missing from the DataFrame: {diff}"
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


def quantile_check(df: DataFrame) -> None:
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
            "Forecast percentiles must be equally spaced. "
            f"The forecast percentiles are {df['percentile'].unique()}."
            "Based on the number of percentiles provided, the expected "
            f"percentiles would be {expected_percentiles}."
        )
        raise ValueError(msg)


def _fill_missing_entries(df, combi_cols, static_cols, site_id_col):
    """Fill the input DataFrame with rows that correspond to missing entries. The
    expected entries are computed using all combinations of the values within the
    combi_cols. In practice, this will allow support for creating entries for times
    that are missing when a new site with an ID is added. If the DataFrame provided
    is completely empty, then the empty DataFrame is returned.

    Args:
        df: DataFrame to be filled with rows corresponding to missing entries.
        combi_cols: The key columns within the DataFrame. All combinations of the
            values within these columns are expected to exist, otherwise, an
            entry will be created.
        static_cols: The names of the columns that are considered "static" and
            therefore can be reliably filled using other entries for the given WMO ID.
        site_id_col: Name of the column used to identify the sites within the DataFrame.

    Returns:
        DataFrame where any missing combination of the combi_cols will have been
        created.
    """
    if df.empty:
        return df

    # Create a DataFrame with rows for all possible combinations of combi_cols.
    # This results in rows with NaNs being created in the DataFrame.
    unique_vals_from_combi_cols = [df[c].unique() for c in combi_cols]
    new_index = pd.MultiIndex.from_product(
        unique_vals_from_combi_cols, names=combi_cols
    )
    df = df.set_index(combi_cols).reindex(new_index).reset_index(level=combi_cols)

    # Fill the NaNs within the static columns for each wmo_id.
    filled_df = df.groupby(site_id_col)[combi_cols + static_cols].ffill().bfill()
    df = df.drop(columns=static_cols)
    df = df.merge(filled_df, on=combi_cols)

    # Fill the blend_time and forecast_reference_time columns.
    if "forecast_period" in df.columns:
        for col in ["blend_time", "forecast_reference_time"]:
            df[col] = df["time"] - df["forecast_period"]
    return df


def _ensure_consistent_static_cols(
    forecast_df: DataFrame, static_cols: List[str], site_id_col: str
) -> DataFrame:
    """Ensure that the columns expected to have the same value for a given site,
    actually have the same values. These "static" columns could change if,
    for example, the altitude of a site is corrected.

    Args:
        forecast_df: Forecast DataFrame.
        static_cols: List of columns that are expected to be "static".
        site_id_col: The name of the column containing the site ID.

    Returns:
        Forecast DataFrame with the same value for a given site for the static columns
        provided.
    """
    # Check if any of the assumed static columns are actually not static when
    # the DataFrame is grouped by the site_id_col.
    if (forecast_df.groupby(site_id_col)[static_cols].nunique().nunique() > 1).any():
        for static_col in static_cols:
            # For each static column, find the last value from the list of unique
            # values for each site. The last value corresponds to the most recent value
            # present when using pd.unique.
            temp_df = forecast_df.groupby(site_id_col)[static_col].apply(
                lambda x: pd.unique(x)[-1]
            )
            # Drop the static column and then merge. The merge will recreate the static
            # column using a constant value for each site.
            forecast_df = forecast_df.drop(columns=static_col)
            forecast_df = forecast_df.merge(temp_df, on=site_id_col)

    return forecast_df


def _define_time_coord(
    adate: pd.Timestamp, time_bounds: Optional[Sequence[pd.Timestamp]] = None
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
        bounds=(
            time_bounds
            if time_bounds is None
            else [
                np.array(t.timestamp(), dtype=TIME_COORDS["time"].dtype)
                for t in time_bounds
            ]
        ),
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
    return AuxCoord(np.array(height, dtype=np.float32), "height", units="m")


def _training_dates_for_calibration(
    cycletime: str, forecast_period: int, training_length: int, adjacent_range: int = 0
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

    An adjacent_range in hours can be provided so that adjacent validity
    times may be used in the training dataset. The last validity time is not
    adjusted to ensure that later adjacent times are in the past, these will
    simply be discarded if they are not present in the truth data.

    Args:
        cycletime:
            Cycletime of a format similar to 20170109T0000Z.
            The training dates will always be in the past, relative
            to the cycletime.
        forecast_period:
            Forecast period in seconds as an integer.
        training_length:
            Training length in days as an integer.
        adjacent_range:
            A period in hours that should be used to either side of the
            defined forecast_period to allow for the inclusion of forecasts and
            observations that are close to the validity time being calibrated.

    Returns:
        Datetimes defining the training dataset. The number of datetimes
        is equal to the training length.
    """
    forecast_period = pd.Timedelta(int(forecast_period), unit="seconds")
    validity_times = [
        pd.Timestamp(cycletime) + forecast_period + pd.Timedelta(hours=offset)
        for offset in range(-adjacent_range, adjacent_range + 1)
    ]
    training_times = []
    for validity_time in validity_times:
        training_times.extend(
            list(
                pd.date_range(
                    end=validity_time
                    - pd.Timedelta(1, unit="days")
                    - forecast_period.floor("D"),
                    periods=int(training_length),
                    freq="D",
                    tz="UTC",
                )
            )
        )
    return pd.DatetimeIndex(sorted(training_times))


def _drop_duplicates(df: DataFrame, cols: Sequence[str]) -> DataFrame:
    """Drop duplicates and then sort the DataFrame.

    Args:
        df: DataFrame to have duplicates removed.
        cols: Columns for use in removing duplicates and for sorting.

    Returns:
        A DataFrame with duplicates removed (only the last duplicate is kept).
        The DataFrame is sorted according to the columns provided.
    """
    df = df.drop_duplicates(subset=cols, keep="last")
    return df.sort_values(by=cols, ignore_index=True)


def get_forecast_representation(df: DataFrame) -> str:
    """Check which of REPRESENTATION_COLUMNS (percentile or realization)
    exists in the DataFrame.

    Args:
        df:
            DataFrame expected to contain exactly one of REPRESENTATION_COLUMNS.

    Returns:
        representation_type:
            The member of REPRESENTATION_COLUMNS found in the DataFrame columns.

    Raises:
        ValueError:
            If none of the allowed columns are present, or more than one is present.
    """
    representations = set(REPRESENTATION_COLUMNS) & set(df.columns)
    if len(representations) > 1:
        raise ValueError(
            f"More than one column of {REPRESENTATION_COLUMNS} "
            "exists in the input dataset. The columns present are "
            f"{representations}."
        )
    if len(representations) == 0:
        raise ValueError(
            f"None of the columns {REPRESENTATION_COLUMNS} exist in the input dataset"
        )
    return representations.pop()


def _prepare_dataframes(
    forecast_df: DataFrame,
    truth_df: DataFrame,
    forecast_period: int,
    percentiles: Optional[List[float]] = None,
    experiment: Optional[str] = None,
    adjacent_range: int = 0,
) -> Tuple[DataFrame, DataFrame]:
    """Prepare DataFrames for conversion to cubes by: 1) checking which forecast
    representation is present, 2) checking that the expected columns are present,
    3) (Optionally) checking the percentiles are as expected, 4) removing
    duplicates from the forecast and truth, 5) finding the sites common to
    both the forecast and truth DataFrames and 6) replacing and supplementing
    the truth DataFrame with information from the forecast DataFrame. Note that
    this fourth step will also ensure that a row containing a NaN for the ob_value
    is inserted for any missing observations.

    Args:
        forecast_df:
            DataFrame expected to contain the following columns: forecast,
            blend_time, forecast_period, forecast_reference_time, time,
            wmo_id, one of REPRESENTATION_COLUMNS (percentile or
            realization), diagnostic, latitude, longitude, altitude,
            period, height, cf_name, units and experiment. Optionally, the
            DataFrame may also contain station_id. Any other columns are ignored.
        truth_df:
            DataFrame expected to contain the following columns: ob_value,
            time, wmo_id, diagnostic, latitude, longitude and altitude.
            Optionally the DataFrame may also contain the following columns:
            station_id, units. Any other columns are ignored.
        forecast_period:
            Forecast period in seconds as an integer.
        percentiles:
            The set of percentiles to be used for estimating EMOS coefficients.
        experiment:
            A value within the experiment column to select from the forecast
            table.
        adjacent_range:
            A period in hours that should be used to either side of the
            defined forecast_period to allow for the inclusion of forecasts and
            observations that are close to the validity time being calibrated.

    Returns:
        A sanitised version of the forecasts and truth DataFrames that
        are ready for conversion to cubes.
    """
    representation_type = get_forecast_representation(forecast_df)

    _dataframe_column_check(
        forecast_df, FORECAST_DATAFRAME_COLUMNS + [representation_type]
    )
    _dataframe_column_check(truth_df, TRUTH_DATAFRAME_COLUMNS)

    if (
        sum(["station_id" in forecast_df.columns, "station_id" in truth_df.columns])
        == 1
    ):
        df_type = "forecast" if "station_id" in forecast_df.columns else "truth"
        msg = (
            f"station_id is only within the {df_type} DataFrame. As station_id "
            "is not present in both DataFrames, station_id will be ignored."
        )
        warnings.warn(msg)

    include_station_id = False
    if ("station_id" in forecast_df.columns) and ("station_id" in truth_df.columns):
        include_station_id = True

    # Filter to select only one experiment.
    if experiment:
        forecast_df = forecast_df.loc[forecast_df["experiment"] == experiment]

    # Filter to forecast periods around target point within the adjacent_range.
    tolerance = pd.Timedelta(adjacent_range, unit="hours")
    fp_point = pd.Timedelta(int(forecast_period), unit="seconds")
    fp_start = fp_point - tolerance
    fp_end = fp_point + tolerance
    forecast_df = forecast_df[forecast_df["forecast_period"].between(fp_start, fp_end)]

    if forecast_df["experiment"].nunique() > 1:
        unique_exps = forecast_df["experiment"].unique()
        msg = (
            "More than one value for the experiment column found in the "
            f"forecast DataFrame. Values for experiment column {unique_exps}"
        )
        raise ValueError(msg)

    # Extract the required percentiles.
    if percentiles:
        indices = [np.isclose(forecast_df["percentile"], float(p)) for p in percentiles]
        forecast_df = forecast_df[np.logical_or.reduce(indices)]

    # Check the percentiles can be considered to be equally space quantiles.
    if representation_type == "percentile":
        quantile_check(forecast_df)

    # Remove forecast duplicates.
    forecast_cols = [
        "diagnostic",
        "forecast_period",
        "time",
        representation_type,
        "wmo_id",
    ]
    if include_station_id:
        forecast_cols.append("station_id")
    forecast_df = _drop_duplicates(forecast_df, forecast_cols)

    # Remove truth duplicates.
    truth_cols = ["diagnostic", "time", "wmo_id"]
    if include_station_id:
        truth_cols.append("station_id")
    truth_df = _drop_duplicates(truth_df, truth_cols)

    if include_station_id:
        # Find the common set of station ids.
        common_station_ids = np.sort(
            np.intersect1d(
                forecast_df["station_id"].unique(), truth_df["station_id"].unique()
            )
        )
        forecast_df = forecast_df[forecast_df["station_id"].isin(common_station_ids)]
        truth_df = truth_df[truth_df["station_id"].isin(common_station_ids)]
    else:
        # Find the common set of WMO IDs.
        common_wmo_ids = np.sort(
            np.intersect1d(forecast_df["wmo_id"].unique(), truth_df["wmo_id"].unique())
        )
        forecast_df = forecast_df[forecast_df["wmo_id"].isin(common_wmo_ids)]
        truth_df = truth_df[truth_df["wmo_id"].isin(common_wmo_ids)]

    # Ensure time in forecasts is present in truths.
    forecast_df = forecast_df[forecast_df["time"].isin(truth_df["time"].unique())]

    # Ensure time in truths is present in forecasts.
    truth_df = truth_df[truth_df["time"].isin(forecast_df["time"].unique())]

    # Fill in any missing instances where every combination of the columns
    # specified is expected to exist. This allows support for the
    # introduction of new sites within the forecast_df and truth_df.
    site_id_col = "station_id" if include_station_id else "wmo_id"
    combi_cols = [site_id_col, "time", representation_type]
    static_cols = [
        "latitude",
        "longitude",
        "altitude",
        "diagnostic",
        "period",
        "height",
        "cf_name",
        "units",
        "experiment",
        "forecast_period",
    ]
    if include_station_id:
        # Add wmo_id as a static column, if station ID is present in both the
        # forecast and truth DataFrames.
        static_cols.append("wmo_id")
    elif "station_id" in forecast_df.columns:
        # Add station_id as a static column, if it is only present in the
        # forecast DataFrame.
        static_cols.append("station_id")

    forecast_df = _fill_missing_entries(
        forecast_df, combi_cols, static_cols, site_id_col
    )

    forecast_df = _ensure_consistent_static_cols(
        forecast_df, ["altitude", "latitude", "longitude"], site_id_col
    )

    # Here we corrupt the forecast_periods of adjacent validity times to have a
    # consistent value with the target forecast period. This allows the
    # construction of cubes without multi-dimensional time coordinates.
    # The important value is the validity time to allow matching to observations.
    if adjacent_range > 0:
        forecast_df["forecast_period"] = fp_point

    combi_cols = [site_id_col, "time"]
    static_cols = ["latitude", "longitude", "altitude", "diagnostic"]
    if include_station_id:
        static_cols.append("wmo_id")
    elif "station_id" in truth_df.columns:
        static_cols.append("station_id")
    truth_df = _fill_missing_entries(truth_df, combi_cols, static_cols, site_id_col)

    # Sort to ensure a consistent ordering after filling in missing entries.
    forecast_df = forecast_df.sort_values(by=forecast_cols, ignore_index=True)
    truth_df = truth_df.sort_values(by=truth_cols, ignore_index=True)

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
        "time",
        "diagnostic",
    ]
    if include_station_id:
        subset_cols.append("station_id")
    # if units not present in truth_df, copy from forecast_df
    if "units" not in truth_df.columns:
        subset_cols.append("units")
    forecast_subset = forecast_df[subset_cols].drop_duplicates()

    # Use "right" to fill in any missing observations in the truth DataFrame
    # and retain the order from the forecast_subset.
    merge_cols = ["wmo_id", "time", "diagnostic"]
    if include_station_id:
        merge_cols.append("station_id")
    truth_df = truth_df.merge(forecast_subset, on=merge_cols, how="right")
    return forecast_df, truth_df


def forecast_dataframe_to_cube(
    df: DataFrame, training_dates: DatetimeIndex, forecast_period: int
) -> Cube:
    """Convert a forecast DataFrame into an iris Cube. The percentiles
    within the forecast DataFrame are rebadged as realizations.

    Args:
        df:
            DataFrame expected to contain the following columns: forecast,
            blend_time, forecast_period, forecast_reference_time, time,
            wmo_id, REPRESENTATION_COLUMNS (percentile or realization),
            diagnostic, latitude, longitude, period, height, cf_name, units.
            Optionally, the DataFrame may also contain station_id. Any other
            columns are ignored.
        training_dates:
            Datetimes spanning the training period.
        forecast_period:
            Forecast period in seconds as an integer.

    Returns:
        Cube containing the forecasts from the training period.
    """

    representation_type = get_forecast_representation(df)
    fp_point = pd.Timedelta(int(forecast_period), unit="seconds")
    cubelist = CubeList()

    for adate in training_dates:
        time_df = df.loc[(df["time"] == adate) & (df["forecast_period"] == fp_point)]

        if time_df.empty:
            continue
        time_df = _preprocess_temporal_columns(time_df)

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
            bounds=(
                fp_bounds
                if fp_bounds is None
                else [
                    np.array(
                        f.total_seconds(), dtype=TIME_COORDS["forecast_period"].dtype
                    )
                    for f in fp_bounds
                ]
            ),
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

        for var_val in sorted(time_df[representation_type].unique()):
            var_df = time_df.loc[time_df[representation_type] == var_val]
            cf_name = var_df["cf_name"].values[0]
            if representation_type == "percentile":
                var_coord = DimCoord(
                    np.float32(var_val), long_name="percentile", units="%"
                )
            elif representation_type == "realization":
                var_coord = DimCoord(
                    np.int32(var_val), standard_name="realization", units="1"
                )

            if "station_id" in var_df.columns:
                unique_site_id = var_df["station_id"].values.astype("<U8")
                unique_site_id_key = "station_id"
            else:
                unique_site_id = None
                unique_site_id_key = None

            cube = build_spotdata_cube(
                var_df["forecast"].astype(np.float32),
                cf_name,
                var_df["units"].values[0],
                var_df["altitude"].astype(np.float32),
                var_df["latitude"].astype(np.float32),
                var_df["longitude"].astype(np.float32),
                var_df["wmo_id"].values.astype("U5"),
                unique_site_id,
                unique_site_id_key,
                scalar_coords=[
                    time_coord,
                    frt_coord,
                    fp_coord,
                    var_coord,
                    height_coord,
                ],
            )
            cubelist.append(cube)

    if not cubelist:
        return
    cube = cubelist.merge_cube()

    if representation_type == "percentile":
        return RebadgePercentilesAsRealizations()(cube)
    return cube


def truth_dataframe_to_cube(df: DataFrame, training_dates: DatetimeIndex) -> Cube:
    """Convert a truth DataFrame into an iris Cube.

    Args:
        df:
            DataFrame expected to contain the following columns: ob_value,
            time, wmo_id, diagnostic, latitude, longitude, altitude, cf_name,
            height, period. Optionally the DataFrame may also contain
            the following columns: station_id, units. Any other columns are ignored.
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

        if "station_id" in time_df.columns:
            unique_site_id = time_df["station_id"].values.astype("<U8")
            unique_site_id_key = "station_id"
        else:
            unique_site_id = None
            unique_site_id_key = None

        cube = build_spotdata_cube(
            time_df["ob_value"].astype(np.float32),
            time_df["cf_name"].values[0],
            time_df["units"].values[0],
            time_df["altitude"].astype(np.float32),
            time_df["latitude"].astype(np.float32),
            time_df["longitude"].astype(np.float32),
            time_df["wmo_id"].values.astype("U5"),
            unique_site_id,
            unique_site_id_key,
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
    adjacent_range: int = 0,
) -> Tuple[Cube, Cube]:
    """Convert a forecast DataFrame into an iris Cube and a
    truth DataFrame into an iris Cube.

    Args:
        forecast_df:
            DataFrame expected to contain the following columns: forecast,
            blend_time, forecast_period, forecast_reference_time, time,
            wmo_id, one of REPRESENTATION_COLUMNS (percentile or realization),
            diagnostic, latitude, longitude, period, height, cf_name, units.
            Optionally, the DataFrame may also contain station_id. Any other
            columns are ignored.
        truth_df:
            DataFrame expected to contain the following columns: ob_value,
            time, wmo_id, diagnostic, latitude, longitude and altitude.  Optionally the
            DataFrame may also contain the following columns: station_id, units.
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
        adjacent_range:
            A period in hours that should be used to either side of the
            defined forecast_period to allow for the inclusion of forecasts and
            observations that are close to the validity time being calibrated.

    Returns:
        Forecasts and truths for the training period in Cube format.
    """
    training_dates = _training_dates_for_calibration(
        cycletime, forecast_period, training_length, adjacent_range
    )

    forecast_df, truth_df = _prepare_dataframes(
        forecast_df,
        truth_df,
        forecast_period,
        percentiles=percentiles,
        experiment=experiment,
        adjacent_range=adjacent_range,
    )

    forecast_cube = forecast_dataframe_to_cube(
        forecast_df, training_dates, forecast_period
    )
    truth_cube = truth_dataframe_to_cube(truth_df, training_dates)
    return forecast_cube, truth_cube
