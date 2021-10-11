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
"""init for calibration"""

from collections import OrderedDict
from datetime import timedelta
from typing import Any, List, Optional, Tuple

import iris
import numpy as np
import pandas as pd
from iris.cube import Cube, CubeList
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
from improver.utilities.temporal import cycletime_to_datetime


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


def _unique_check(table: DataFrame, column: str) -> Any:
    """Check whether the value in the column is unique.

    Args:
        table:
            The DataFrame to be checked.
        column:
            Name of a column in the table.

    Raises:
        ValueError: Only one unique value within the specifed column
            is expected.
    """
    if len(table[column].unique()) > 1:
        msg = (
            f"Multiple values provided for the {column}: "
            f"{table[column].unique()}. "
            f"Only one value for the {column} is expected."
        )
        raise ValueError(msg)


def _date_range_for_calibration(
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
    validity_time = cycletime_to_datetime(cycletime) + timedelta(
        hours=int(forecast_period)
    )

    return pd.date_range(
        end=validity_time - timedelta(days=1), periods=int(training_length), freq="D"
    )


def forecast_table_to_cube(
    table: DataFrame, date_range: DatetimeIndex, forecast_period: int
) -> Cube:
    """Convert a forecast table into an iris Cube. The percentiles within the
    forecast table are rebadged as realizations.

    Args:
        table:
            DataFrame expected to contain the following columns: forecast,
            blend_time, forecast_period, forecast_reference_time, time,
            wmo_id, percentile, diagnostic, latitude, longitude, period,
            height, cf_name, units.
        date_range:
            Datetimes spanning the training period.
        forecast_period:
            Forecast period in hours as an integer.

    Returns:
        Cube containing the forecasts from the training period.
    """
    for coord in ["time", "forecast_reference_time"]:
        table[coord] = table[coord].dt.tz_localize(None)

    cubelist = CubeList()
    for adate in date_range:
        time_table = table.loc[
            (table["time"] == adate)
            & (table["forecast_period"] == timedelta(hours=int(forecast_period)))
        ]

        if time_table.empty:
            continue

        # The following columns are expected to contain one unique value
        # per column.
        for col in ["period", "height", "cf_name", "units", "diagnostic"]:
            _unique_check(time_table, col)

        time_point = np.datetime64(adate, "s")
        fp_point = np.timedelta64(int(forecast_period), "h").astype("timedelta64[s]")
        frt_point = np.datetime64(
            time_table["forecast_reference_time"].values[0], "s"
        ).astype(np.int64)

        if time_table["period"].isna().all():
            time_bounds = None
            fp_bounds = None
        else:
            period = np.timedelta64(time_table["period"].values[0], "s")
            time_bounds = [time_point - period, time_point]
            fp_bounds = [fp_point - period, fp_point]

        time_coord = iris.coords.DimCoord(
            time_point.astype(TIME_COORDS["time"].dtype),
            "time",
            bounds=[t.astype(TIME_COORDS["time"].dtype) for t in time_bounds]
            if time_bounds
            else time_bounds,
            units=TIME_COORDS["time"].units,
        )
        fp_coord = iris.coords.AuxCoord(
            fp_point.astype(TIME_COORDS["forecast_period"].dtype),
            "forecast_period",
            bounds=[f.astype(TIME_COORDS["forecast_period"].dtype) for f in fp_bounds]
            if fp_bounds
            else fp_bounds,
            units=TIME_COORDS["forecast_period"].units,
        )
        frt_coord = iris.coords.AuxCoord(
            frt_point,
            "forecast_reference_time",
            units=TIME_COORDS["forecast_reference_time"].units,
        )

        height_coord = iris.coords.AuxCoord(
            time_table["height"].values[0], "height", units="m",
        )

        for percentile in table["percentile"].unique():
            perc_coord = iris.coords.DimCoord(
                np.int32(percentile), long_name="percentile", units=1
            )
            perc_table = time_table.loc[time_table["percentile"] == percentile]

            cube = build_spotdata_cube(
                perc_table["forecast"].astype(np.float32),  # data
                perc_table["cf_name"].values[0],
                perc_table["units"].values[0],
                perc_table["altitude"].astype(np.float32),  # altitude
                perc_table["latitude"].astype(np.float32),  # latitude
                perc_table["longitude"].astype(np.float32),  # longitude
                perc_table["wmo_id"].values,
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
        msg = (
            f"No entries matching these dates {date_range} and "
            f"this forecast period {forecast_period} are "
            "available within the dataframe provided."
        )
        raise ValueError(msg)
    cube = cubelist.merge_cube()

    return RebadgePercentilesAsRealizations()(cube)


def truth_table_to_cube(
    table: DataFrame,
    date_range: DatetimeIndex,
    period: float,
    height: float,
    cf_name: str,
    units: str,
) -> Cube:
    """Convert a truth table into an iris Cube.

    Args:
        table:
            DataFrame expected to contain the following columns: ob_value,
            time, wmo_id, diagnostic, latitude, longitude and altitude.
        date_range:
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
    table["time"] = table["time"].dt.tz_localize(None)

    cubelist = CubeList()
    for adate in date_range:

        time_table = table.loc[table["time"] == adate]

        if time_table.empty:
            continue

        # The following columns are expected to contain one unique value
        # per column.
        _unique_check(time_table, "diagnostic")

        # Ensure that every WMO ID has an entry for a particular time.
        new_index = Index(table["wmo_id"].unique(), name="wmo_id")
        time_table = time_table.set_index("wmo_id").reindex(new_index)
        # Fill the alt/lat/lon with the mode to ensure consistent coordinates
        # to support merging. Also fill other columns known to contain one
        # unique value. Columns containing entirely NaNs will return NaNs.
        for col in ["altitude", "latitude", "longitude", "diagnostic"]:
            time_table[col] = table.groupby(by="wmo_id", sort=False)[col].agg(
                lambda x: pd.Series.mode(x, dropna=not table[col].isna().all())
            )
            # Replace empty arrays generated by the mode with NaNs.
            time_table[col] = time_table[col].apply(
                lambda x: np.nan if isinstance(x, np.ndarray) else x
            )
        time_table = time_table.reset_index()

        time_point = np.datetime64(adate, "s")
        if period is np.nan:
            time_bounds = None
        else:
            time_bounds = [
                time_point - np.timedelta64(period, "s"),
                time_point,
            ]

        time_coord = iris.coords.DimCoord(
            time_point.astype(TIME_COORDS["time"].dtype),
            "time",
            bounds=[t.astype(TIME_COORDS["time"].dtype) for t in time_bounds]
            if time_bounds
            else time_bounds,
            units=TIME_COORDS["time"].units,
        )

        height_coord = iris.coords.AuxCoord(height, "height", units="m",)

        cube = build_spotdata_cube(
            time_table["ob_value"].astype(np.float32),  # data
            cf_name,
            units,
            time_table["altitude"],
            time_table["latitude"],
            time_table["longitude"],
            time_table["wmo_id"].values,
            scalar_coords=[time_coord, height_coord],
        )
        cubelist.append(cube)

    if not cubelist:
        msg = (
            f"No entries matching these dates {date_range} "
            "are available within the dataframe provided."
        )
        raise ValueError(msg)
    return cubelist.merge_cube()


def _filter_forecasts_and_truths(forecast: Cube, truth: Cube) -> Tuple[Cube, Cube]:
    """Filter forecasts and truths to ensure consistent WMO IDs.

    Args:
        forecast:
            The training period forecasts.
        truth:
            The training period truths.

    Returns:
        Forecasts and truths for the training period that have been filtered
        only include sites that are present in both the forecasts and truths
        based on the WMO ID.

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


def forecast_and_truth_tables_to_cubes(
    forecast_table: DataFrame,
    truth_table: DataFrame,
    cycletime: str,
    forecast_period: int,
    training_length: int,
) -> Tuple[Cube, Cube]:
    """Convert a truth table into an iris Cube.

    Args:
        forecast_table:
            DataFrame expected to contain the following columns: .
        truth_table:
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
    date_range = _date_range_for_calibration(
        cycletime, forecast_period, training_length
    )
    forecast_cube = forecast_table_to_cube(forecast_table, date_range, forecast_period)
    truth_cube = truth_table_to_cube(
        truth_table,
        date_range,
        period=forecast_table["period"].values[0],
        height=forecast_table["height"].values[0],
        cf_name=forecast_table["cf_name"].values[0],
        units=forecast_table["units"].values[0],
    )
    forecast_cube, truth_cube = _filter_forecasts_and_truths(forecast_cube, truth_cube)
    return forecast_cube, truth_cube
