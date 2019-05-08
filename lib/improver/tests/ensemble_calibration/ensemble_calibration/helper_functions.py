# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
Functions for use within unit tests for `ensemble_calibration` plugins.
"""
import datetime

import cf_units
from cf_units import Unit
import iris
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList
import numpy as np

from improver.utilities.cube_manipulation import concatenate_cubes


def set_up_probability_threshold_cube(
        data, phenomenon_standard_name, phenomenon_units,
        forecast_thresholds=np.array([8, 10, 12]), timesteps=1,
        y_dimension_length=3, x_dimension_length=3,
        relative_to_threshold='above'):
    """
    Create a cube containing multiple probability_above/below_threshold
    values for the coordinate.
    """
    cube_long_name = (
        "probability_of_{}_{}_threshold".format(
            phenomenon_standard_name, relative_to_threshold))
    cube = Cube(data, long_name=cube_long_name,
                units=1)

    try:
        cube.add_dim_coord(
            DimCoord(forecast_thresholds, phenomenon_standard_name,
                     units=phenomenon_units, var_name="threshold"), 0)
    except ValueError:
        cube.add_dim_coord(
            DimCoord(forecast_thresholds, long_name=phenomenon_standard_name,
                     units=phenomenon_units, var_name="threshold"), 0)

    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"
    tunit = Unit(time_origin, calendar)
    cube.add_dim_coord(DimCoord(np.linspace(412227.0, 412327.0, timesteps,
                                            dtype=np.float32),
                                "time", units=tunit), 1)
    cube.add_dim_coord(DimCoord(np.linspace(-45.0, 45.0, y_dimension_length,
                                            dtype=np.float32),
                                'latitude', units='degrees'), 2)
    cube.add_dim_coord(DimCoord(np.linspace(120, 180, x_dimension_length,
                                            dtype=np.float32),
                                'longitude', units='degrees'), 3)
    cube.attributes["relative_to_threshold"] = relative_to_threshold
    return cube


def set_up_probability_above_threshold_temperature_cube():
    """
    Create a cube with metadata and values suitable for air temperature.
    """
    data = np.array([[[[1.0, 0.9, 1.0],
                       [0.8, 0.9, 0.5],
                       [0.5, 0.2, 0.0]]],
                     [[[1.0, 0.5, 1.0],
                       [0.5, 0.5, 0.3],
                       [0.2, 0.0, 0.0]]],
                     [[[1.0, 0.2, 0.5],
                       [0.2, 0.0, 0.1],
                       [0.0, 0.0, 0.0]]]], dtype=np.float32)
    return (
        set_up_probability_threshold_cube(
            data, "air_temperature", "degreesC",
            relative_to_threshold='above'))


def set_up_probability_above_threshold_spot_cube(
        data, phenomenon_standard_name, phenomenon_units,
        forecast_thresholds=np.array([8, 10, 12]),
        y_dimension_length=9, x_dimension_length=9):
    """
    Create a cube containing multiple realizations, where one of the
    dimensions is an index used for spot forecasts.
    """
    cube_long_name = (
        "probability_of_{}_above_threshold".format(phenomenon_standard_name))
    cube = Cube(data, long_name=cube_long_name,
                units=1)

    try:
        cube.add_dim_coord(
            DimCoord(forecast_thresholds, phenomenon_standard_name,
                     units=phenomenon_units, var_name="threshold"), 0)
    except ValueError:
        cube.add_dim_coord(
            DimCoord(forecast_thresholds, long_name=phenomenon_standard_name,
                     units=phenomenon_units, var_name="threshold"), 0)

    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"
    tunit = Unit(time_origin, calendar)
    cube.add_dim_coord(DimCoord(np.array([412227], dtype=np.float64),
                                "time", units=tunit), 1)
    cube.add_dim_coord(
        DimCoord(np.arange(9, dtype=np.float32), long_name='locnum',
                 units="1"),
        2
    )
    cube.add_aux_coord(AuxCoord(np.linspace(-45.0, 45.0, y_dimension_length,
                                            dtype=np.float32),
                                'latitude', units='degrees'), data_dims=2)
    cube.add_aux_coord(AuxCoord(np.linspace(120, 180, x_dimension_length,
                                            dtype=np.float32),
                                'longitude', units='degrees'), data_dims=2)
    cube.attributes["relative_to_threshold"] = "above"
    return cube


def set_up_probability_above_threshold_spot_temperature_cube():
    """
    Create a cube with metadata and values suitable for air temperature
    for spot forecasts.
    """
    data = np.array([[[1.0, 0.9, 1.0,
                       0.8, 0.9, 0.5,
                       0.5, 0.2, 0.0]],
                     [[1.0, 0.5, 1.0,
                       0.5, 0.5, 0.3,
                       0.2, 0.0, 0.0]],
                     [[1.0, 0.2, 0.5,
                       0.2, 0.0, 0.1,
                       0.0, 0.0, 0.0]]], dtype=np.float32)
    return (
        set_up_probability_above_threshold_spot_cube(
            data, "air_temperature", "degreesC"))


def set_up_cube(data, name, units,
                realizations=np.array([0, 1, 2], dtype=np.float32),
                timesteps=1, y_dimension_length=3, x_dimension_length=3):
    """Create a cube containing multiple realizations."""
    try:
        cube = Cube(data, standard_name=name, units=units)
    except ValueError:
        cube = Cube(data, long_name=name, units=units)

    cube.add_dim_coord(DimCoord(realizations, 'realization',
                                units='1'), 0)
    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"
    tunit = Unit(time_origin, calendar)
    dt1 = datetime.datetime(2017, 1, 10, 3, 0)
    dt2 = datetime.datetime(2017, 1, 10, 4, 0)
    num1 = cf_units.date2num(dt1, time_origin, calendar)
    num2 = cf_units.date2num(dt2, time_origin, calendar)
    cube.add_dim_coord(DimCoord(np.linspace(num1, num2, timesteps,
                                            dtype=np.float64),
                                "time", units=tunit), 1)
    cube.add_dim_coord(DimCoord(np.linspace(-45.0, 45.0, y_dimension_length,
                                            dtype=np.float32),
                                'latitude', units='degrees'), 2)
    cube.add_dim_coord(DimCoord(np.linspace(120, 180, x_dimension_length,
                                            dtype=np.float32),
                                'longitude', units='degrees'), 3)
    return cube


def set_up_temperature_cube(dtype=np.float32):
    """Create a cube with metadata and values suitable for air temperature."""
    data = (np.tile(np.linspace(-45.0, 45.0, 9), 3).reshape(3, 1, 3, 3) +
            273.15)
    data[0] -= 2
    data[1] += 2
    data[2] += 4
    return set_up_cube(data.astype(dtype), "air_temperature", "K")


def set_up_spot_cube(data, phenomenon_standard_name, phenomenon_units):
    """Create a cube containing multiple realizations."""
    cube = Cube(data, standard_name=phenomenon_standard_name,
                units=phenomenon_units)
    cube.add_dim_coord(DimCoord([0, 1, 2], 'realization',
                                units='1'), 0)
    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"
    tunit = Unit(time_origin, calendar)
    cube.add_dim_coord(DimCoord(np.array(412227, dtype=np.float64),
                                "time", units=tunit), 1)
    cube.add_dim_coord(DimCoord(np.arange(9, dtype=np.float32),
                                long_name='locnum',
                                units="1"), 2)
    cube.add_aux_coord(AuxCoord(np.linspace(-45.0, 45.0, 9, dtype=np.float32),
                                'latitude', units='degrees'), data_dims=2)
    cube.add_aux_coord(AuxCoord(np.linspace(120, 180, 9, dtype=np.float32),
                                'longitude', units='degrees'), data_dims=2)
    return cube


def set_up_spot_temperature_cube():
    """Create a cube with metadata and values suitable for air temperature."""
    data = (np.tile(np.linspace(-45.0, 45.0, 9), 3).reshape(3, 1, 9) +
            273.15)
    data[0] -= 2
    data[1] += 2
    data[2] += 4
    return set_up_spot_cube(data.astype(np.float32), "air_temperature", "K")


def set_up_wind_speed_cube():
    """Create a cube with metadata and values suitable for wind speed."""
    data = np.tile(np.linspace(0, 60, 9), 3).reshape(3, 1, 3, 3)
    data[0] += 0
    data[1] += 2
    data[2] += 4
    return set_up_cube(data.astype(np.float32), "wind_speed", "m s^-1")


def add_forecast_reference_time_and_forecast_period(
        cube, time_point=402295.0, fp_point=4.0):
    """
    Function to add forecast_reference_time and forecast_period coordinates
    to the input cube.
    """
    cube.coord("time").points = time_point
    coord_position = cube.coord_dims("time")
    if not isinstance(fp_point, list):
        fp_point = [fp_point]
    fp_points = fp_point
    frt_points = cube.coord("time").points[0] - fp_points[0]
    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"
    tunit = Unit(time_origin, calendar)
    cube.add_aux_coord(
        DimCoord([frt_points], "forecast_reference_time", units=tunit))
    cube.add_aux_coord(
        DimCoord(fp_points, "forecast_period", units="hours"),
        data_dims=coord_position)
    return cube


def _create_historic_forecasts(cube, number_of_days=5):
    """
    Function to create a set of pseudo historic forecast cubes, based on the
    input cube, and assuming that there will be one forecast per day at the
    same hour of the day.
    """
    historic_forecasts = CubeList([])
    no_of_hours_in_day = 24
    time_range = np.linspace(
        no_of_hours_in_day, no_of_hours_in_day*number_of_days,
        num=number_of_days, endpoint=True)
    for index in time_range:
        temp_cube = cube.copy()
        # TODO: Remove conversion to hours, once all ensemble calibration
        # unit tests have been upgraded to use set_up_variable_cube.
        for coord_name in ["forecast_reference_time", "time"]:
            orig_units = temp_cube.coord(coord_name).units
            temp_cube.coord(coord_name).convert_units(
                "hours since 1970-01-01 00:00:00")
            temp_cube.coord(coord_name).points = (
                temp_cube.coord(coord_name).points - index)
            temp_cube.coord(coord_name).convert_units(orig_units)
        temp_cube.data -= 2
        historic_forecasts.append(temp_cube)

    historic_forecast = concatenate_cubes(
        historic_forecasts,
        coordinates_for_association=["forecast_reference_time"])
    return historic_forecast


def _create_truth(cube):
    """
    Function to create truth cubes, based on the input cube, and assuming that
    there will be one forecast per day at the same hour of the day.
    """
    truth = CubeList([])
    time_range = np.array([24.0, 48.0, 72.0, 96.0, 120.0], dtype=np.float32)
    for index in time_range:
        temp_cube = cube.copy()
        index -= temp_cube.coord("forecast_period").points[0]
        temp_cube.coord("forecast_reference_time").points = (
            temp_cube.coord("forecast_reference_time").points - index)
        temp_cube.coord("time").points = temp_cube.coord("time").points - index
        temp_cube.coord("forecast_period").points = 0
        temp_cube.data -= 3
        temp_cube = temp_cube.collapsed("realization", iris.analysis.MAX)
        temp_cube.remove_coord("realization")
        temp_cube.cell_methods = {}
        truth.append(temp_cube)
    truth = concatenate_cubes(
        truth, coordinates_for_association=["forecast_reference_time"])
    return truth


if __name__ == "__main__":
    pass
