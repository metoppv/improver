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
import iris
import numpy as np
from cf_units import Unit
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.tests import IrisTest

from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.metadata.probabilistic import extract_diagnostic_name
from improver.utilities.warnings_handler import ManageWarnings

from ...set_up_test_cubes import set_up_variable_cube

IGNORED_MESSAGES = ["Collapsing a non-contiguous coordinate"]
WARNING_TYPES = [UserWarning]


class EnsembleCalibrationAssertions(IrisTest):

    """Additional assertions, specifically for usage in the
    ensemble calibration unit tests."""

    def assertEMOSCoefficientsAlmostEqual(self, first, second):
        """Overriding of the assertArrayAlmostEqual method to check whether
        array are matching to 4 decimal places. This is specifically
        for use in assertions involving the EMOS coefficients. This is
        justified based on the default tolerance of the minimisation using the
        Nelder-Mead algorithm of 0.0001, so that minimisations on different
        machines would only be aiming to match to 4 decimal places.

        Args:
            first (numpy.ndarray):
                First array to compare.
            second (numpy.ndarray):
                Second array to compare.
         """
        self.assertArrayAlmostEqual(first, second, decimal=4)

    def assertCalibratedVariablesAlmostEqual(self, first, second):
        """Overriding of the assertArrayAlmostEqual method to check whether
        array are matching to 4 decimal places. This is specifically
        for use in assertions following applying the EMOS coefficients,
        in order to calibrate the chosen variables. This is justified
        based on the default tolerance of the minimisation using the
        Nelder-Mead algorithm of 0.0001, so that minimisations on different
        machines would only be aiming to match to 4 decimal places.
        Args:
            first (numpy.ndarray):
                First array to compare.
            second (numpy.ndarray):
                Second array to compare.
         """
        self.assertArrayAlmostEqual(first, second, decimal=4)


class SetupCubes(IrisTest):

    """Set up cubes for testing."""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def setUp(self):
        """Set up temperature and wind speed cubes for testing."""
        super().setUp()
        frt_dt = datetime.datetime(2017, 11, 10, 0, 0)
        time_dt = datetime.datetime(2017, 11, 10, 4, 0)

        base_data = np.array([[[0.3, 1.1, 2.6],
                               [4.2, 5.3, 6.],
                               [7.1, 8.2, 9.]],
                              [[0.7, 2., 3],
                               [4.3, 5.6, 6.4],
                               [7., 8., 9.]],
                              [[2.1, 3., 3.],
                               [4.8, 5., 6.],
                               [7.9, 8., 8.9]]], dtype=np.float32)
        temperature_data = base_data + 273.15
        self.current_temperature_forecast_cube = set_up_variable_cube(
            temperature_data, units="Kelvin", realizations=[0, 1, 2],
            time=time_dt, frt=frt_dt, attributes=MANDATORY_ATTRIBUTE_DEFAULTS)

        # Create historic forecasts and truth
        self.historic_forecasts = _create_historic_forecasts(
            temperature_data, time_dt, frt_dt, realizations=[0, 1, 2])
        self.truth = _create_truth(temperature_data, time_dt)

        # Create a combined list of historic forecasts and truth
        self.combined = self.historic_forecasts + self.truth

        # Create the historic and truth cubes
        self.historic_temperature_forecast_cube = (
            self.historic_forecasts.merge_cube())
        self.temperature_truth_cube = self.truth.merge_cube()

        # Create a cube for testing wind speed.
        self.current_wind_speed_forecast_cube = set_up_variable_cube(
            base_data, name="wind_speed", units="m s-1",
            realizations=[0, 1, 2], attributes=MANDATORY_ATTRIBUTE_DEFAULTS)

        self.historic_wind_speed_forecast_cube = _create_historic_forecasts(
            base_data, time_dt, frt_dt, realizations=[0, 1, 2],
            name="wind_speed", units="m s-1").merge_cube()

        self.wind_speed_truth_cube = _create_truth(
            base_data, time_dt, name="wind_speed", units="m s-1").merge_cube()

        # Set up another set of cubes which have a halo of zeros round the
        # original data. This data will be masked out in tests using a
        # landsea_mask
        base_data = np.array([[[0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.3, 1.1, 2.6, 0.0],
                               [0.0, 4.2, 5.3, 6., 0.0],
                               [0.0, 7.1, 8.2, 9., 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.7, 2., 3, 0.0],
                               [0.0, 4.3, 5.6, 6.4, 0.0],
                               [0.0, 7., 8., 9., 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 2.1, 3., 3., 0.0],
                               [0.0, 4.8, 5., 6., 0.0],
                               [0.0, 7.9, 8., 8.9, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0]]],
                             dtype=np.float32)
        temperature_data = base_data + 273.15
        # Create historic forecasts and truth
        self.historic_forecasts_halo = _create_historic_forecasts(
            temperature_data, time_dt, frt_dt, realizations=[0, 1, 2])
        self.truth_halo = _create_truth(temperature_data, time_dt)

        # Create the historic and truth cubes
        self.historic_temperature_forecast_cube_halo = (
            self.historic_forecasts_halo.merge_cube())
        self.temperature_truth_cube_halo = self.truth_halo.merge_cube()

        # Create a cube for testing wind speed.
        self.historic_wind_speed_forecast_cube_halo = (
            _create_historic_forecasts(
                base_data, time_dt, frt_dt, realizations=[0, 1, 2],
                name="wind_speed", units="m s-1").merge_cube())

        self.wind_speed_truth_cube_halo = _create_truth(
            base_data, time_dt, name="wind_speed", units="m s-1").merge_cube()


def set_up_probability_threshold_cube(
        data, phenomenon_standard_name, phenomenon_units,
        forecast_thresholds=np.array([8, 10, 12]), timesteps=1,
        y_dimension_length=3, x_dimension_length=3,
        spp__relative_to_threshold='above'):
    """
    Create a cube containing multiple probability_above/below_threshold
    values for the coordinate.
    """
    cube_long_name = (
        "probability_of_{}_{}_threshold".format(
            phenomenon_standard_name, spp__relative_to_threshold))
    cube = Cube(data, long_name=cube_long_name,
                units=1)
    threshold_coord_name = extract_diagnostic_name(cube_long_name)

    try:
        cube.add_dim_coord(
            DimCoord(forecast_thresholds, threshold_coord_name,
                     units=phenomenon_units, var_name="threshold"), 0)
    except ValueError:
        cube.add_dim_coord(
            DimCoord(forecast_thresholds, long_name=threshold_coord_name,
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
    cube.coord(
        var_name="threshold").attributes["spp__relative_to_threshold"] = (
            spp__relative_to_threshold)
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
            spp__relative_to_threshold='above'))


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
    cube.coord(
        var_name="threshold").attributes[
            "spp__relative_to_threshold"] = "above"
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
                realizations=np.array([0, 1, 2], dtype=np.int32),
                timesteps=1, y_dimension_length=3, x_dimension_length=3):
    """Create a cube containing multiple realizations."""
    try:
        cube = Cube(data, standard_name=name, units=units)
    except ValueError:
        cube = Cube(data, long_name=name, units=units)

    cube.add_dim_coord(DimCoord(realizations, 'realization',
                                units='1'), 0)
    time_origin = "seconds since 1970-01-01 00:00:00"
    calendar = "gregorian"
    tunit = Unit(time_origin, calendar)
    dt1 = datetime.datetime(2017, 1, 10, 3, 0)
    dt2 = datetime.datetime(2017, 1, 10, 4, 0)
    num1 = cf_units.date2num(dt1, time_origin, calendar)
    num2 = cf_units.date2num(dt2, time_origin, calendar)
    cube.add_dim_coord(DimCoord(np.linspace(num1, num2, timesteps,
                                            dtype=np.int64),
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


def _create_historic_forecasts(data, time_dt, frt_dt,
                               standard_grid_metadata="uk_ens",
                               number_of_days=5, **kwargs):
    """
    Function to create a set of pseudo historic forecast cubes, based on the
    input cube, and assuming that there will be one forecast per day at the
    same hour of the day.

    Please see improver.tests.set_up_test_cubes.set_up_variable_cube for the
    supported keyword arguments.

    Args:
        data (numpy.ndarray):
            Numpy array to define the data that will be used to fill the cube.
            This will be subtracted by 2 with the aim of giving a difference
            between the current forecast and the historic forecasts.
            Therefore, the current forecast would contain the unadjusted data.
        time_dt (datetime.datetime):
            Datetime to define the initial validity time. This will be
            incremented in days up to the defined number_of_days.
        frt_dt (datetime.datetime):
            Datetime to define the initial forecast reference time. This will
            be incremented in days up to the defined number_of_days.
        standard_grid_metadata (str):
            Please see improver.tests.set_up_test_cubes.set_up_variable_cube.
        number_of_days(int):
            Number of days to increment when constructing a cubelist of the
            historic forecasts.

    Returns:
        iris.cube.CubeList:
            Cubelist of historic forecasts in one day increments.
    """
    historic_forecasts = iris.cube.CubeList([])
    for day in range(number_of_days):
        new_frt_dt = frt_dt + datetime.timedelta(days=day)
        new_time_dt = time_dt + datetime.timedelta(days=day)
        historic_forecasts.append(
            set_up_variable_cube(
                data - 2, time=new_time_dt, frt=new_frt_dt,
                standard_grid_metadata=standard_grid_metadata, **kwargs))
    return historic_forecasts


def _create_truth(data, time_dt, number_of_days=5, **kwargs):
    """
    Function to create truth cubes, based on the input cube, and assuming that
    there will be one forecast per day at the same hour of the day.

    Please see improver.tests.set_up_test_cubes.set_up_variable_cube for the
    other supported keyword arguments.

    Args:
        data (numpy.ndarray):
            Numpy array to define the data that will be used to fill the cube.
            This will be subtracted by 3 with the aim of giving a difference
            between the current forecast and the truths.
            Therefore, the current forecast would contain the unadjusted data.
        time_dt (datetime.datetime):
            Datetime to define the initial validity time. This will be
            incremented in days up to the defined number_of_days.
        standard_grid_metadata (str):
            Please see improver.tests.set_up_test_cubes.set_up_variable_cube.
        number_of_days(int):
            Number of days to increment when constructing a cubelist of the
            historic forecasts.
    """
    truth = iris.cube.CubeList([])
    for day in range(number_of_days):
        new_time_dt = time_dt + datetime.timedelta(days=day)
        truth.append(set_up_variable_cube(
            np.amax(data - 3, axis=0), time=new_time_dt,
            frt=new_time_dt, standard_grid_metadata="uk_det", **kwargs))
    return truth


if __name__ == "__main__":
    pass
