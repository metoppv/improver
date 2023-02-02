# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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

import iris
import numpy as np
from cf_units import Unit
from iris.tests import IrisTest

from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.synthetic_data.set_up_test_cubes import (
    construct_scalar_time_coords,
    set_up_variable_cube,
)

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

    def setUp(self):
        """Set up temperature and wind speed cubes for testing."""
        super().setUp()
        frt_dt = datetime.datetime(2017, 11, 10, 0, 0)
        time_dt = datetime.datetime(2017, 11, 10, 4, 0)

        base_data = np.array(
            [
                [[0.3, 1.1, 2.6], [4.2, 5.3, 5.9], [7.1, 8.2, 8.8]],
                [[0.7, 2.0, 2.9], [4.3, 5.6, 6.4], [7.0, 7.0, 9.2]],
                [[2.1, 3.0, 3.1], [4.8, 5.0, 6.1], [7.9, 8.1, 8.9]],
            ],
            dtype=np.float32,
        )
        temperature_data = Unit("Celsius").convert(base_data, "Kelvin")
        self.current_temperature_forecast_cube = set_up_variable_cube(
            temperature_data,
            units="Kelvin",
            realizations=[0, 1, 2],
            time=time_dt,
            frt=frt_dt,
            attributes=MANDATORY_ATTRIBUTE_DEFAULTS,
        )
        time_dt = time_dt - datetime.timedelta(days=5)
        frt_dt = frt_dt - datetime.timedelta(days=5)

        # Create historic forecasts and truth
        self.historic_forecasts = _create_historic_forecasts(
            temperature_data, time_dt, frt_dt, realizations=[0, 1, 2]
        )
        self.truth = _create_truth(temperature_data, time_dt)

        # Create a combined list of historic forecasts and truth
        self.combined = self.historic_forecasts + self.truth

        # Create the historic and truth cubes
        self.historic_temperature_forecast_cube = self.historic_forecasts.merge_cube()
        # Ensure the forecast coordinates are in the order: realization, time, lat, lon.
        self.historic_temperature_forecast_cube.transpose([1, 0, 2, 3])
        self.temperature_truth_cube = self.truth.merge_cube()

        # Create a cube for testing wind speed.
        self.current_wind_speed_forecast_cube = set_up_variable_cube(
            base_data,
            name="wind_speed",
            units="m s-1",
            realizations=[0, 1, 2],
            attributes=MANDATORY_ATTRIBUTE_DEFAULTS,
        )

        self.historic_wind_speed_forecast_cube = _create_historic_forecasts(
            base_data,
            time_dt,
            frt_dt,
            realizations=[0, 1, 2],
            name="wind_speed",
            units="m s-1",
        ).merge_cube()
        # Ensure the forecast coordinates are in the order: realization, time, lat, lon.
        self.historic_wind_speed_forecast_cube.transpose([1, 0, 2, 3])

        self.wind_speed_truth_cube = _create_truth(
            base_data, time_dt, name="wind_speed", units="m s-1"
        ).merge_cube()

        # Set up another set of cubes which have a halo of zeros round the
        # original data. This data will be masked out in tests using a
        # landsea_mask
        base_data = np.pad(base_data, ((0, 0), (1, 1), (1, 1)), mode="constant")
        temperature_data = Unit("Celsius").convert(base_data, "Kelvin")

        # Create historic forecasts and truth
        self.historic_forecasts_halo = _create_historic_forecasts(
            temperature_data, time_dt, frt_dt, realizations=[0, 1, 2]
        )
        self.truth_halo = _create_truth(temperature_data, time_dt)

        # Create the historic and truth cubes
        self.historic_temperature_forecast_cube_halo = (
            self.historic_forecasts_halo.merge_cube()
        )
        self.temperature_truth_cube_halo = self.truth_halo.merge_cube()

        # Create a cube for testing wind speed.
        self.historic_wind_speed_forecast_cube_halo = _create_historic_forecasts(
            base_data,
            time_dt,
            frt_dt,
            realizations=[0, 1, 2],
            name="wind_speed",
            units="m s-1",
        ).merge_cube()

        self.wind_speed_truth_cube_halo = _create_truth(
            base_data, time_dt, name="wind_speed", units="m s-1"
        ).merge_cube()

        data = np.array([1.6, 1.3, 1.4, 1.1])
        altitude = np.array([10, 20, 30, 40])
        latitude = np.linspace(58.0, 59.5, 4)
        longitude = np.linspace(-0.25, 0.5, 4)
        wmo_id = ["03001", "03002", "03003", "03004"]
        forecast_spot_cubes = iris.cube.CubeList()
        for realization in range(1, 3):
            realization_coord = [
                iris.coords.DimCoord(realization, standard_name="realization")
            ]
            for day in range(5, 11):
                time_coords = construct_scalar_time_coords(
                    datetime.datetime(2017, 11, day, 4, 0),
                    None,
                    datetime.datetime(2017, 11, day, 0, 0),
                )
                time_coords = [t[0] for t in time_coords]
                forecast_spot_cubes.append(
                    build_spotdata_cube(
                        data + 0.2 * day,
                        "air_temperature",
                        "degC",
                        altitude,
                        latitude,
                        longitude,
                        wmo_id,
                        scalar_coords=time_coords + realization_coord,
                    )
                )
        forecast_spot_cube = forecast_spot_cubes.merge_cube()

        self.historic_forecast_spot_cube = forecast_spot_cube[:, :5, :]
        self.historic_forecast_spot_cube.convert_units("Kelvin")
        self.historic_forecast_spot_cube.attributes = MANDATORY_ATTRIBUTE_DEFAULTS

        self.current_forecast_spot_cube = forecast_spot_cube[:, 5, :]
        self.current_forecast_spot_cube.convert_units("Kelvin")
        self.current_forecast_spot_cube.attributes = MANDATORY_ATTRIBUTE_DEFAULTS

        self.truth_spot_cube = self.historic_forecast_spot_cube[0].copy()
        self.truth_spot_cube.remove_coord("realization")
        self.truth_spot_cube.data = self.truth_spot_cube.data + 1.0

        self.spot_altitude_cube = forecast_spot_cube[0, 0].copy(
            forecast_spot_cube.coord("altitude").points
        )
        self.spot_altitude_cube.rename("altitude")
        self.spot_altitude_cube.units = "m"
        for coord in [
            "altitude",
            "forecast_period",
            "forecast_reference_time",
            "realization",
            "time",
        ]:
            self.spot_altitude_cube.remove_coord(coord)


def _create_historic_forecasts(
    data, time_dt, frt_dt, standard_grid_metadata="uk_ens", number_of_days=5, **kwargs
):
    """
    Function to create a cubelist of historic forecasts, based on the inputs
    provided, and assuming that there will be one forecast per day at the
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
                data - 2 + 0.2 * day,
                time=new_time_dt,
                frt=new_frt_dt,
                standard_grid_metadata=standard_grid_metadata,
                **kwargs,
            )
        )
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
        truth.append(
            set_up_variable_cube(
                np.amax(data - 3, axis=0) + 0.2 * day,
                time=new_time_dt,
                frt=new_time_dt,
                standard_grid_metadata="uk_det",
                **kwargs,
            )
        )
    return truth


if __name__ == "__main__":
    pass
