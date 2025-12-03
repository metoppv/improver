# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for PystepsExtrapolate plugin"""

import datetime
import unittest

import iris
import numpy as np
import pytest

from improver.nowcasting.pysteps_advection import PystepsExtrapolate
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)


def _make_initial_rain_cube(analysis_time):
    """Construct an 8x8 masked cube of rainfall rate for testing"""

    rain_data = np.array(
        [
            [np.nan, np.nan, 0.1, 0.1, 0.1, np.nan, np.nan, np.nan],
            [np.nan, 0.1, 0.2, 0.3, 0.2, 0.1, np.nan, np.nan],
            [0.1, 0.3, 0.5, 0.6, 0.4, 0.2, 0.1, np.nan],
            [0.2, 0.6, 1.0, 1.3, 1.1, 0.5, 0.3, 0.1],
            [0.1, 0.2, 0.6, 1.0, 0.7, 0.4, 0.1, 0.0],
            [0.0, 0.1, 0.2, 0.5, 0.4, 0.1, 0.0, np.nan],
            [np.nan, 0.0, 0.1, 0.2, 0.1, 0.0, np.nan, np.nan],
            [np.nan, np.nan, 0.0, 0.1, 0.0, np.nan, np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    rain_data = np.ma.masked_invalid(rain_data)
    rain_cube = set_up_variable_cube(
        rain_data,
        name="rainfall_rate",
        units="mm/h",
        spatial_grid="equalarea",
        time=analysis_time,
        frt=analysis_time,
    )
    rain_cube.remove_coord("forecast_period")
    rain_cube.remove_coord("forecast_reference_time")
    return rain_cube


def _make_orogenh_cube(analysis_time, interval, max_lead_time):
    """Construct an orographic enhancement cube with data valid for
    every lead time"""
    orogenh_data = 0.05 * np.ones((8, 8), dtype=np.float32)
    orogenh_cube = set_up_variable_cube(
        orogenh_data,
        name="orographic_enhancement",
        units="mm/h",
        spatial_grid="equalarea",
        time=analysis_time,
        frt=analysis_time,
    )

    time_points = [analysis_time]
    lead_time = 0
    while lead_time <= max_lead_time:
        lead_time += interval
        new_point = time_points[-1] + datetime.timedelta(seconds=60 * interval)
        time_points.append(new_point)

    orogenh_cube = add_coordinate(orogenh_cube, time_points, "time", is_datetime=True)
    return orogenh_cube


class Test_process(unittest.TestCase):
    """Test wrapper for pysteps semi-Lagrangian extrapolation"""

    def setUp(self):
        """Set up test velocity and rainfall cubes"""
        # Skip if pysteps not available
        pytest.importorskip("pysteps")

        analysis_time = datetime.datetime(2019, 9, 10, 15)
        wind_data = 4 * np.ones((8, 8), dtype=np.float32)
        self.ucube = set_up_variable_cube(
            wind_data,
            name="precipitation_advection_x_velocity",
            units="m/s",
            spatial_grid="equalarea",
            time=analysis_time,
            frt=analysis_time,
        )
        self.vcube = set_up_variable_cube(
            wind_data,
            name="precipitation_advection_y_velocity",
            units="m/s",
            spatial_grid="equalarea",
            time=analysis_time,
            frt=analysis_time,
        )
        self.rain_cube = _make_initial_rain_cube(analysis_time)

        self.interval = 15
        self.max_lead_time = 120
        self.orogenh_cube = _make_orogenh_cube(
            analysis_time, self.interval, self.max_lead_time
        )
        self.plugin = PystepsExtrapolate(self.interval, self.max_lead_time)

        # set up all grids with 3.6 km spacing (1 m/s = 3.6 km/h,
        # using a 15 minute time step this is one grid square per step)
        xmin = 0
        ymin = 200000
        step = 3600
        xpoints = np.arange(xmin, xmin + 8 * step, step).astype(np.float32)
        ypoints = np.arange(ymin, ymin + 8 * step, step).astype(np.float32)
        for cube in [self.ucube, self.vcube, self.rain_cube, self.orogenh_cube]:
            cube.coord(axis="x").points = xpoints
            cube.coord(axis="y").points = ypoints

    def test_basic(self):
        """Test output is a list of cubes with expected contents and
        global attributes"""
        expected_analysis = self.rain_cube.data.copy()
        result = self.plugin(self.rain_cube, self.ucube, self.vcube, self.orogenh_cube)
        self.assertIsInstance(result, list)
        # check result is a list including a cube at the analysis time
        self.assertEqual(len(result), 9)
        self.assertIsInstance(result[0], iris.cube.Cube)
        self.assertIsInstance(result[0].data, np.ma.MaskedArray)
        self.assertEqual(result[0].data.dtype, np.float32)
        np.testing.assert_array_almost_equal(result[0].data, expected_analysis)
        # check for expected attributes
        self.assertEqual(result[0].attributes["source"], "MONOW")
        self.assertEqual(
            result[0].attributes["title"],
            "MONOW Extrapolation Nowcast on UK 2 km Standard Grid",
        )
        expected_history = (
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T" r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z: Nowcast"
        )
        self.assertRegex(result[0].attributes["history"], expected_history)

    def test_set_attributes(self):
        """Test plugin returns a cube with the specified attributes."""
        attributes_dict = {
            "mosg__grid_version": "1.0.0",
            "mosg__model_configuration": "nc_det",
            "source": "Met Office Nowcast",
            "institution": "Met Office",
            "title": "Nowcast on UK 2 km Standard Grid",
        }
        plugin = PystepsExtrapolate(self.interval, self.max_lead_time)
        result = plugin(
            self.rain_cube,
            self.ucube,
            self.vcube,
            self.orogenh_cube,
            attributes_dict=attributes_dict.copy(),
        )
        result[0].attributes.pop("history")
        self.assertEqual(result[0].attributes, attributes_dict)

    def test_time_coordinates(self):
        """Test cubelist has correct time metadata"""
        result = self.plugin(self.rain_cube, self.ucube, self.vcube, self.orogenh_cube)
        for i, cube in enumerate(result):
            # check values (and implicitly units - all seconds)
            tdiff_seconds = i * self.interval * 60
            self.assertEqual(
                cube.coord("forecast_reference_time").points[0],
                self.rain_cube.coord("time").points[0],
            )
            self.assertEqual(cube.coord("forecast_period").points[0], tdiff_seconds)
            self.assertEqual(
                cube.coord("time").points[0],
                self.rain_cube.coord("time").points[0] + tdiff_seconds,
            )

            # check datatypes
            self.assertEqual(cube.coord("time").dtype, np.int64)
            self.assertEqual(cube.coord("forecast_reference_time").dtype, np.int64)
            self.assertEqual(cube.coord("forecast_period").dtype, np.int32)

    def test_existing_time_coordinates_respected(self):
        """Test that time coordinates are not added to a cube that already
        has them"""
        expected_coords = {
            "projection_y_coordinate",
            "projection_x_coordinate",
            "forecast_period",
            "forecast_reference_time",
            "time",
        }
        frt_coord = self.rain_cube.coord("time").copy()
        frt_coord.rename("forecast_reference_time")
        self.rain_cube.add_aux_coord(frt_coord)
        self.rain_cube.add_aux_coord(
            iris.coords.AuxCoord(
                np.array([0], dtype=np.int32), "forecast_period", "seconds"
            )
        )
        result = self.plugin(self.rain_cube, self.ucube, self.vcube, self.orogenh_cube)
        result_coords = {coord.name() for coord in result[0].coords()}
        self.assertSetEqual(result_coords, expected_coords)

    def test_values_integer_step(self):
        """Test values for an advection speed of one grid square per time step
        over 8 time steps (9 output forecasts including T+0)"""
        result = self.plugin(self.rain_cube, self.ucube, self.vcube, self.orogenh_cube)
        for i, cube in enumerate(result):
            expected_data = np.full((8, 8), np.nan)
            if i == 0:
                # the first time step is the analysis field
                expected_data = self.rain_cube.data
            elif i < 8:
                # each time step advects the field by 1 grid cell along each
                # axis
                expected_data[i:, i:] = self.rain_cube.data[:-i, :-i]
            else:
                # the final step (i==8) has no data (as initialised); the
                # original rain field has been advected out of the domain
                pass
            np.testing.assert_allclose(cube.data.data, expected_data, equal_nan=True)

    def test_values_noninteger_step(self):
        """Test values for an advection speed of 0.6 grid squares per time
        step"""
        nanmatrix = np.full((8, 8), np.nan).astype(np.float32)
        # displacement at T+1 is 0.6, rounded up to 1
        expected_data_1 = nanmatrix.copy()
        expected_data_1[1:, 1:] = self.rain_cube.data[:-1, :-1]
        # displacement at T+2 is 1.2, rounded down to 1, BUT nans are advected
        # in at trailing edge
        expected_data_2 = expected_data_1.copy()
        expected_data_2[:2, :] = np.nan
        expected_data_2[:, :2] = np.nan
        # displacement at T+3 is 1.8, rounded up to 2
        expected_data_3 = nanmatrix.copy()
        expected_data_3[2:, 2:] = self.rain_cube.data[:-2, :-2]

        self.ucube.data = 0.6 * self.ucube.data
        self.vcube.data = 0.6 * self.vcube.data
        result = self.plugin(self.rain_cube, self.ucube, self.vcube, self.orogenh_cube)

        np.testing.assert_allclose(result[1].data.data, expected_data_1, equal_nan=True)
        np.testing.assert_allclose(result[2].data.data, expected_data_2, equal_nan=True)
        np.testing.assert_allclose(result[3].data.data, expected_data_3, equal_nan=True)

    def test_error_non_rate_cube(self):
        """Test plugin rejects cube of non-rate data"""
        invalid_cube = set_up_variable_cube(
            275 * np.ones((5, 5), dtype=np.float32), spatial_grid="equalarea"
        )
        msg = "air_temperature is not a precipitation rate cube"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin(invalid_cube, self.ucube, self.vcube, self.orogenh_cube)

    def test_error_unsuitable_grid(self):
        """Test plugin rejects a precipitation cube on a non-equal-area grid"""
        invalid_cube = set_up_variable_cube(
            np.ones((5, 5), dtype=np.float32), name="rainfall_rate", units="mm/h"
        )
        with self.assertRaises(ValueError):
            self.plugin(invalid_cube, self.ucube, self.vcube, self.orogenh_cube)


if __name__ == "__main__":
    unittest.main()
