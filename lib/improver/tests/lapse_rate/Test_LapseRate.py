# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
"""Unit tests for the LapseRate plugin."""

import numpy as np
import unittest

import cf_units
from iris.cube import Cube
from iris.tests import IrisTest
from iris.coords import (DimCoord,
                         AuxCoord)

from improver.grids import STANDARD_GRID_CCRS

from improver.lapse_rate import LapseRate


def reset_cube_data(temperature_cube, orography_cube, land_sea_mask_cube):
    """ Resets the cube data to its defaults. """
    temperature_cube.data[:, :, :] = 0
    orography_cube.data[:] = 0
    land_sea_mask_cube.data[:] = 1


class Test__repr__(IrisTest):
    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(LapseRate())
        msg = ('<LapseRate>')
        self.assertEqual(result, msg)


class Test_calc_lapse_rate(IrisTest):
    """Test the _calc_lapse_rate function."""

    def setUp(self):
        """Sets up arrays."""

        self.temperature = np.array([280.06, 279.97, 279.90, 280.15, 280.03,
                                     279.96, 280.25, 280.33, 280.27])
        self.orography = np.array([174.67, 179.87, 188.46, 155.84, 169.58,
                                   185.05, 134.90, 144.00, 157.89])

    def test_returns_expected_values(self):
        """Test that the function returns expected lapse rate. """

        expected_out = -0.00765005774676
        result = LapseRate()._calc_lapse_rate(self.temperature, self.orography)
        self.assertArrayAlmostEqual(result, expected_out)

    def test_handles_nan(self):
        """Test that the function returns a blank value when central point
           is NaN."""

        self.temperature[4] = np.nan
        expected_out = 0.0
        result = LapseRate()._calc_lapse_rate(self.temperature, self.orography)
        self.assertArrayAlmostEqual(result, expected_out)


class Test_create_heightdiff_mask(IrisTest):
    """Test the _create_heightdiff_mask function."""

    def setUp(self):
        """Sets up arrays."""

        self.orography = np.array([[35, 40, 20, 10, 0, 10, 20, -30, -40],
                                   [35, 40, 20, 10, 0, 10, 20, -30, -40]])

    def test_returns_expected_values(self):
        """Test that the function returns True at points where the height
           difference to the central pixel is greater than 35m."""

        expected_out = np.array(
            [[True, True, False, False, False, False, False, False, True],
             [True, True, False, False, False, False, False, False, True]])

        result = LapseRate()._create_heightdiff_mask(self.orography)
        self.assertArrayAlmostEqual(result, expected_out)

    def test_change_height_thresh(self):
        """Test that the function performs as expected when the height
           difference threshold has been changed."""

        expected_out = np.array(
            [[False, True, False, False, False, False, False, False, True],
             [False, True, False, False, False, False, False, False, True]])

        result = LapseRate(max_height_diff=40)._create_heightdiff_mask(
            self.orography)
        self.assertArrayAlmostEqual(result, expected_out)


class Test_process(IrisTest):
    """Test the LapseRate processing works"""

    def setUp(self):
        """Create cubes containing a regular grid."""

        grid_size = 5
        data = np.zeros((1, grid_size, grid_size))

        realization = DimCoord([0], 'realization', units=1)
        time = DimCoord([402192.5], standard_name='time',
                        units=cf_units.Unit('hours since 1970-01-01 00:00:00',
                                            calendar='gregorian'))
        projection_y = DimCoord(np.arange(0, grid_size, 1),
                                'projection_y_coordinate',
                                units='m',
                                coord_system=STANDARD_GRID_CCRS)
        projection_x = DimCoord(np.arange(0, grid_size, 1),
                                'projection_x_coordinate',
                                units='m',
                                coord_system=STANDARD_GRID_CCRS)

        # Set up temperature cube.
        self.temperature = Cube(data, standard_name='air_temperature',
                                dim_coords_and_dims=[(realization, 0),
                                                     (projection_y, 1),
                                                     (projection_x, 2)],
                                units='K')

        self.temperature.add_aux_coord(time)
        height = AuxCoord([1.5], standard_name='height', units='m')
        self.temperature.add_aux_coord(height)
        self.temperature.attributes['institution'] = 'Met Office'

        # Copies temperature cube to create orography cube.
        self.orography = self.temperature.copy()[0]
        self.orography.remove_coord('realization')
        self.orography.rename('surface_altitude')
        self.orography.units = cf_units.Unit('m')

        # Copies orography cube to create land/sea mask cube.
        self.land_sea_mask = self.orography.copy()
        self.land_sea_mask.rename('land_binary_mask')
        self.land_sea_mask.units = cf_units.Unit('1')

    def test_basic(self):
        """Test that the plugin returns expected data type. """

        result = LapseRate().process(self.temperature, self.orography,
                                     self.land_sea_mask)
        self.assertIsInstance(result, Cube)

    def test_fails_if_data_is_not_cube(self):
        """Test code raises a Type Error if input cubes are not cubes."""
        incorrect_input = 50.0

        msg = 'Temperature input is not a cube, but {0}'.format(
            type(incorrect_input))
        with self.assertRaisesRegexp(TypeError, msg):
            LapseRate().process(incorrect_input, self.orography,
                                self.land_sea_mask)

        msg = 'Orography input is not a cube, but {0}'.format(
            type(incorrect_input))
        with self.assertRaisesRegexp(TypeError, msg):
            LapseRate().process(self.temperature, incorrect_input,
                                self.land_sea_mask)

        msg = 'Land/Sea mask input is not a cube, but {0}'.format(
            type(incorrect_input))
        with self.assertRaisesRegexp(TypeError, msg):
            LapseRate().process(self.temperature, self.orography,
                                incorrect_input)

    def test_fails_if_cube_wrong_units(self):
        """Test code raises a Value Error if input cubes are the
           wrong units."""

        # Copies cube and converts to wrong units.
        incorrect_temp = self.temperature.copy()
        incorrect_temp.convert_units('celsius')
        incorrect_orog = self.orography.copy()
        incorrect_orog.convert_units('feet')

        msg = 'Temperature cube units are {}, must be Kelvin!'.format(
           incorrect_temp.units)
        with self.assertRaisesRegexp(ValueError, msg):
            LapseRate().process(incorrect_temp, self.orography,
                                self.land_sea_mask)

        msg = 'Orography cube units are {}, must be metres!'.format(
           incorrect_orog.units)
        with self.assertRaisesRegexp(ValueError, msg):
            LapseRate().process(self.temperature, incorrect_orog,
                                self.land_sea_mask)

    def test_return_single_precision(self):
        """Test that the function returns cube of float32."""
        result = LapseRate().process(self.temperature, self.orography,
                                     self.land_sea_mask)
        self.assertEqual(result.dtype, np.float32)

    def test_constant_temp_orog(self):
        """Test that the function returns expected 0.01980198 values where the
           temperature and orography fields are constant values.
           This value is because, where the data is such that the gradient
           value is meaningless, then numpy.linalg.lstsq will calculate
           the gradient assuming that the intercept is zero.
           The mathematical validity of this approach is unclear. The FORTRAN
           code used in the operational system has statements that catch
           odd gradient values and subistuite them with DALR. This will
           require further investigation.
        """
        reset_cube_data(self.temperature, self.orography, self.land_sea_mask)

        expected_out = np.full((5, 5), 0.01980198)

        self.temperature.data[:, :, :] = 0.2
        self.orography.data[:, :] = 10

        result = LapseRate().process(self.temperature, self.orography,
                                     self.land_sea_mask)

        self.assertArrayAlmostEqual(result.data, expected_out)

    def test_handles_nan_value(self):
        """Test that the function handles a NaN temperature value by replacing
           it with zero.
        """
        reset_cube_data(self.temperature, self.orography, self.land_sea_mask)

        expected_out = np.full((5, 5), 0.01980198)
        expected_out[2, 2] = 0.0

        self.temperature.data[:, :, :] = 0.2
        self.temperature.data[:, 2, 2] = np.nan
        self.orography.data[:, :] = 10

        result = LapseRate().process(self.temperature, self.orography,
                                     self.land_sea_mask)

        self.assertArrayAlmostEqual(result.data, expected_out)

    def test_lapserate_limits(self):
        """Test that the function limits the lapse rate to +DALR and -3*DALR.
           Where DALR = Dry Adibatic Lapse Rate.
        """
        reset_cube_data(self.temperature, self.orography, self.land_sea_mask)

        expected_out = np.array([[0.0294, 0.0294, 0.0, -0.0098, -0.0098],
                                 [0.0294, 0.0294, 0.0, -0.0098, -0.0098],
                                 [0.0294, 0.0294, 0.0, -0.0098, -0.0098],
                                 [0.0294, 0.0294, 0.0, -0.0098, -0.0098],
                                 [0.0294, 0.0294, 0.0, -0.0098, -0.0098]])

        # West data points should be -3*DALR and East should be DALR.
        self.temperature.data[:, :, 0:2] = 1
        self.temperature.data[:, :, 3:5] = -1
        self.orography.data[:, :] = 10

        result = LapseRate().process(self.temperature, self.orography,
                                     self.land_sea_mask)

        self.assertArrayAlmostEqual(result.data, expected_out)

    def test_landsea_mask(self):
        """Test that the function returns zero values whereever a land/sea
           mask is true.
        """
        reset_cube_data(self.temperature, self.orography, self.land_sea_mask)

        expected_out = np.array([[0.0294, 0.0294, 0.0, -0.0098, -0.0098],
                                 [0.0294, 0.0294, 0.0, -0.0098, -0.0098],
                                 [0.0294, 0.0294, 0.0, -0.0098, -0.0098],
                                 [0.0, 0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0, 0.0]])

        # West data points should be -3*DALR and East should be DALR, South
        # should be zero.
        self.temperature.data[:, :, 0:2] = 1
        self.temperature.data[:, :, 3:5] = -1
        self.orography.data[:, :] = 10
        self.land_sea_mask.data[3:5, :] = 0

        result = LapseRate().process(self.temperature, self.orography,
                                     self.land_sea_mask)

        self.assertArrayAlmostEqual(result.data, expected_out)

    def test_decr_temp_incr_orog(self):
        """ Test code where temperature is decreasing with height. This is the
            expected scenario for lapse rate.
        """
        reset_cube_data(self.temperature, self.orography, self.land_sea_mask)

        expected_out = np.array([[0.0, -0.0098, -0.0098, -0.00642857, -0.005],
                                 [0.0, -0.0098, -0.0098, -0.00642857, -0.005],
                                 [0.0, -0.0098, -0.0098, -0.00642857, -0.005],
                                 [0.0, -0.0098, -0.0098, -0.00642857, -0.005],
                                 [0.0, -0.0098, -0.0098, -0.00642857, -0.005]])

        self.temperature.data[:, :, 0:2] = 0.4
        self.temperature.data[:, :, 2] = 0.3
        self.temperature.data[:, :, 3] = 0.2
        self.temperature.data[:, :, 4] = 0.1

        self.orography.data[:, 2] = 10
        self.orography.data[:, 3] = 20
        self.orography.data[:, 4] = 40

        result = LapseRate().process(self.temperature, self.orography,
                                     self.land_sea_mask)

        self.assertArrayAlmostEqual(result.data, expected_out)

    def test_decr_temp_decr_orog(self):
        """ Test code where the temperature increases with height.
        """
        reset_cube_data(self.temperature, self.orography, self.land_sea_mask)

        expected_out = np.array([[0.0, 0.01, 0.01, 0.00642857, 0.005],
                                 [0.0, 0.01, 0.01, 0.00642857, 0.005],
                                 [0.0, 0.01, 0.01, 0.00642857, 0.005],
                                 [0.0, 0.01, 0.01, 0.00642857, 0.005],
                                 [0.0, 0.01, 0.01, 0.00642857, 0.005]])

        self.temperature.data[:, :, 0:2] = 0.1
        self.temperature.data[:, :, 2] = 0.2
        self.temperature.data[:, :, 3] = 0.3
        self.temperature.data[:, :, 4] = 0.4

        self.orography.data[:, 2] = 10
        self.orography.data[:, 3] = 20
        self.orography.data[:, 4] = 40

        result = LapseRate().process(self.temperature, self.orography,
                                     self.land_sea_mask)

        self.assertArrayAlmostEqual(result.data, expected_out)


if __name__ == '__main__':
    unittest.main()
