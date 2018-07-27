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
from improver.constants import DALR
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
        msg = ('<LapseRate: max_height_diff: 35, nbhood_radius: 7,'
               'max_lapse_rate: 0.0294, min_lapse_rate: -0.0098>')
        self.assertEqual(result, msg)


class Test__calc_lapse_rate(IrisTest):
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
        result = LapseRate(nbhood_radius=1)._calc_lapse_rate(self.temperature,
                                                             self.orography)
        self.assertArrayAlmostEqual(result, expected_out)

    def test_handles_nan(self):
        """Test that the function returns DALR value when central point
           is NaN."""

        self.temperature[4] = np.nan
        expected_out = DALR
        result = LapseRate(nbhood_radius=1)._calc_lapse_rate(self.temperature,
                                                             self.orography)
        self.assertArrayAlmostEqual(result, expected_out)


class Test__create_heightdiff_mask(IrisTest):
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

        result = LapseRate(nbhood_radius=1)._create_heightdiff_mask(
            self.orography)
        self.assertArrayAlmostEqual(result, expected_out)

    def test_change_height_thresh(self):
        """Test that the function performs as expected when the height
           difference threshold has been changed."""

        expected_out = np.array(
            [[False, True, False, False, False, False, False, False, True],
             [False, True, False, False, False, False, False, False, True]])

        result = LapseRate(max_height_diff=40,
                           nbhood_radius=1)._create_heightdiff_mask(
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

        result = LapseRate(nbhood_radius=1).process(self.temperature,
                                                    self.orography,
                                                    self.land_sea_mask)
        self.assertIsInstance(result, Cube)

    def test_fails_if_temperature_is_not_cube(self):
        """Test code raises a Type Error if input temperature cube is
           not a cube."""
        incorrect_input = 50.0

        msg = 'Temperature input is not a cube, but {0}'.format(
            type(incorrect_input))
        with self.assertRaisesRegexp(TypeError, msg):
            LapseRate(nbhood_radius=1).process(incorrect_input,
                                               self.orography,
                                               self.land_sea_mask)

    def test_fails_if_orography_is_not_cube(self):
        """Test code raises a Type Error if input orography cube is
           not a cube."""
        incorrect_input = 50.0

        msg = 'Orography input is not a cube, but {0}'.format(
            type(incorrect_input))
        with self.assertRaisesRegexp(TypeError, msg):
            LapseRate(nbhood_radius=1).process(self.temperature,
                                               incorrect_input,
                                               self.land_sea_mask)

    def test_fails_if_land_sea_mask_is_not_cube(self):
        """Test code raises a Type Error if input land/sea mask cube is
           not a cube."""
        incorrect_input = 50.0

        msg = 'Land/Sea mask input is not a cube, but {0}'.format(
            type(incorrect_input))
        with self.assertRaisesRegexp(TypeError, msg):
            LapseRate(nbhood_radius=1).process(self.temperature,
                                               self.orography,
                                               incorrect_input)

    def test_fails_if_temperature_wrong_units(self):
        """Test code raises a Value Error if the temperature cube is the
           wrong unit."""

        #  Swap cubes around so have wrong units.
        msg = r"Unable to convert from 'Unit\('m'\)' to 'Unit\('K'\)'."
        with self.assertRaisesRegexp(ValueError, msg):
            LapseRate(nbhood_radius=1).process(self.orography, self.orography,
                                               self.land_sea_mask)

    def test_fails_if_orography_wrong_units(self):
        """Test code raises a Value Error if the orography cube is the
           wrong unit."""

        msg = r"Unable to convert from 'Unit\('K'\)' to 'Unit\('metres'\)'."
        with self.assertRaisesRegexp(ValueError, msg):
            LapseRate(nbhood_radius=1).process(self.temperature,
                                               self.temperature,
                                               self.land_sea_mask)

    def test_return_single_precision(self):
        """Test that the function returns cube of float32."""
        result = LapseRate(nbhood_radius=1).process(self.temperature,
                                                    self.orography,
                                                    self.land_sea_mask)
        self.assertEqual(result.dtype, np.float32)

    def test_constant_temp_orog(self):
        """Test that the function returns expected DALR values where the
           temperature and orography fields are constant values.
        """
        reset_cube_data(self.temperature, self.orography, self.land_sea_mask)

        expected_out = np.array([[0.0082, 0.0081, 0.0081, DALR, DALR],
                                 [0.0081, 0.008, 0.008, DALR, DALR],
                                 [0.0081, 0.008, 0.008, DALR, DALR],
                                 [DALR, DALR, DALR, DALR, DALR],
                                 [DALR, DALR, DALR, DALR, DALR]])

        self.temperature.data[:, :, :] = 0.08
        # The array should contain non DALR values around this single point
        # and DALR values elsewhere.
        self.temperature.data[:, 1, 1] = 0.09
        self.orography.data[:, :] = 10

        result = LapseRate(nbhood_radius=1).process(self.temperature,
                                                    self.orography,
                                                    self.land_sea_mask)
        self.assertArrayAlmostEqual(result.data, expected_out, decimal=4)

    def test_lapse_rate_limits(self):
        """Test that the function limits the lapse rate to +DALR and -3*DALR.
           Where DALR = Dry Adibatic Lapse Rate.
        """
        reset_cube_data(self.temperature, self.orography, self.land_sea_mask)

        expected_out = np.array([[0.0294, 0.0294, 0.0, DALR, DALR],
                                 [0.0294, 0.0294, 0.0, DALR, DALR],
                                 [0.0294, 0.0294, 0.0, DALR, DALR],
                                 [0.0294, 0.0294, 0.0, DALR, DALR],
                                 [0.0294, 0.0294, 0.0, DALR, DALR]])

        # West data points should be -3*DALR and East should be DALR.
        self.temperature.data[:, :, 0] = 2
        self.temperature.data[:, :, 1] = 1
        self.temperature.data[:, :, 3] = -1
        self.temperature.data[:, :, 4] = -2
        self.orography.data[:, :] = 10

        result = LapseRate(nbhood_radius=1).process(self.temperature,
                                                    self.orography,
                                                    self.land_sea_mask)
        self.assertArrayAlmostEqual(result.data, expected_out)

    def test_handles_nan_value(self):
        """Test that the function handles a NaN temperature value by replacing
           it with DALR.
        """
        reset_cube_data(self.temperature, self.orography, self.land_sea_mask)

        expected_out = np.array([[0.0294, 0.0294, 0.0, DALR, DALR],
                                 [0.0294, 0.0294, 0.0, DALR, DALR],
                                 [0.0294, 0.0294, DALR, DALR, DALR],
                                 [0.0294, 0.0294, 0.0, DALR, DALR],
                                 [0.0294, 0.0294, 0.0, DALR, DALR]])

        # West data points should be -3*DALR and East should be DALR.
        self.temperature.data[:, :, 0] = 2
        self.temperature.data[:, :, 1] = 1
        self.temperature.data[:, :, 3] = -1
        self.temperature.data[:, :, 4] = -2
        self.temperature.data[:, 2, 2] = np.nan
        self.orography.data[:, :] = 10

        result = LapseRate(nbhood_radius=1).process(self.temperature,
                                                    self.orography,
                                                    self.land_sea_mask)
        self.assertArrayAlmostEqual(result.data, expected_out)

    def test_landsea_mask(self):
        """Test that the function returns DALR values wherever a land/sea
           mask is true. Mask is True for land-points and False for Sea.
        """
        reset_cube_data(self.temperature, self.orography, self.land_sea_mask)

        expected_out = np.array([[0.0294, 0.0294, 0.0, DALR, DALR],
                                 [0.0294, 0.0294, 0.0, DALR, DALR],
                                 [0.0294, 0.0294, 0.0, DALR, DALR],
                                 [DALR, DALR, DALR, DALR, DALR],
                                 [DALR, DALR, DALR, DALR, DALR]])

        # West data points should be -3*DALR and East should be DALR, South
        # should be zero.
        self.temperature.data[:, :, 0] = 2
        self.temperature.data[:, :, 1] = 1
        self.temperature.data[:, :, 3] = -1
        self.temperature.data[:, :, 4] = -2
        self.orography.data[:, :] = 10
        self.land_sea_mask.data[3:5, :] = 0

        result = LapseRate(nbhood_radius=1).process(self.temperature,
                                                    self.orography,
                                                    self.land_sea_mask)
        self.assertArrayAlmostEqual(result.data, expected_out)

    def test_mask_max_height_diff(self):
        """Test that the function removes neighbours where their height
           difference from the centre point is greater than the default
           max_height_diff = 35metres.
        """
        reset_cube_data(self.temperature, self.orography, self.land_sea_mask)

        expected_out = np.array([[DALR, DALR, DALR, -0.00642857, -0.005],
                                 [DALR, DALR, DALR, -0.0065517, -0.003],
                                 [DALR, DALR, DALR, -0.0065517, 0.0],
                                 [DALR, DALR, DALR, -0.0065517, -0.003],
                                 [DALR, DALR, DALR, -0.00642857, -0.005]])

        self.temperature.data[:, :, 0:2] = 0.4
        self.temperature.data[:, :, 2] = 0.3
        self.temperature.data[:, :, 3] = 0.2
        self.temperature.data[:, :, 4] = 0.1

        self.orography.data[:, 2] = 10
        self.orography.data[:, 3] = 20
        self.orography.data[:, 4] = 40
        self.orography.data[2, 4] = 60

        result = LapseRate(nbhood_radius=1).process(self.temperature,
                                                    self.orography,
                                                    self.land_sea_mask)
        self.assertArrayAlmostEqual(result.data, expected_out)

    def test_decr_temp_incr_orog(self):
        """ Test code where temperature is decreasing with height. This is the
            expected scenario for lapse rate.
        """
        reset_cube_data(self.temperature, self.orography, self.land_sea_mask)

        expected_out = np.array([[DALR, DALR, DALR, -0.00642857, -0.005],
                                 [DALR, DALR, DALR, -0.00642857, -0.005],
                                 [DALR, DALR, DALR, -0.00642857, -0.005],
                                 [DALR, DALR, DALR, -0.00642857, -0.005],
                                 [DALR, DALR, DALR, -0.00642857, -0.005]])

        self.temperature.data[:, :, 0:2] = 0.4
        self.temperature.data[:, :, 2] = 0.3
        self.temperature.data[:, :, 3] = 0.2
        self.temperature.data[:, :, 4] = 0.1

        self.orography.data[:, 2] = 10
        self.orography.data[:, 3] = 20
        self.orography.data[:, 4] = 40

        result = LapseRate(nbhood_radius=1).process(self.temperature,
                                                    self.orography,
                                                    self.land_sea_mask)
        self.assertArrayAlmostEqual(result.data, expected_out)

    def test_decr_temp_decr_orog(self):
        """ Test code where the temperature increases with height."""

        reset_cube_data(self.temperature, self.orography, self.land_sea_mask)

        expected_out = np.array([[DALR, 0.01, 0.01, 0.00642857, 0.005],
                                 [DALR, 0.01, 0.01, 0.00642857, 0.005],
                                 [DALR, 0.01, 0.01, 0.00642857, 0.005],
                                 [DALR, 0.01, 0.01, 0.00642857, 0.005],
                                 [DALR, 0.01, 0.01, 0.00642857, 0.005]])

        self.temperature.data[:, :, 0:2] = 0.1
        self.temperature.data[:, :, 2] = 0.2
        self.temperature.data[:, :, 3] = 0.3
        self.temperature.data[:, :, 4] = 0.4

        self.orography.data[:, 2] = 10
        self.orography.data[:, 3] = 20
        self.orography.data[:, 4] = 40

        result = LapseRate(nbhood_radius=1).process(self.temperature,
                                                    self.orography,
                                                    self.land_sea_mask)
        self.assertArrayAlmostEqual(result.data, expected_out)


if __name__ == '__main__':
    unittest.main()
