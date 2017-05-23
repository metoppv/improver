# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
"""Unit tests for the spotdata.NeighbourFinding plugin."""


import unittest

from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests import IrisTest
from collections import OrderedDict
import numpy as np

from improver.spotdata.neighbour_finding import PointSelection


class TestNeighbourFinding(IrisTest):

    """Test the neighbour finding plugin."""

    def setUp(self):
        """Create a cube containing a regular lat-lon grid."""
        data = np.zeros((20, 20))
        latitudes = np.linspace(-90, 90, 20)
        longitudes = np.linspace(-180, 180, 20)
        latitude = DimCoord(latitudes, standard_name='latitude',
                            units='degrees')
        longitude = DimCoord(longitudes, standard_name='longitude',
                             units='degrees')

        cube = Cube(data,
                    long_name="test_data",
                    dim_coords_and_dims=[(latitude, 0), (longitude, 1)],
                    units="1")

        orography = cube.copy()
        orography.rename('surface_altitude')
        land = cube.copy()
        land.rename('land_binary_mask')
        land.data = land.data + 1

        ancillary_data = {}
        ancillary_data.update({'orography': orography})
        ancillary_data.update({'land': land})

        sites = OrderedDict()
        sites.update({'100': {'latitude': 50,
                              'longitude': 0,
                              'altitude': 10,
                              'gmtoffset': 0
                              }
                      })

        neighbour_list = np.empty(1, dtype=[('i', 'i8'),
                                            ('j', 'i8'),
                                            ('dz', 'f8'),
                                            ('edge', 'bool_')])

        self.cube = cube
        self.ancillary_data = ancillary_data
        self.sites = sites
        self.neighbour_list = neighbour_list

    def return_types(self, method):
        """Test that the plugin returns a numpy array."""
        plugin = PointSelection(method)
        result = plugin.process(self.cube, self.sites,
                                  ancillary_data=self.ancillary_data)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, self.neighbour_list.dtype)

    def correct_neighbour(self, method, i_expected, j_expected, dz_expected):
        """Test that the plugin returns the expected neighbour"""
        plugin = PointSelection(method)
        result = plugin.process(self.cube, self.sites,
                                  ancillary_data=self.ancillary_data)
        self.assertEqual(result['i'], i_expected)
        self.assertEqual(result['j'], j_expected)
        self.assertEqual(result['dz'], dz_expected)

    def without_ancillary_data(self, method):
        """Test plugins behaviour with no ancillary data provided"""
        plugin = PointSelection(method)
        if method == 'fast_nearest_neighbour':
            result = plugin.process(self.cube, self.sites)
            self.assertIsInstance(result, np.ndarray)
        else:
            msg = 'Ancillary data'
            with self.assertRaisesRegexp(Exception, msg):
                result = plugin.process(self.cube, self.sites)


class fast_nearest_neighbour(TestNeighbourFinding):
    '''
    Tests for fast_nearest_neighbour method. No other conditions beyond
    proximity are considered.

    '''
    method = 'fast_nearest_neighbour'

    def test_return_type(self):
        '''Ensure a numpy array of the format expected is returned.'''
        self.return_types(self.method)

    def test_correct_neighbour(self):
        '''Nearest neighbouring grid point with no other conditions'''
        self.correct_neighbour(self.method, 15, 10, 10.)

    def test_without_ancillary_data(self):
        '''
        Should function without any ancillary fields and return expected type.
        '''
        self.without_ancillary_data(self.method)


class min_dz_no_bias(TestNeighbourFinding):
    '''
    Tests for min_dz_no_bias method. This method seeks to minimise
    the vertical displacement between a site and the selected neigbouring
    grid point. There is no bias as to whether dz is positive (grid point
    below site) or dz is negative (grid point above site).

    '''

    method = 'min_dz_no_bias'

    def test_return_type(self):
        '''Ensure a numpy array of the format expected is returned.'''
        self.return_types(self.method)

    def test_without_ancillary_data(self):
        '''
        Ensure an exception is raised if needed ancillary fields are
        missing.
        '''
        self.without_ancillary_data(self.method)

    def test_correct_neighbour_no_orography(self):
        '''Nearest neighbouring grid point with no other conditions'''
        self.correct_neighbour(self.method, 15, 10, 10.)

    def test_correct_neighbour_orography(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. No relative altitude bias in selection.
        '''
        self.ancillary_data['orography'].data[14, 10] = 10.
        self.correct_neighbour(self.method, 14, 10, 0.)

    def test_correct_neighbour_orography_equal_displacement(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. No relative altitude bias in selection.

        In this case of equal minimum vertical grid point displacements above
        and below the site the code will select the first occurence of this
        smallest dz that is comes across; (14, 10) is tested before (16, 10).
        '''
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.correct_neighbour(self.method, 14, 10, 1.)

    def test_correct_neighbour_orography_unequal_displacement(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. No relative altitude bias in selection.

        With no vertical displacement bias the smallest of the two unequal
        dz values is chosen; -1 at (16, 10).
        '''
        self.ancillary_data['orography'].data[14, 10] = 8.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.correct_neighbour(self.method, 16, 10, -1.)


class min_dz_biased_above(TestNeighbourFinding):
    '''
    Tests for min_dz_biased_above. This method seeks to minimise
    the vertical displacement between a site and the selected neigbouring
    grid point. There is a bias towards dz being negative (grid point above
    site), but if this condition cannot be met, a minimum positive dz (grid
    point below site) neighbour will be returned.

    '''

    method = 'min_dz_biased_above'

    def test_return_type(self):
        '''Ensure a numpy array of the format expected is returned.'''
        self.return_types(self.method)

    def test_without_ancillary_data(self):
        '''
        Ensure an exception is raised if needed ancillary fields are
        missing.
        '''
        self.without_ancillary_data(self.method)

    def test_correct_neighbour_no_orography(self):
        '''Nearest neighbouring grid point with no other conditions'''
        self.correct_neighbour(self.method, 15, 10, 10.)

    def test_correct_neighbour_orography(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes above the site if these are available.
        '''
        self.ancillary_data['orography'].data[14, 10] = 10.
        self.correct_neighbour(self.method, 14, 10, 0.)

    def test_correct_neighbour_orography_equal_displacement(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes ABOVE the site if these are available.

        In this case of equal minimum vertical grid point displacements above
        and below the site the code will select the point which obeys the bias
        condition; here (16, 10) is ABOVE the site and will be chosen instead
        of (14, 10).
        '''
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.correct_neighbour(self.method, 16, 10, -1.)

    def test_correct_neighbour_orography_unequal_displacement(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes ABOVE the site if these are available.

        In this case the minimum vertical grid point displacement 1 at (14, 10)
        goes against the selection bias of grid points ABOVE the site. As such
        the next nearest dz that fulfils the bias condition is chosen; -2 at
        (16, 10).
        '''
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 12.
        self.correct_neighbour(self.method, 16, 10, -2.)


class min_dz_biased_below(TestNeighbourFinding):
    '''
    Tests for min_dz_biased_below. This method seeks to minimise
    the vertical displacement between a site and the selected neigbouring
    grid point. There is a bias towards dz being positive (grid point below
    site), but if this condition cannot be met, a minimum negative dz (grid
    point above site) neighbour will be returned.

    '''

    method = 'min_dz_biased_below'

    def test_return_type(self):
        '''Ensure a numpy array of the format expected is returned.'''
        self.return_types(self.method)

    def test_without_ancillary_data(self):
        '''
        Ensure an exception is raised if needed ancillary fields are
        missing.
        '''
        self.without_ancillary_data(self.method)

    def test_correct_neighbour_no_orography(self):
        '''Nearest neighbouring grid point with no other conditions'''
        self.correct_neighbour(self.method, 15, 10, 10.)

    def test_correct_neighbour_orography(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes below the site if these are available.
        '''
        self.ancillary_data['orography'].data[14, 10] = 10.
        self.correct_neighbour(self.method, 14, 10, 0.)

    def test_correct_neighbour_orography_equal_displacement(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes BELOW the site if these are available.

        In this case of equal minimum vertical grid point displacements above
        and below the site the code will select the point which obeys the bias
        condition; here (14, 10) is BELOW the site and will be chosen instead
        of (16, 10).
        '''
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.correct_neighbour(self.method, 14, 10, 1.)

    def test_correct_neighbour_orography_unequal_displacement(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes BELOW the site if these are available.

        In this case the minimum vertical grid point displacement -1 at
        (16, 10) goes against the selection bias of grid points BELOW the site.
        As such the next nearest dz that fulfils the bias condition is chosen;
        2 at (14, 10).
        '''
        self.ancillary_data['orography'].data[14, 10] = 8.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.correct_neighbour(self.method, 14, 10, 2.)


class min_dz_land_no_bias(TestNeighbourFinding):
    '''
    Tests for min_dz_land_no_bias method. This method seeks to
    minimise the vertical displacement between a site and the selected
    neigbouring grid point. There is no bias as to whether dz is positive
    (grid point below site) or dz is negative (grid point above site).

    A neighbouring grid point is REQUIRED to be a land point if the site's
    first guess nearest neigbour is a land point. If the first guess neighbour
    is a sea point, the site is assumed to be a sea point as well the
    neighbour point will not be changed.

    '''

    method = 'min_dz_land_no_bias'

    def test_return_type(self):
        '''Ensure a numpy array of the format expected is returned.'''
        self.return_types(self.method)

    def test_without_ancillary_data(self):
        '''
        Ensure an exception is raised if needed ancillary fields are
        missing.
        '''
        self.without_ancillary_data(self.method)

    def test_correct_neighbour_no_orography(self):
        '''Nearest neighbouring grid point with no other conditions'''
        self.correct_neighbour(self.method, 15, 10, 10.)

    def test_correct_neighbour_orography(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. No relative altitude bias in selection.
        '''
        self.ancillary_data['orography'].data[14, 10] = 10.
        self.correct_neighbour(self.method, 14, 10, 0.)

    def test_correct_neighbour_orography_equal_displacement(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. No relative altitude bias in selection.

        In this case of equal minimum vertical grid point displacements above
        and below the site the code will select the first occurence of this
        smallest dz that is comes across; (14, 10) is tested before (16, 10).
        '''
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.correct_neighbour(self.method, 14, 10, 1.)

    def test_correct_neighbour_orography_unequal_displacement(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. No relative altitude bias in selection.

        With no vertical displacement bias the smallest of the two unequal
        dz values is chosen; -1 at (16, 10).
        '''
        self.ancillary_data['orography'].data[14, 10] = 8.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.correct_neighbour(self.method, 16, 10, -1.)

    def test_correct_neighbour_no_orography_land(self):
        '''
        Sets nearest grid point to be a sea point. Assumes site is sea point
        and leaves coordinates unchanged (dz should not vary over the sea).

        '''
        self.ancillary_data['land'].data[15, 10] = 0.
        self.correct_neighbour(self.method, 15, 10, 10.)

    def test_correct_neighbour_orography_equal_displacement_land(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. No relative altitude bias in selection.

        This test requires a land point be selected. Any grid point not above
        land is discounted. So (14, 10) is disregarded leaving (16, 10) to give
        the smallest dz, so this point is returned as the neighbour.

        '''
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.ancillary_data['land'].data[14, 10] = 0.
        self.correct_neighbour(self.method, 16, 10, -1.)

    def test_correct_neighbour_orography_unequal_displacement_land(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. A land point is required. No relative altitude
        bias in selection.

        (16, 10) is disregarded as it is a sea point. As such (14, 10) is
        returned as the neighbour despite its slightly larger dz.

        '''
        self.ancillary_data['orography'].data[14, 10] = 8.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.ancillary_data['land'].data[16, 10] = 0.
        self.correct_neighbour(self.method, 14, 10, 2.)


class min_dz_land_biased_above(TestNeighbourFinding):
    '''
    Tests for min_dz_land_biased_above. This method seeks to minimise
    the vertical displacement between a site and the selected neigbouring
    grid point. There is a bias towards dz being negative (grid point above
    site), but if this condition cannot be met, a minimum positive dz (grid
    point below site) neighbour will be returned.

    A neighbouring grid point is REQUIRED to be a land point if the site's
    first guess nearest neigbour is a land point. If the first guess neighbour
    is a sea point, the site is assumed to be a sea point as well the
    neighbour point will not be changed.

    '''

    method = 'min_dz_land_biased_above'

    def test_return_type(self):
        '''Ensure a numpy array of the format expected is returned.'''
        self.return_types(self.method)

    def test_without_ancillary_data(self):
        '''
        Ensure an exception is raised if needed ancillary fields are
        missing.
        '''
        self.without_ancillary_data(self.method)

    def test_correct_neighbour_no_orography(self):
        '''Nearest neighbouring grid point with no other conditions'''
        self.correct_neighbour(self.method, 15, 10, 10.)

    def test_correct_neighbour_orography(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes above the site if these are available.
        '''
        self.ancillary_data['orography'].data[14, 10] = 10.
        self.correct_neighbour(self.method, 14, 10, 0.)

    def test_correct_neighbour_orography_equal_displacement(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes ABOVE the site if these are available.

        In this case of equal minimum vertical grid point displacements above
        and below the site the code will select the point which obeys the bias
        condition; here (16, 10) is ABOVE the site and will be chosen instead
        of (14, 10).
        '''
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.correct_neighbour(self.method, 16, 10, -1.)

    def test_correct_neighbour_orography_unequal_displacement(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes ABOVE the site if these are available.

        In this case the minimum vertical grid point displacement 1 at (14, 10)
        goes against the selection bias of grid points ABOVE the site. As such
        the next nearest dz that fulfils the bias condition is chosen; -2 at
        (16, 10).
        '''
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 12.
        self.correct_neighbour(self.method, 16, 10, -2.)

    def test_correct_neighbour_no_orography_land(self):
        '''
        Sets nearest grid point to be a sea point. Assumes site is sea point
        and leaves coordinates unchanged (dz should not vary over the sea).

        '''
        self.ancillary_data['land'].data[15, 10] = 0.
        self.correct_neighbour(self.method, 15, 10, 10.)

    def test_correct_neighbour_orography_equal_displacement_land(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes ABOVE the site if these are available.

        This test requires a land point be selected. Any grid point not above
        land is discounted. So (16, 10) is disregarded leaving (14, 10) to give
        the smallest dz. This point is returned as the neighbour as no points
        fulfill the ABOVE bias condition.

        '''
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.ancillary_data['land'].data[16, 10] = 0.
        self.correct_neighbour(self.method, 14, 10, 1.)

    def test_correct_neighbour_orography_unequal_displacement_land(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. A land point is required. Biased to prefer
        grid points with relative altitudes ABOVE the site if these are
        available.

        This test requires a land point be selected. Any grid point not above
        land is discounted. So (16, 10) is disregarded leaving (14, 10) to give
        the smallest dz. This point is returned as the neighbour as no other
        point fulfills the ABOVE bias condition.

        '''
        self.ancillary_data['orography'].data[14, 10] = 8.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.ancillary_data['land'].data[16, 10] = 0.
        self.correct_neighbour(self.method, 14, 10, 2.)


class min_dz_land_biased_below(TestNeighbourFinding):
    '''
    Tests for min_dz_land_biased_below. This method seeks to minimise
    the vertical displacement between a site and the selected neigbouring
    grid point. There is a bias towards dz being positive (grid point below
    site), but if this condition cannot be met, a minimum negative dz (grid
    point above site) neighbour will be returned.

    A neighbouring grid point is REQUIRED to be a land point if the site's
    first guess nearest neigbour is a land point. If the first guess neighbour
    is a sea point, the site is assumed to be a sea point as well the
    neighbour point will not be changed.

    '''

    method = 'min_dz_land_biased_below'

    def test_return_type(self):
        '''Ensure a numpy array of the format expected is returned.'''
        self.return_types(self.method)

    def test_without_ancillary_data(self):
        '''
        Ensure an exception is raised if needed ancillary fields are
        missing.
        '''
        self.without_ancillary_data(self.method)

    def test_correct_neighbour_no_orography(self):
        '''Nearest neighbouring grid point with no other conditions'''
        self.correct_neighbour(self.method, 15, 10, 10.)

    def test_correct_neighbour_orography(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes below the site if these are available.
        '''
        self.ancillary_data['orography'].data[14, 10] = 10.
        self.correct_neighbour(self.method, 14, 10, 0.)

    def test_correct_neighbour_orography_equal_displacement(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes BELOW the site if these are available.

        In this case of equal minimum vertical grid point displacements above
        and below the site the code will select the point which obeys the bias
        condition; here (14, 10) is BELOW the site and will be chosen instead
        of (16, 10).
        '''
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.correct_neighbour(self.method, 14, 10, 1.)

    def test_correct_neighbour_orography_unequal_displacement(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes BELOW the site if these are available.

        In this case the minimum vertical grid point displacement -1 at
        (16, 10) goes against the selection bias of grid points BELOW the site.
        As such the next nearest dz that fulfils the bias condition is chosen;
        2 at (14, 10).
        '''
        self.ancillary_data['orography'].data[14, 10] = 8.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.correct_neighbour(self.method, 14, 10, 2.)

    def test_correct_neighbour_no_orography_land(self):
        '''
        Sets nearest grid point to be a sea point. Assumes site is sea point
        and leaves coordinates unchanged (dz should not vary over the sea).

        '''
        self.ancillary_data['land'].data[15, 10] = 0.
        self.correct_neighbour(self.method, 15, 10, 10.)

    def test_correct_neighbour_orography_equal_displacement_land(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes BELOW the site if these are available.

        This test requires a land point be selected. Any grid point not above
        land is discounted. So (14, 10) is disregarded leaving (16, 10) to give
        the smallest dz. This point is returned as the neighbour as no points
        fulfill the BELOW bias condition.

        '''
        self.ancillary_data['orography'].data = (
            self.ancillary_data['orography'].data + 20.)
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.ancillary_data['land'].data[14, 10] = 0.
        self.correct_neighbour(self.method, 16, 10, -1.)

    def test_correct_neighbour_orography_unequal_displacement_land(self):
        '''
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. A land point is required. Biased to prefer
        grid points with relative altitudes BELOW the site if these are
        available.

        This test requires a land point be selected. Any grid point not above
        land is discounted. So (14, 10) is disregarded leaving (16, 10) to give
        the smallest dz. This point is returned as the neighbour as no points
        fulfill the BELOW bias condition.

        '''
        self.ancillary_data['orography'].data = (
            self.ancillary_data['orography'].data + 20.)
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 12.
        self.ancillary_data['land'].data[14, 10] = 0.
        self.correct_neighbour(self.method, 16, 10, -2.)


if __name__ == '__main__':
    unittest.main()
