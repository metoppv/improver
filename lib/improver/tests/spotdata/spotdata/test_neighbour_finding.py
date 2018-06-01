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
"""Unit tests for the spotdata.PointSelection plugin."""

from collections import OrderedDict
import unittest
import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests import IrisTest


from improver.spotdata.neighbour_finding import PointSelection as Plugin


class Test_PointSelection(IrisTest):

    """Test the point selection (grid point neighbour finding) plugin."""

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
        ancillary_data.update({'land_mask': land})

        sites = OrderedDict()
        sites.update({'100': {'latitude': 50,
                              'longitude': 0,
                              'altitude': 10,
                              'gmtoffset': 0}})

        neighbour_list = np.empty(1, dtype=[('i', 'i8'),
                                            ('j', 'i8'),
                                            ('dz', 'f8'),
                                            ('edgepoint', 'bool_')])

        self.cube = cube
        self.ancillary_data = ancillary_data
        self.sites = sites
        self.neighbour_list = neighbour_list

    def return_types(self, method, vertical_bias=None, land_constraint=False):
        """Test that the plugin returns a numpy array."""
        plugin = Plugin(method, vertical_bias, land_constraint)
        result = plugin.process(self.cube, self.sites, self.ancillary_data)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, self.neighbour_list.dtype)

    def correct_neighbour(self, method, i_expected, j_expected, dz_expected,
                          vertical_bias=None, land_constraint=False):
        """Test that the plugin returns the expected neighbour."""
        plugin = Plugin(method, vertical_bias, land_constraint)
        result = plugin.process(self.cube, self.sites, self.ancillary_data)
        self.assertEqual(result['i'], i_expected)
        self.assertEqual(result['j'], j_expected)
        self.assertEqual(result['dz'], dz_expected)

    def without_ancillary_data(self, method, vertical_bias=None,
                               land_constraint=False):
        """Test plugins behaviour with no ancillary data provided."""
        plugin = Plugin(method, vertical_bias, land_constraint)
        if method == 'fast_nearest_neighbour':
            result = plugin.process(self.cube, self.sites, {})
            self.assertIsInstance(result, np.ndarray)
        else:
            with self.assertRaises(KeyError):
                plugin.process(self.cube, self.sites, {})


class Test_miscellaneous(Test_PointSelection):
    """Miscellaneous tests"""
    def test_invalid_method(self):
        """
        Test that the plugin can handle an invalid method being passed in.

        """
        plugin = Plugin('smallest distance')
        msg = 'Unknown method'
        with self.assertRaisesRegex(AttributeError, msg):
            plugin.process(self.cube, self.sites, self.ancillary_data)

    def test_variable_no_neighbours(self):
        """
        Test that the plugin can handle a variable number of neigbours to use
        when relaxing the 'nearest' condition. Make the smallest displacement
        point 2-grid cells away, so it should be captured with no_neighbours
        set to 25.

        """
        self.ancillary_data['orography'].data[13, 10] = 10.
        plugin = Plugin(method='minimum_height_error_neighbour',
                        vertical_bias=None,
                        land_constraint=False)
        result = plugin.process(self.cube, self.sites, self.ancillary_data,
                                no_neighbours=25)
        self.assertEqual(result['i'], 13)
        self.assertEqual(result['j'], 10)
        self.assertEqual(result['dz'], 0.)

    def test_invalid_no_neighbours(self):
        """
        Test use of a larger but invalid no of neighbours over which to find
        the minimum vertical displacement.

        """
        plugin = Plugin(method='minimum_height_error_neighbour',
                        vertical_bias=None,
                        land_constraint=False)
        msg = 'Invalid nearest no'
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(self.cube, self.sites, self.ancillary_data,
                           no_neighbours=20)


class Test_fast_nearest_neighbour(Test_PointSelection):
    """
    Tests for fast_nearest_neighbour method. No other conditions beyond
    proximity are considered.

    """
    method = 'fast_nearest_neighbour'

    def test_return_type(self):
        """Ensure a numpy array of the format expected is returned."""
        self.return_types(self.method)

    def test_correct_neighbour(self):
        """Nearest neighbouring grid point with no other conditions."""
        self.correct_neighbour(self.method, 15, 10, 10.)

    def test_without_ancillary_data(self):
        """
        Should function without any ancillary fields and return expected type.
        """
        self.without_ancillary_data(self.method)


class Test_minimum_height_error_neighbour_no_bias(Test_PointSelection):
    """
    Tests for the minimum_height_error neighbour method of point selection.
    This method seeks to minimise the vertical displacement between a spotdata
    site and a neigbouring grid point.

    In this case there is no bias as to whether dz is positive (grid point
    below site) or dz is negative (grid point above site).

    """

    method = 'minimum_height_error_neighbour'

    def test_return_type(self):
        """Ensure a numpy array of the format expected is returned."""
        self.return_types(self.method)

    def test_without_ancillary_data(self):
        """
        Ensure an exception is raised if needed ancillary fields are
        missing.
        """
        self.without_ancillary_data(self.method)

    def test_correct_neighbour_no_orography(self):
        """Nearest neighbouring grid point with no other conditions."""
        self.correct_neighbour(self.method, 15, 10, 10.)

    def test_correct_neighbour_orography(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. No relative altitude bias in selection.
        """
        self.ancillary_data['orography'].data[14, 10] = 10.
        self.correct_neighbour(self.method, 14, 10, 0.)

    def test_correct_neighbour_orography_equal_displacement(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. No relative altitude bias in selection.

        In this case of equal minimum vertical grid point displacements above
        and below the site the code will select the first occurrence of this
        smallest dz that is comes across; (14, 10) is tested before (16, 10).
        """
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.correct_neighbour(self.method, 14, 10, 1.)

    def test_correct_neighbour_orography_unequal_displacement(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. No relative altitude bias in selection.

        With no vertical displacement bias the smallest of the two unequal
        dz values is chosen; -1 at (16, 10).
        """
        self.ancillary_data['orography'].data[14, 10] = 8.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.correct_neighbour(self.method, 16, 10, -1.)


class Test_minimum_height_error_neighbour_bias_above(Test_PointSelection):
    """
    Tests for the minimum_height_error neighbour method of point selection.
    This method seeks to minimise the vertical displacement between a spotdata
    site and a neigbouring grid point.

    In this case there is a bias towards dz being negative (grid point ABOVE
    site), but if this condition cannot be met, a minimum positive dz (grid
    point below site) neighbour will be returned.

    """

    method = 'minimum_height_error_neighbour'

    def test_return_type(self):
        """Ensure a numpy array of the format expected is returned."""
        self.return_types(self.method, vertical_bias='above')

    def test_without_ancillary_data(self):
        """
        Ensure an exception is raised if needed ancillary fields are
        missing.
        """
        self.without_ancillary_data(self.method, vertical_bias='above')

    def test_correct_neighbour_no_orography(self):
        """Nearest neighbouring grid point with no other conditions."""
        self.correct_neighbour(self.method, 15, 10, 10., vertical_bias='above')

    def test_correct_neighbour_orography(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes above the site if these are available.
        """
        self.ancillary_data['orography'].data[14, 10] = 10.
        self.correct_neighbour(self.method, 14, 10, 0., vertical_bias='above')

    def test_correct_neighbour_orography_equal_displacement(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes ABOVE the site if these are available.

        In this case of equal minimum vertical grid point displacements above
        and below the site the code will select the point which obeys the bias
        condition; here (16, 10) is ABOVE the site and will be chosen instead
        of (14, 10).
        """
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.correct_neighbour(self.method, 16, 10, -1., vertical_bias='above')

    def test_correct_neighbour_orography_unequal_displacement(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes ABOVE the site if these are available.

        In this case the minimum vertical grid point displacement 1 at (14, 10)
        goes against the selection bias of grid points ABOVE the site. As such
        the next nearest dz that fulfils the bias condition is chosen; -2 at
        (16, 10).
        """
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 12.
        self.correct_neighbour(self.method, 16, 10, -2., vertical_bias='above')


class Test_minimum_height_error_neighbour_bias_below(Test_PointSelection):
    """
    Tests for the minimum_height_error neighbour method of point selection.
    This method seeks to minimise the vertical displacement between a spotdata
    site and a neigbouring grid point.

    In this case there is a bias towards dz being positive (grid point BELOW
    site), but if this condition cannot be met, a minimum negative dz (grid
    point above site) neighbour will be returned.

    """

    method = 'minimum_height_error_neighbour'

    def test_return_type(self):
        """Ensure a numpy array of the format expected is returned."""
        self.return_types(self.method, vertical_bias='below')

    def test_without_ancillary_data(self):
        """
        Ensure an exception is raised if needed ancillary fields are
        missing.
        """
        self.without_ancillary_data(self.method, vertical_bias='below')

    def test_correct_neighbour_no_orography(self):
        """Nearest neighbouring grid point with no other conditions."""
        self.correct_neighbour(self.method, 15, 10, 10., vertical_bias='below')

    def test_correct_neighbour_orography(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes below the site if these are available.
        """
        self.ancillary_data['orography'].data[14, 10] = 10.
        self.correct_neighbour(self.method, 14, 10, 0., vertical_bias='below')

    def test_correct_neighbour_orography_equal_displacement(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes BELOW the site if these are available.

        In this case of equal minimum vertical grid point displacements above
        and below the site the code will select the point which obeys the bias
        condition; here (14, 10) is BELOW the site and will be chosen instead
        of (16, 10).
        """
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.correct_neighbour(self.method, 14, 10, 1., vertical_bias='below')

    def test_correct_neighbour_orography_unequal_displacement(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes BELOW the site if these are available.

        In this case the minimum vertical grid point displacement -1 at
        (16, 10) goes against the selection bias of grid points BELOW the site.
        As such the next nearest dz that fulfils the bias condition is chosen;
        2 at (14, 10).
        """
        self.ancillary_data['orography'].data[14, 10] = 8.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.correct_neighbour(self.method, 14, 10, 2., vertical_bias='below')


class Test_minimum_height_error_neighbour_land_no_bias(Test_PointSelection):
    """
    Tests for the minimum_height_error neighbour method of point selection.
    This method seeks to minimise the vertical displacement between a spotdata
    site and a neigbouring grid point.

    In this case there is no bias as to whether dz is positive (grid point
    below site) or dz is negative (grid point above site).

    A neighbouring grid point is REQUIRED to be a land point if the site's
    first guess nearest neigbour is a land point. If the first guess neighbour
    is a sea point, the site is assumed to be a sea point as well and the
    neighbour point will not be changed.

    """

    method = 'minimum_height_error_neighbour'

    def test_return_type(self):
        """Ensure a numpy array of the format expected is returned."""
        self.return_types(self.method, land_constraint=True)

    def test_without_ancillary_data(self):
        """
        Ensure an exception is raised if needed ancillary fields are
        missing.
        """
        self.without_ancillary_data(self.method, land_constraint=True)

    def test_correct_neighbour_no_orography(self):
        """Nearest neighbouring grid point with no other conditions."""
        self.correct_neighbour(self.method, 15, 10, 10., land_constraint=True)

    def test_correct_neighbour_orography(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. No relative altitude bias in selection.
        """
        self.ancillary_data['orography'].data[14, 10] = 10.
        self.correct_neighbour(self.method, 14, 10, 0., land_constraint=True)

    def test_correct_neighbour_orography_equal_displacement(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. No relative altitude bias in selection.

        In this case of equal minimum vertical grid point displacements above
        and below the site the code will select the first occurrence of this
        smallest dz that is comes across; (14, 10) is tested before (16, 10).
        """
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.correct_neighbour(self.method, 14, 10, 1., land_constraint=True)

    def test_correct_neighbour_orography_unequal_displacement(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. No relative altitude bias in selection.

        With no vertical displacement bias the smallest of the two unequal
        dz values is chosen; -1 at (16, 10).
        """
        self.ancillary_data['orography'].data[14, 10] = 8.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.correct_neighbour(self.method, 16, 10, -1., land_constraint=True)

    def test_correct_neighbour_no_orography_land(self):
        """
        Sets nearest grid point to be a sea point. Assumes site is sea point
        and leaves coordinates unchanged (dz should not vary over the sea).

        """
        self.ancillary_data['land_mask'].data[15, 10] = 0.
        self.correct_neighbour(self.method, 15, 10, 10., land_constraint=True)

    def test_correct_neighbour_orography_equal_displacement_land(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. No relative altitude bias in selection.

        This test requires a land point be selected. Any grid point not above
        land is discounted. So (14, 10) is disregarded leaving (16, 10) to give
        the smallest dz, so this point is returned as the neighbour.

        """
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.ancillary_data['land_mask'].data[14, 10] = 0.
        self.correct_neighbour(self.method, 16, 10, -1., land_constraint=True)

    def test_correct_neighbour_orography_unequal_displacement_land(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. A land point is required. No relative altitude
        bias in selection.

        (16, 10) is disregarded as it is a sea point. As such (14, 10) is
        returned as the neighbour despite its slightly larger dz.

        """
        self.ancillary_data['orography'].data[14, 10] = 8.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.ancillary_data['land_mask'].data[16, 10] = 0.
        self.correct_neighbour(self.method, 14, 10, 2., land_constraint=True)


class Test_minimum_height_error_neighbour_land_bias_above(Test_PointSelection):
    """
    Tests for the minimum_height_error neighbour method of point selection.
    This method seeks to minimise the vertical displacement between a spotdata
    site and a neigbouring grid point.

    In this case there is a bias towards dz being negative (grid point ABOVE
    site), but if this condition cannot be met, a minimum positive dz (grid
    point below site) neighbour will be returned.

    A neighbouring grid point is REQUIRED to be a land point if the site's
    first guess nearest neigbour is a land point. If the first guess neighbour
    is a sea point, the site is assumed to be a sea point as well and the
    neighbour point will not be changed.

    """

    method = 'minimum_height_error_neighbour'

    def test_return_type(self):
        """Ensure a numpy array of the format expected is returned."""
        self.return_types(self.method, vertical_bias='above',
                          land_constraint=True)

    def test_without_ancillary_data(self):
        """
        Ensure an exception is raised if needed ancillary fields are
        missing.
        """
        self.without_ancillary_data(self.method, vertical_bias='above',
                                    land_constraint=True)

    def test_correct_neighbour_no_orography(self):
        """Nearest neighbouring grid point with no other conditions."""
        self.correct_neighbour(self.method, 15, 10, 10., vertical_bias='above',
                               land_constraint=True)

    def test_correct_neighbour_orography(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes above the site if these are available.
        """
        self.ancillary_data['orography'].data[14, 10] = 10.
        self.correct_neighbour(self.method, 14, 10, 0., vertical_bias='above',
                               land_constraint=True)

    def test_correct_neighbour_orography_equal_displacement(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes ABOVE the site if these are available.

        In this case of equal minimum vertical grid point displacements above
        and below the site the code will select the point which obeys the bias
        condition; here (16, 10) is ABOVE the site and will be chosen instead
        of (14, 10).
        """
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.correct_neighbour(self.method, 16, 10, -1., vertical_bias='above',
                               land_constraint=True)

    def test_correct_neighbour_orography_unequal_displacement(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes ABOVE the site if these are available.

        In this case the minimum vertical grid point displacement 1 at (14, 10)
        goes against the selection bias of grid points ABOVE the site. As such
        the next nearest dz that fulfils the bias condition is chosen; -2 at
        (16, 10).
        """
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 12.
        self.correct_neighbour(self.method, 16, 10, -2., vertical_bias='above',
                               land_constraint=True)

    def test_correct_neighbour_no_orography_land(self):
        """
        Sets nearest grid point to be a sea point. Assumes site is sea point
        and leaves coordinates unchanged (dz should not vary over the sea).

        """
        self.ancillary_data['land_mask'].data[15, 10] = 0.
        self.correct_neighbour(self.method, 15, 10, 10., vertical_bias='above',
                               land_constraint=True)

    def test_correct_neighbour_orography_equal_displacement_land(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes ABOVE the site if these are available.

        This test requires a land point be selected. Any grid point not above
        land is discounted. So (16, 10) is disregarded leaving (14, 10) to give
        the smallest dz. This point is returned as the neighbour as no points
        fulfill the ABOVE bias condition.

        """
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.ancillary_data['land_mask'].data[16, 10] = 0.
        self.correct_neighbour(self.method, 14, 10, 1., vertical_bias='above',
                               land_constraint=True)

    def test_correct_neighbour_orography_unequal_displacement_land(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. A land point is required. Biased to prefer
        grid points with relative altitudes ABOVE the site if these are
        available.

        This test requires a land point be selected. Any grid point not above
        land is discounted. So (16, 10) is disregarded leaving (14, 10) to give
        the smallest dz. This point is returned as the neighbour as no other
        point fulfills the ABOVE bias condition.

        """
        self.ancillary_data['orography'].data[14, 10] = 8.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.ancillary_data['land_mask'].data[16, 10] = 0.
        self.correct_neighbour(self.method, 14, 10, 2., vertical_bias='above',
                               land_constraint=True)


class Test_minimum_height_error_neighbour_land_bias_below(Test_PointSelection):
    """
    Tests for the minimum_height_error neighbour method of point selection.
    This method seeks to minimise the vertical displacement between a spotdata
    site and a neigbouring grid point.

    In this case there is a bias towards dz being positive (grid point BELOW
    site), but if this condition cannot be met, a minimum negative dz (grid
    point above site) neighbour will be returned.

    A neighbouring grid point is REQUIRED to be a land point if the site's
    first guess nearest neigbour is a land point. If the first guess neighbour
    is a sea point, the site is assumed to be a sea point as well and the
    neighbour point will not be changed.

    """

    method = 'minimum_height_error_neighbour'

    def test_return_type(self):
        """Ensure a numpy array of the format expected is returned."""
        self.return_types(self.method, vertical_bias='below',
                          land_constraint=True)

    def test_without_ancillary_data(self):
        """
        Ensure an exception is raised if needed ancillary fields are
        missing.
        """
        self.without_ancillary_data(self.method, vertical_bias='below',
                                    land_constraint=True)

    def test_correct_neighbour_no_orography(self):
        """Nearest neighbouring grid point with no other conditions."""
        self.correct_neighbour(self.method, 15, 10, 10., vertical_bias='below',
                               land_constraint=True)

    def test_correct_neighbour_orography(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes below the site if these are available.
        """
        self.ancillary_data['orography'].data[14, 10] = 10.
        self.correct_neighbour(self.method, 14, 10, 0., vertical_bias='below',
                               land_constraint=True)

    def test_correct_neighbour_orography_equal_displacement(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes BELOW the site if these are available.

        In this case of equal minimum vertical grid point displacements above
        and below the site the code will select the point which obeys the bias
        condition; here (14, 10) is BELOW the site and will be chosen instead
        of (16, 10).
        """
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.correct_neighbour(self.method, 14, 10, 1., vertical_bias='below',
                               land_constraint=True)

    def test_correct_neighbour_orography_unequal_displacement(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes BELOW the site if these are available.

        In this case the minimum vertical grid point displacement -1 at
        (16, 10) goes against the selection bias of grid points BELOW the site.
        As such the next nearest dz that fulfils the bias condition is chosen;
        2 at (14, 10).
        """
        self.ancillary_data['orography'].data[14, 10] = 8.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.correct_neighbour(self.method, 14, 10, 2., vertical_bias='below',
                               land_constraint=True)

    def test_correct_neighbour_no_orography_land(self):
        """
        Sets nearest grid point to be a sea point. Assumes site is sea point
        and leaves coordinates unchanged (dz should not vary over the sea).

        """
        self.ancillary_data['land_mask'].data[15, 10] = 0.
        self.correct_neighbour(self.method, 15, 10, 10., vertical_bias='below',
                               land_constraint=True)

    def test_correct_neighbour_orography_equal_displacement_land(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. Biased to prefer grid points with relative
        altitudes BELOW the site if these are available.

        This test requires a land point be selected. Any grid point not above
        land is discounted. So (14, 10) is disregarded leaving (16, 10) to give
        the smallest dz. This point is returned as the neighbour as no points
        fulfill the BELOW bias condition.

        """
        self.ancillary_data['orography'].data = (
            self.ancillary_data['orography'].data + 20.)
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 11.
        self.ancillary_data['land_mask'].data[14, 10] = 0.
        self.correct_neighbour(self.method, 16, 10, -1., vertical_bias='below',
                               land_constraint=True)

    def test_correct_neighbour_orography_unequal_displacement_land(self):
        """
        Nearest neighbouring grid point condition relaxed to give smallest
        vertical displacement. A land point is required. Biased to prefer
        grid points with relative altitudes BELOW the site if these are
        available.

        This test requires a land point be selected. Any grid point not above
        land is discounted. So (14, 10) is disregarded leaving (16, 10) to give
        the smallest dz. This point is returned as the neighbour as no points
        fulfill the BELOW bias condition.

        """
        self.ancillary_data['orography'].data = (
            self.ancillary_data['orography'].data + 20.)
        self.ancillary_data['orography'].data[14, 10] = 9.
        self.ancillary_data['orography'].data[16, 10] = 12.
        self.ancillary_data['land_mask'].data[14, 10] = 0.
        self.correct_neighbour(self.method, 16, 10, -2., vertical_bias='below',
                               land_constraint=True)


if __name__ == '__main__':
    unittest.main()
