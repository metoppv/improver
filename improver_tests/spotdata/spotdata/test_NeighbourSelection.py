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
"""Unit tests for NeighbourSelection class"""

import unittest

import cartopy.crs as ccrs
import iris
import numpy as np
import scipy
from iris.tests import IrisTest

from improver.metadata.utilities import create_coordinate_hash
from improver.spotdata.neighbour_finding import NeighbourSelection
from improver.utilities.warnings_handler import ManageWarnings


class Test_NeighbourSelection(IrisTest):

    """Test class for the NeighbourSelection tests, setting up inputs."""

    def setUp(self):
        """Set up cubes and sitelists for use in testing NeighbourSelection"""
        # Set up orography and land mask data
        land_data = np.zeros((9, 9))
        land_data[0:2, 4] = 1
        land_data[4, 4] = 1
        orography_data = np.zeros((9, 9))
        orography_data[0, 4] = 1
        orography_data[1, 4] = 5

        # Global coordinates and cubes
        projection = iris.coord_systems.GeogCS(6371229.0)
        xcoord = iris.coords.DimCoord(
            np.linspace(-160, 160, 9), standard_name='longitude',
            units='degrees', coord_system=projection,
            circular=True)
        xcoord.guess_bounds()
        ycoord = iris.coords.DimCoord(
            np.linspace(-80, 80, 9), standard_name='latitude',
            units='degrees', coord_system=projection,
            circular=False)
        ycoord.guess_bounds()

        global_land_mask = iris.cube.Cube(
            land_data, standard_name="land_binary_mask", units=1,
            dim_coords_and_dims=[(ycoord, 1), (xcoord, 0)])
        global_orography = iris.cube.Cube(
            orography_data, standard_name="surface_altitude", units='m',
            dim_coords_and_dims=[(ycoord, 1), (xcoord, 0)])

        # Regional grid coordinates and cubes
        projection = iris.coord_systems.LambertAzimuthalEqualArea(
            ellipsoid=iris.coord_systems.GeogCS(
                semi_major_axis=6378137.0, semi_minor_axis=6356752.314140356))
        xcoord = iris.coords.DimCoord(
            np.linspace(-1E5, 1E5, 9), standard_name='projection_x_coordinate',
            units='m', coord_system=projection)
        xcoord.guess_bounds()
        ycoord = iris.coords.DimCoord(
            np.linspace(-5E4, 5E4, 9), standard_name='projection_y_coordinate',
            units='degrees', coord_system=projection)
        ycoord.guess_bounds()

        region_land_mask = iris.cube.Cube(
            land_data, standard_name="land_binary_mask", units=1,
            dim_coords_and_dims=[(ycoord, 1), (xcoord, 0)])
        region_orography = iris.cube.Cube(
            orography_data, standard_name="surface_altitude", units='m',
            dim_coords_and_dims=[(ycoord, 1), (xcoord, 0)])

        # Create site lists
        self.global_sites = [
            {'altitude': 2.0, 'latitude': 0.0, 'longitude': -64.0,
             'wmo_id': 1}]
        self.region_sites = [
            {'altitude': 2.0, 'projection_x_coordinate': -4.0E4,
             'projection_y_coordinate': 0.0, 'wmo_id': 1}]

        self.global_land_mask = global_land_mask
        self.global_orography = global_orography
        self.region_land_mask = region_land_mask
        self.region_orography = region_orography
        self.region_projection = projection


class Test__repr__(IrisTest):

    """Tests the class __repr__ function."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string with defaults."""
        plugin = NeighbourSelection()
        result = str(plugin)
        msg = ("<NeighbourSelection: land_constraint: False, minimum_dz: False"
               ", search_radius: 10000.0, site_coordinate_system: <class "
               "'cartopy.crs.PlateCarree'>, site_x_coordinate:longitude, "
               "site_y_coordinate: latitude, node_limit: 36>")
        self.assertEqual(result, msg)

    def test_non_default(self):
        """Test that the __repr__ returns the expected string with defaults."""
        plugin = NeighbourSelection(land_constraint=True, minimum_dz=True,
                                    search_radius=1000,
                                    site_coordinate_system=ccrs.Mercator(),
                                    site_x_coordinate='x_axis',
                                    site_y_coordinate='y_axis',
                                    node_limit=100)
        result = str(plugin)
        msg = ("<NeighbourSelection: land_constraint: True, minimum_dz: True,"
               " search_radius: 1000, site_coordinate_system: <class "
               "'cartopy.crs.Mercator'>, site_x_coordinate:x_axis, "
               "site_y_coordinate: y_axis, node_limit: 100>")
        self.assertEqual(result, msg)


class Test_neighbour_finding_method_name(IrisTest):

    """Test the function for generating the name that describes the neighbour
    finding method."""

    def test_nearest(self):
        """Test name generated when using the default nearest neighbour
        method."""
        plugin = NeighbourSelection()
        expected = 'nearest'
        result = plugin.neighbour_finding_method_name()
        self.assertEqual(result, expected)

    def test_nearest_land(self):
        """Test name generated when using the nearest land neighbour
        method."""
        plugin = NeighbourSelection(land_constraint=True)
        expected = 'nearest_land'
        result = plugin.neighbour_finding_method_name()
        self.assertEqual(result, expected)

    def test_nearest_land_minimum_dz(self):
        """Test name generated when using the nearest land neighbour
        with smallest vertical displacment method."""
        plugin = NeighbourSelection(land_constraint=True, minimum_dz=True)
        expected = 'nearest_land_minimum_dz'
        result = plugin.neighbour_finding_method_name()
        self.assertEqual(result, expected)

    def test_nearest_minimum_dz(self):
        """Test name generated when using the nearest neighbour with the
        smallest vertical displacment method."""
        plugin = NeighbourSelection(minimum_dz=True)
        expected = 'nearest_minimum_dz'
        result = plugin.neighbour_finding_method_name()
        self.assertEqual(result, expected)


class Test__transform_sites_coordinate_system(Test_NeighbourSelection):

    """Test the function for converting arrays of site coordinates into the
    correct coordinate system for the model/grid cube."""

    def test_global_to_region(self):
        """Test coordinates generated when transforming from a global to
        regional coordinate system, in this case PlateCarree to Lambert
        Azimuthal Equal Areas."""
        plugin = NeighbourSelection()
        x_points = np.array([0, 10, 20])
        y_points = np.array([0, 0, 10])
        expected = [[0., 0.], [1111782.53516264, 0.],
                    [2189747.33076441, 1121357.32401753]]
        result = plugin._transform_sites_coordinate_system(
            x_points, y_points,
            self.region_orography.coord_system().as_cartopy_crs())
        self.assertArrayAlmostEqual(result, expected)

    def test_region_to_global(self):
        """Test coordinates generated when transforming from a regional to
        global coordinate system, in this case Lambert Azimuthal Equal Areas
        to PlateCarree."""
        plugin = NeighbourSelection(
            site_coordinate_system=self.region_projection.as_cartopy_crs())
        x_points = np.array([0, 1, 2])
        y_points = np.array([0, 0, 1])
        expected = [[0., 0.], [8.98315284e-06, 0.],
                    [1.79663057e-05, 9.04369476e-06]]
        result = plugin._transform_sites_coordinate_system(
            x_points, y_points,
            self.global_orography.coord_system().as_cartopy_crs())
        self.assertArrayAlmostEqual(result, expected)

    def test_global_to_global(self):
        """Test coordinates generated when the input and output coordinate
        systems are the same, in this case Plate-Carree."""
        plugin = NeighbourSelection()
        x_points = np.array([0, 10, 20])
        y_points = np.array([0, 0, 10])
        expected = np.stack((x_points, y_points), axis=1)
        result = plugin._transform_sites_coordinate_system(
            x_points, y_points,
            self.global_orography.coord_system().as_cartopy_crs())

        self.assertArrayAlmostEqual(result, expected)

    def test_region_to_region(self):
        """Test coordinates generated when the input and output coordinate
        systems are the same, in this case Lambert Azimuthal Equal Areas."""
        plugin = NeighbourSelection(
            site_coordinate_system=self.region_projection.as_cartopy_crs())
        x_points = np.array([0, 1, 2])
        y_points = np.array([0, 0, 1])
        expected = np.stack((x_points, y_points), axis=1)
        result = plugin._transform_sites_coordinate_system(
            x_points, y_points,
            self.region_orography.coord_system().as_cartopy_crs())

        self.assertArrayAlmostEqual(result, expected)


class Test_check_sites_are_within_domain(Test_NeighbourSelection):

    """Test the function that removes sites falling outside the model domain
    from the site list and raises a warning."""

    def test_all_valid(self):
        """Test case in which all sites are valid and fall within domain."""
        plugin = NeighbourSelection()
        sites = [{'projection_x_coordinate': 1.0E4,
                  'projection_y_coordinate': 1.0E4},
                 {'projection_x_coordinate': 1.0E5,
                  'projection_y_coordinate': 5.0E4}]
        x_points = np.array(
            [site['projection_x_coordinate'] for site in sites])
        y_points = np.array(
            [site['projection_y_coordinate'] for site in sites])
        site_coords = np.stack((x_points, y_points), axis=1)

        sites_out, site_coords_out, out_x, out_y = (
            plugin.check_sites_are_within_domain(
                sites, site_coords, x_points, y_points,
                self.region_orography))

        self.assertArrayEqual(sites_out, sites)
        self.assertArrayEqual(site_coords_out, site_coords)
        self.assertArrayEqual(out_x, x_points)
        self.assertArrayEqual(out_y, y_points)

    @ManageWarnings(record=True)
    def test_some_invalid(self, warning_list=None):
        """Test case with some sites falling outside the regional domain."""
        plugin = NeighbourSelection()
        sites = [{'projection_x_coordinate': 1.0E4,
                  'projection_y_coordinate': 1.0E4},
                 {'projection_x_coordinate': 1.0E5,
                  'projection_y_coordinate': 5.0E4},
                 {'projection_x_coordinate': 1.0E6,
                  'projection_y_coordinate': 1.0E5}]

        x_points = np.array(
            [site['projection_x_coordinate'] for site in sites])
        y_points = np.array(
            [site['projection_y_coordinate'] for site in sites])
        site_coords = np.stack((x_points, y_points), axis=1)

        sites_out, site_coords_out, out_x, out_y = (
            plugin.check_sites_are_within_domain(
                sites, site_coords, x_points, y_points,
                self.region_orography))

        self.assertArrayEqual(sites_out, sites[0:2])
        self.assertArrayEqual(site_coords_out[0:2], site_coords[0:2])
        self.assertArrayEqual(out_x, x_points[0:2])
        self.assertArrayEqual(out_y, y_points[0:2])

        msg = "1 spot sites fall outside the grid"
        self.assertTrue(any([msg in str(warning) for warning in warning_list]))
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))

    @ManageWarnings(record=True)
    def test_global_invalid(self, warning_list=None):
        """Test case with some sites falling outside the global domain."""
        plugin = NeighbourSelection()
        sites = [
            {'latitude': 0.0, 'longitude': 0.0},
            {'latitude': 50.0, 'longitude': 0.0},
            {'latitude': 100.0, 'longitude': 0.0}]

        x_points = np.array(
            [site['longitude'] for site in sites])
        y_points = np.array(
            [site['latitude'] for site in sites])
        site_coords = np.stack((x_points, y_points), axis=1)

        plugin.global_coordinate_system = True

        sites_out, site_coords_out, out_x, out_y = (
            plugin.check_sites_are_within_domain(
                sites, site_coords, x_points, y_points,
                self.global_orography))

        self.assertArrayEqual(sites_out, sites[0:2])
        self.assertArrayEqual(site_coords_out[0:2], site_coords[0:2])
        self.assertArrayEqual(out_x, x_points[0:2])
        self.assertArrayEqual(out_y, y_points[0:2])

        msg = "1 spot sites fall outside the grid"
        self.assertTrue(any([msg in str(warning) for warning in warning_list]))
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))

    def test_global_circular_valid(self):
        """Test case with a site defined using a longitide exceeding 180
        degrees (e.g. with longitudes that run 0 to 360) is still included
        as the circular x-coordinate means it will still be used correctly."""
        plugin = NeighbourSelection()
        sites = [
            {'latitude': 0.0, 'longitude': 100.0},
            {'latitude': 30.0, 'longitude': 200.0},
            {'latitude': 60.0, 'longitude': 300.0}]

        x_points = np.array(
            [site['longitude'] for site in sites])
        y_points = np.array(
            [site['latitude'] for site in sites])
        site_coords = np.stack((x_points, y_points), axis=1)

        plugin.global_coordinate_system = True

        sites_out, site_coords_out, out_x, out_y = (
            plugin.check_sites_are_within_domain(
                sites, site_coords, x_points, y_points,
                self.global_orography))

        self.assertArrayEqual(sites_out, sites)
        self.assertArrayEqual(site_coords_out, site_coords)
        self.assertArrayEqual(out_x, x_points)
        self.assertArrayEqual(out_y, y_points)


class Test_get_nearest_indices(Test_NeighbourSelection):

    """Test function wrapping iris functionality to get nearest grid point
    indices to arbitrary coordinates."""

    def test_basic(self):
        """Test that expected coordinates are returned."""

        plugin = NeighbourSelection()

        x_points = np.array([site['projection_x_coordinate']
                             for site in self.region_sites])
        y_points = np.array([site['projection_y_coordinate']
                             for site in self.region_sites])
        site_coords = np.stack((x_points, y_points), axis=1)

        expected = [[2, 4]]
        result = plugin.get_nearest_indices(site_coords,
                                            self.region_orography)
        self.assertArrayEqual(result, expected)


class Test_geocentric_cartesian(Test_NeighbourSelection):

    """Test conversion of global coordinates to geocentric cartesians. In  this
    coordinate system, x and y are in the equitorial plane, and z is towards
    the poles."""

    def test_basic(self):
        """Test a (0, 0) coordinate conversion to geocentric cartesian. This is
        expected to give an x coordinate which is the semi-major axis of the
        globe defined in the global coordinate system."""

        plugin = NeighbourSelection()
        x_points = np.array([0])
        y_points = np.array([0])
        result = plugin.geocentric_cartesian(self.global_orography,
                                             x_points, y_points)
        radius = self.global_orography.coord_system().semi_major_axis
        expected = [[radius, 0, 0]]
        self.assertArrayAlmostEqual(result, expected)

    def test_north_pole(self):
        """Test a (0, 90) coordinate conversion to geocentric cartesian, this
        being the north pole. This is expected to give an x coordinate which 0
        and a z coordinate equivalent to the semi-major axis of the globe
        defined in the global coordinate system."""

        plugin = NeighbourSelection()
        x_points = np.array([0])
        y_points = np.array([90])
        result = plugin.geocentric_cartesian(self.global_orography,
                                             x_points, y_points)
        radius = self.global_orography.coord_system().semi_major_axis
        expected = [[0, 0, radius]]
        self.assertArrayAlmostEqual(result, expected)

    def test_45_degrees_latitude(self):
        """Test a (0, 45) coordinate conversion to geocentric cartesian. In
        this case the components of the semi-major axis of the globe
        defined in the global coordinate system should be shared between the
        resulting x and z coordinates."""

        plugin = NeighbourSelection()
        x_points = np.array([0])
        y_points = np.array([45])
        result = plugin.geocentric_cartesian(self.global_orography,
                                             x_points, y_points)
        radius = self.global_orography.coord_system().semi_major_axis
        component = radius/np.sqrt(2.)
        expected = [[component, 0, component]]

        self.assertArrayAlmostEqual(result, expected)

    def test_45_degrees_longitude(self):
        """Test a (45, 0) coordinate conversion to geocentric cartesian. In
        this case the components of the semi-major axis of the globe
        defined in the global coordinate system should be shared between the
        resulting x and y coordinates."""

        plugin = NeighbourSelection()
        x_points = np.array([45])
        y_points = np.array([0])
        result = plugin.geocentric_cartesian(self.global_orography,
                                             x_points, y_points)
        radius = self.global_orography.coord_system().semi_major_axis
        component = radius/np.sqrt(2.)
        expected = [[component, component, 0]]

        self.assertArrayAlmostEqual(result, expected)

    def test_45_degrees_latitude_and_longitude(self):
        """Test a (45, 45) coordinate conversion to geocentric cartesian. In
        this case the z component should be a cos(45) component of the
        semi-major axis of the globe defined in the global coordinate system.
        The x and y coordinates should be cos(45) components of the remaining
        cos(45) component of the semi-major axis."""

        plugin = NeighbourSelection()
        x_points = np.array([45])
        y_points = np.array([45])
        result = plugin.geocentric_cartesian(self.global_orography,
                                             x_points, y_points)
        radius = self.global_orography.coord_system().semi_major_axis
        component = radius/np.sqrt(2.)
        sub_component = component/np.sqrt(2.)
        expected = [[sub_component, sub_component, component]]

        self.assertArrayAlmostEqual(result, expected)

    def test_negative_45_degrees_latitude_and_longitude(self):
        """Test a (-45, -45) coordinate conversion to geocentric cartesian.
        In this case the x is expected to remain positive, whilst y and z
        become negative."""

        plugin = NeighbourSelection()
        x_points = np.array([-45])
        y_points = np.array([-45])
        result = plugin.geocentric_cartesian(self.global_orography,
                                             x_points, y_points)
        radius = self.global_orography.coord_system().semi_major_axis
        component = radius/np.sqrt(2.)
        sub_component = component/np.sqrt(2.)
        expected = [[sub_component, -sub_component, -component]]

        self.assertArrayAlmostEqual(result, expected)


class Test_build_KDTree(Test_NeighbourSelection):

    """Test construction of a KDTree with scipy."""

    def test_basic(self):
        """Test that the expected number of nodes are created and that a tree
        is returned; this should be the lengths of the x and y coordinates
        multiplied in the simple case."""

        plugin = NeighbourSelection()
        result, result_nodes = plugin.build_KDTree(self.region_land_mask)
        expected_length = (self.region_land_mask.shape[0] *
                           self.region_land_mask.shape[1])

        self.assertEqual(
            result_nodes.shape[0],  # pylint: disable=unsubscriptable-object
            expected_length)
        self.assertIsInstance(result, scipy.spatial.ckdtree.cKDTree)

    def test_only_land(self):
        """Test that the expected number of nodes are created and that a tree
        is returned. In this case the number of nodes should be this should be
        equal to the number of land points."""

        plugin = NeighbourSelection(land_constraint=True)
        result, result_nodes = plugin.build_KDTree(self.region_land_mask)
        expected_length = np.nonzero(self.region_land_mask.data)[0].shape[0]

        self.assertEqual(
            result_nodes.shape[0],  # pylint: disable=unsubscriptable-object
            expected_length)
        self.assertIsInstance(result, scipy.spatial.ckdtree.cKDTree)


class Test_select_minimum_dz(Test_NeighbourSelection):

    """Test extraction of the minimum height difference points from a provided
    array of neighbours. Note that the region orography has a series of islands
    at a y index of 4, changing elevation with x. As such the nodes are chosen
    along this line, e.g. [0, 4], [1, 4], etc."""

    @ManageWarnings(ignored_messages=["Limit on number of nearest neighbours"])
    def test_basic(self):
        """Test a simple case where the first element in the provided lists
        has the smallest vertical displacement to the site. Expect the
        coordinates of the first node to be returned."""

        plugin = NeighbourSelection()
        site_altitude = 3.
        nodes = np.array([[0, 4], [1, 4], [2, 4], [3, 4], [4, 4]])
        distance = np.arange(5)
        indices = np.arange(5)

        result = plugin.select_minimum_dz(self.region_orography,
                                          site_altitude, nodes,
                                          distance, indices)
        self.assertArrayEqual(result, nodes[0])

    def test_some_invalid_points(self):
        """Test a case where some nodes are beyond the imposed search_radius,
        which means they have a distance of np.inf, ensuring this is handled.
        Also change the site height so the second node is the expected
        result."""

        plugin = NeighbourSelection()
        site_altitude = 5.
        nodes = np.array([[0, 4], [1, 4], [2, 4], [3, 4], [4, 4]])
        distance = np.array([0, 1, 2, 3, np.inf])
        indices = np.arange(5)

        result = plugin.select_minimum_dz(self.region_orography,
                                          site_altitude, nodes,
                                          distance, indices)
        self.assertArrayEqual(result, nodes[1])

    def test_all_invalid_points(self):
        """Test a case where all nodes are beyond the imposed search_radius,
        so the returned value should be None."""

        plugin = NeighbourSelection()
        site_altitude = 5.
        nodes = np.array([[0, 4], [1, 4], [2, 4], [3, 4], [4, 4]])
        distance = np.full(5, np.inf)
        indices = np.arange(5)

        result = plugin.select_minimum_dz(self.region_orography,
                                          site_altitude, nodes,
                                          distance, indices)
        self.assertEqual(result, None)

    @ManageWarnings(record=True)
    def test_incomplete_search(self, warning_list=None):
        """Test a warning is raised when the number of nearest neighbours
        searched for the minimum dz neighbour does not exhaust the
        search_radius."""

        plugin = NeighbourSelection(search_radius=6)
        site_altitude = 3.
        nodes = np.array([[0, 4], [1, 4], [2, 4], [3, 4], [4, 4]])
        distance = np.arange(5)
        indices = np.arange(5)

        plugin.select_minimum_dz(self.region_orography,
                                 site_altitude, nodes,
                                 distance, indices)

        msg = "Limit on number of nearest neighbours"
        self.assertTrue(any([msg in str(warning) for warning in warning_list]))
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))


class Test_process(Test_NeighbourSelection):

    """Test the process method of the NeighbourSelection class."""

    def test_non_metre_spatial_dimensions(self):
        """Test an error is raised if a regional grid is provided for which the
        spatial coordinates do not have units of metres."""

        self.region_orography.coord(axis='x').convert_units('feet')
        msg = 'Cube spatial coordinates for regional grids'

        plugin = NeighbourSelection()
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(self.region_sites, self.region_orography,
                           self.region_land_mask)

    def test_different_cube_grids(self):
        """Test an error is raised if the land mask and orography cubes
        provided are on different grids."""

        msg = 'Orography and land_mask cubes are not on the same'
        plugin = NeighbourSelection()
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(self.region_sites, self.region_orography,
                           self.global_land_mask)

    def test_global_attribute(self):
        """Test that a cube is returned with a model_grid_hash that matches
        that of the global input grids."""

        expected = create_coordinate_hash(self.global_orography)
        plugin = NeighbourSelection()
        result = plugin.process(self.global_sites, self.global_orography,
                                self.global_land_mask)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.attributes['model_grid_hash'], expected)

    def test_wmo_ids(self):
        """Test that the returned cube has the wmo_ids present when they are
        available. Should be None when they are not provided."""

        plugin = NeighbourSelection()
        sites = self.global_sites + [self.global_sites.copy()[0].copy()]
        sites[1]['wmo_id'] = None
        expected = ['1', 'None']

        result = plugin.process(sites, self.global_orography,
                                self.global_land_mask)

        self.assertArrayEqual(result.coord('wmo_id').points, expected)

    def test_global_nearest(self):
        """Test that a cube is returned, here using a conventional site list
        with lat/lon site coordinates. Neighbour coordinates of [2, 4] are
        expected, with a vertical displacement of 2.."""

        plugin = NeighbourSelection()
        result = plugin.process(self.global_sites, self.global_orography,
                                self.global_land_mask)
        expected = [[[2, 4, 2]]]

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.data, expected)

    def test_global_returned_site_coordinates(self):
        """Test that the site coordinates in the returned neighbour cube are
        latitudes and longitudes. Here the input site list contains site
        coordinates defined in latitudes and longitudes, so they should be
        unchanged."""

        latitude_expected = np.array([self.global_sites[0]['latitude']],
                                     dtype=np.float32)
        longitude_expected = np.array([self.global_sites[0]['longitude']],
                                      dtype=np.float32)
        plugin = NeighbourSelection()
        result = plugin.process(self.global_sites, self.global_orography,
                                self.global_land_mask)

        self.assertIsNotNone(result.coord('latitude'))
        self.assertIsNotNone(result.coord('longitude'))
        self.assertArrayAlmostEqual(result.coord('latitude').points,
                                    latitude_expected)
        self.assertArrayAlmostEqual(result.coord('longitude').points,
                                    longitude_expected)

    def test_global_land(self):
        """Test how the neighbour index changes when a land constraint is
        imposed. Here we expect to go 'west' to the first band of land
        which has an altitude of 5m. So we expect [1, 4, -3]."""

        plugin = NeighbourSelection(land_constraint=True, search_radius=1E7)
        result = plugin.process(self.global_sites, self.global_orography,
                                self.global_land_mask)
        expected = [[[1, 4, -3]]]

        self.assertArrayEqual(result.data, expected)

    def test_global_land_minimum_dz(self):
        """Test how the neighbour index changes when a land constraint is
        imposed and a minimum height difference condition. Here we expect to go
        'west' to the second band of land that we encounter, which has an
        altitude closer to that of the site. So we expect [0, 4, 1]."""

        plugin = NeighbourSelection(land_constraint=True, minimum_dz=True,
                                    search_radius=1E8)
        result = plugin.process(self.global_sites, self.global_orography,
                                self.global_land_mask)
        expected = [[[0, 4, 1]]]

        self.assertArrayEqual(result.data, expected)

    def test_global_dateline(self):
        """Test that for a global grid with a circular longitude coordinate,
        the code selects the nearest neighbour matching constraints even if it
        falls at the opposite edge of the grid. The spot site is nearest to
        grid point [6, 4], and the nearest land point is at [4, 4]. However
        by imposing a minimum vertical displacement constraint the code will
        return a point across the dateline at [0, 4]. We can be sure we have
        crossed the dateline by the fact that there is an island of land with
        the same vertical displacment to the spot site between the point and
        the grid point returned. Therefore, the short path must be across the
        dateline, rather than across this island travelling west."""

        self.global_sites[0]['longitude'] = 64.
        self.global_sites[0]['altitude'] = 3.

        plugin = NeighbourSelection(land_constraint=True, minimum_dz=True,
                                    search_radius=1E8)
        result = plugin.process(self.global_sites, self.global_orography,
                                self.global_land_mask)
        expected = [[[0, 4, 2]]]

        self.assertArrayEqual(result.data, expected)

    def test_region_attribute(self):
        """Test that a cube is returned with a model_grid_hash that matches
        that of the regional input grids."""

        expected = create_coordinate_hash(self.region_orography)
        plugin = NeighbourSelection(
            site_coordinate_system=self.region_projection.as_cartopy_crs(),
            site_x_coordinate='projection_x_coordinate',
            site_y_coordinate='projection_y_coordinate')
        result = plugin.process(self.region_sites, self.region_orography,
                                self.region_land_mask)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.attributes['model_grid_hash'], expected)

    def test_region_nearest(self):
        """Test that a cube is returned, this time using the site list in
        which site coordinates are defined in metres in an equal areas
        projection. Neighbour coordinates of [2, 4] are expected, with a
        vertical displacement of 2."""

        plugin = NeighbourSelection(
            site_coordinate_system=self.region_projection.as_cartopy_crs(),
            site_x_coordinate='projection_x_coordinate',
            site_y_coordinate='projection_y_coordinate')
        result = plugin.process(self.region_sites, self.region_orography,
                                self.region_land_mask)
        expected = [[[2, 4, 2]]]

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.data, expected)

    def test_region_returned_site_coordinates(self):
        """Test that the site coordinates in the returned neighbour cube are
        latitudes and longitudes. Here the input site list contains site
        coordinates defined in metres in an equal areas projection."""

        plugin = NeighbourSelection(
            site_coordinate_system=self.region_projection.as_cartopy_crs(),
            site_x_coordinate='projection_x_coordinate',
            site_y_coordinate='projection_y_coordinate')
        result = plugin.process(self.region_sites, self.region_orography,
                                self.region_land_mask)
        latitude_expected = np.array([0], dtype=np.float32)
        longitude_expected = np.array([-0.359327], dtype=np.float32)

        self.assertIsNotNone(result.coord('latitude'))
        self.assertIsNotNone(result.coord('longitude'))
        self.assertArrayAlmostEqual(result.coord('latitude').points,
                                    latitude_expected)
        self.assertArrayAlmostEqual(result.coord('longitude').points,
                                    longitude_expected)

    def test_region_land(self):
        """Test how the neighbour index changes when a land constraint is
        imposed. Here we expect to go 'west' to the first island of land
        which has an altitude of 5m. So we expect [1, 4, -3]."""

        plugin = NeighbourSelection(
            land_constraint=True, search_radius=2E5,
            site_coordinate_system=self.region_projection.as_cartopy_crs(),
            site_x_coordinate='projection_x_coordinate',
            site_y_coordinate='projection_y_coordinate')
        result = plugin.process(self.region_sites, self.region_orography,
                                self.region_land_mask)
        expected = [[[1, 4, -3]]]

        self.assertArrayEqual(result.data, expected)

    def test_region_land_minimum_dz(self):
        """Test how the neighbour index changes when a land constraint is
        imposed and a minimum height difference condition. Here we expect to go
        'west' to the second band of land that we encounter, which has an
        altitude closer to that of the site. So we expect [0, 4, 1]."""

        plugin = NeighbourSelection(
            land_constraint=True, minimum_dz=True, search_radius=2E5,
            site_coordinate_system=self.region_projection.as_cartopy_crs(),
            site_x_coordinate='projection_x_coordinate',
            site_y_coordinate='projection_y_coordinate')
        result = plugin.process(self.region_sites, self.region_orography,
                                self.region_land_mask)
        expected = [[[0, 4, 1]]]

        self.assertArrayEqual(result.data, expected)

    def test_global_tied_case_nearest(self):
        """Test which neighbour is returned in an artificial case in which two
        neighbouring grid points are identically close. First with no
        constraints using the iris method. We put a site exactly half way
        between the two islands at -60 degrees longitude ([1, 4] and [4, 4] are
        equal distances either side of the site). This consistently returns the
        western island. Note that nudging the value to -59.9 will return the
        island to the east as expected."""

        self.global_sites[0]['longitude'] = -60.
        plugin = NeighbourSelection()
        result = plugin.process(self.global_sites, self.global_orography,
                                self.global_land_mask)
        expected = [[[2, 4, 2]]]

        self.assertArrayEqual(result.data, expected)

    def test_global_tied_case_nearest_land(self):
        """Test which neighbour is returned in an artificial case in which two
        neighbouring grid points are identically close. Identical to the test
        above except for the land constraint is now applied, so the neigbour is
        found using the KDTree. Using the KDTree the neighbour to the east is
        returned everytime the test is run."""

        self.global_sites[0]['longitude'] = -60.0
        plugin = NeighbourSelection(land_constraint=True, search_radius=1E8)
        result = plugin.process(self.global_sites, self.global_orography,
                                self.global_land_mask)
        expected = [[[4, 4, 2]]]

        self.assertArrayEqual(result.data, expected)

    def test_global_tied_case_nearest_land_min_dz(self):
        """Test which neighbour is returned in an artificial case in which two
        neighbouring grid points are identically close. Identical to the test
        above except for now with both a land constraint and minimum dz
        constraint. The neighbouring islands have been set to have the
        same vertical displacement as each other from the spot site. The
        neigbour is found using the KDTree.  Using the KDTree the neighbour to
        the east is returned everytime the test is run."""

        self.global_sites[0]['longitude'] = -60.0
        self.global_sites[0]['altitude'] = 5.
        self.global_orography.data[4, 4] = 5.

        plugin = NeighbourSelection(land_constraint=True, search_radius=1E8,
                                    minimum_dz=True)
        result = plugin.process(self.global_sites, self.global_orography,
                                self.global_land_mask)
        expected = [[[4, 4, 0]]]

        self.assertArrayEqual(result.data, expected)


if __name__ == '__main__':
    unittest.main()
