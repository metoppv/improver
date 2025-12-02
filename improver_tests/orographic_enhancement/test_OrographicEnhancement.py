# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the OrographicEnhancement plugin."""

import unittest
from datetime import datetime

import iris
import numpy as np
from iris.coord_systems import GeogCS, TransverseMercator
from iris.coords import DimCoord

from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.metadata.constants.mo_attributes import MOSG_GRID_ATTRIBUTES
from improver.orographic_enhancement import OrographicEnhancement
from improver.synthetic_data.set_up_test_cubes import construct_scalar_time_coords
from improver.utilities.cube_manipulation import sort_coord_in_cube

# UKPP projection
TMercCS = TransverseMercator(
    latitude_of_projection_origin=49.0,
    longitude_of_central_meridian=-2.0,
    false_easting=400000.0,
    false_northing=-100000.0,
    scale_factor_at_central_meridian=0.9996013045310974,
    ellipsoid=GeogCS(semi_major_axis=6377563.396, semi_minor_axis=6356256.91),
)


def set_up_variable_cube(
    data, name="temperature", units="degC", xo=400000.0, yo=0.0, attributes=None
):
    """
    Set up cube containing diagnostic variable data for regridding tests.
    Data are on a 2 km Transverse Mercator grid with an inverted y-axis,
    located in the UK.
    """
    y_points = 2000.0 * (data.shape[0] - np.arange(data.shape[0])) + yo
    x_points = 2000.0 * np.arange(data.shape[1]) + xo

    y_coord = DimCoord(
        y_points, "projection_y_coordinate", units="m", coord_system=TMercCS
    )
    x_coord = DimCoord(
        x_points, "projection_x_coordinate", units="m", coord_system=TMercCS
    )

    time_coords = construct_scalar_time_coords(
        datetime(2015, 11, 23, 4, 30), None, datetime(2015, 11, 22, 22, 30)
    )
    cube = iris.cube.Cube(
        data,
        long_name=name,
        units=units,
        attributes=attributes,
        dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)],
        aux_coords_and_dims=time_coords,
    )
    return cube


def set_up_invalid_variable_cube(valid_cube):
    """
    Generate a new cube with an extra dimension from a 2D variable cube, to
    create an invalid cube for testing the process method.
    """
    realization_coord = DimCoord(np.array([0], dtype=np.int32), "realization")
    cube1 = valid_cube.copy()
    cube1.add_aux_coord(realization_coord)
    cube2 = cube1.copy()
    cube2.coord("realization").points = [1]
    return iris.cube.CubeList([cube1, cube2]).merge_cube()


def set_up_orography_cube(data, xo=400000.0, yo=0.0):
    """
    Set up cube containing high resolution UK orography data.
    """
    y_points = 1000.0 * (data.shape[0] - np.arange(data.shape[0])) + yo
    x_points = 1000.0 * np.arange(data.shape[1]) + xo

    y_coord = DimCoord(
        y_points, "projection_y_coordinate", units="m", coord_system=TMercCS
    )
    x_coord = DimCoord(
        x_points, "projection_x_coordinate", units="m", coord_system=TMercCS
    )

    cube = iris.cube.Cube(
        data,
        long_name="topography",
        units="m",
        dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)],
    )
    cube.attributes["mosg__grid_type"] = "standard"
    cube.attributes["mosg__grid_version"] = "1.0.0"
    cube.attributes["mosg__grid_domain"] = "uk"

    return cube


class Test__init__(unittest.TestCase):
    """Test the __init__ method"""

    def test_basic(self):
        """Test initialisation with no arguments"""
        plugin = OrographicEnhancement()
        self.assertAlmostEqual(plugin.orog_thresh_m, 20.0)
        self.assertAlmostEqual(plugin.rh_thresh_ratio, 0.8)
        self.assertAlmostEqual(plugin.vgradz_thresh_ms, 0.0005)
        self.assertAlmostEqual(plugin.upstream_range_of_influence_km, 15.0)
        self.assertAlmostEqual(plugin.efficiency_factor, 0.23265)
        self.assertAlmostEqual(plugin.cloud_lifetime_s, 102.0)

        none_type_attributes = [
            "topography",
            "temperature",
            "humidity",
            "pressure",
            "uwind",
            "vwind",
            "svp",
            "vgradz",
            "grid_spacing_km",
        ]
        for attr in none_type_attributes:
            self.assertIsNone(getattr(plugin, attr))


class Test__repr__(unittest.TestCase):
    """Test the __repr__ method"""

    def test_basic(self):
        """Test string representation of plugin"""
        plugin = OrographicEnhancement()
        self.assertEqual(str(plugin), "<OrographicEnhancement()>")


class Test__orography_gradients(unittest.TestCase):
    """Test the _orography_gradients method"""

    def setUp(self):
        """Set up an input cube"""
        self.plugin = OrographicEnhancement()
        data = np.array(
            [[200.0, 450.0, 850.0], [320.0, 500.0, 1000.0], [230.0, 600.0, 900.0]]
        )
        x_coord = DimCoord(np.arange(3), "projection_x_coordinate", units="km")
        y_coord = DimCoord(np.arange(3), "projection_y_coordinate", units="km")
        self.plugin.topography = iris.cube.Cube(
            data,
            long_name="topography",
            units="m",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)],
        )

    def test_basic(self):
        """Test outputs are cubes"""
        gradx, grady = self.plugin._orography_gradients()
        self.assertIsInstance(gradx, iris.cube.Cube)
        self.assertIsInstance(grady, iris.cube.Cube)

    def test_values(self):
        """Test output values and units"""
        expected_gradx = np.array(
            [
                [0.12333333, 0.33, 0.53666667],
                [0.2, 0.33333333, 0.46666667],
                [0.27666667, 0.33666667, 0.39666667],
            ]
        )

        expected_grady = np.array(
            [
                [0.15833333, 0.175, 0.19166667],
                [0.035, 0.03833333, 0.04166667],
                [-0.08833333, -0.09833333, -0.10833333],
            ]
        )
        gradx, grady = self.plugin._orography_gradients()
        np.testing.assert_array_almost_equal(gradx.data, expected_gradx)
        np.testing.assert_array_almost_equal(grady.data, expected_grady)
        for cube in [gradx, grady]:
            self.assertEqual(cube.units, "1")


class Test__regrid_variable(unittest.TestCase):
    """Test the _regrid_variable method"""

    def setUp(self):
        """Set up input cubes"""
        temperature = np.arange(6).reshape(2, 3)
        self.temperature_cube = set_up_variable_cube(temperature)
        orography = np.array(
            [
                [20.0, 30.0, 40.0, 30.0, 25.0, 25.0],
                [30.0, 50.0, 80.0, 60.0, 50.0, 45.0],
                [50.0, 65.0, 90.0, 70.0, 60.0, 50.0],
                [45.0, 60.0, 85.0, 65.0, 55.0, 45.0],
            ]
        )
        orography_cube = set_up_orography_cube(orography)
        self.plugin = OrographicEnhancement()
        self.plugin.topography = sort_coord_in_cube(
            orography_cube, orography_cube.coord(axis="y")
        )

    def test_basic(self):
        """Test cube of the correct shape and type is returned"""
        expected_data = np.array(
            [
                [4.5, 5.0, 5.5, 6.0, 6.5, 7.0],
                [3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
                [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            ]
        )
        result = self.plugin._regrid_variable(self.temperature_cube, "degC")
        self.assertIsInstance(result, iris.cube.Cube)
        np.testing.assert_array_almost_equal(result.data, expected_data)
        self.assertEqual(result.data.dtype, "float32")

    def test_axis_inversion(self):
        """Test axes are output in ascending order"""
        result = self.plugin._regrid_variable(self.temperature_cube, "degC")
        x_points = result.coord(axis="x").points
        y_points = result.coord(axis="y").points
        self.assertTrue(x_points[1] > x_points[0])
        self.assertTrue(y_points[1] > y_points[0])

    def test_unit_conversion(self):
        """Test units are correctly converted"""
        expected_data = np.array(
            [
                [277.65, 278.15, 278.65, 279.15, 279.65, 280.15],
                [276.15, 276.65, 277.15, 277.65, 278.15, 278.65],
                [274.65, 275.15, 275.65, 276.15, 276.65, 277.15],
                [273.15, 273.65, 274.15, 274.65, 275.15, 275.65],
            ],
            dtype=np.float32,
        )
        result = self.plugin._regrid_variable(self.temperature_cube, "kelvin")
        self.assertEqual(result.units, "kelvin")
        np.testing.assert_array_almost_equal(result.data, expected_data)

    def test_null(self):
        """Test cube is unchanged if axes and grid are already correct"""
        correct_cube = self.plugin.topography.copy()
        result = self.plugin._regrid_variable(correct_cube, "m")
        np.testing.assert_array_almost_equal(result.data, correct_cube.data)
        self.assertEqual(result.metadata, correct_cube.metadata)

    def test_input_unchanged(self):
        """Test the input cube is not modified in place"""
        reference_cube = self.temperature_cube.copy()
        _ = self.plugin._regrid_variable(self.temperature_cube, "degC")
        np.testing.assert_array_almost_equal(
            self.temperature_cube.data, reference_cube.data
        )
        self.assertEqual(self.temperature_cube.metadata, reference_cube.metadata)


class DataCubeTest(unittest.TestCase):
    """Shared setUp function for tests requiring full input data cubes
    with an inverted y-axis"""

    def setUp(self):
        """Set up input cubes"""
        temperature = np.arange(6).reshape(2, 3)
        self.temperature = set_up_variable_cube(temperature)
        humidity = np.arange(0.75, 0.86, 0.02).reshape(2, 3)
        self.humidity = set_up_variable_cube(humidity, "relhumidity", "1")
        pressure = np.arange(820, 921, 20).reshape(2, 3)
        self.pressure = set_up_variable_cube(pressure, "pressure", "hPa")
        uwind = np.full((2, 3), 20.0, dtype=np.float32)
        self.uwind = set_up_variable_cube(uwind, "wind-u", "knots")
        vwind = np.full((2, 3), 12.0, dtype=np.float32)
        self.vwind = set_up_variable_cube(vwind, "wind-v", "knots")

        orography = np.array(
            [
                [20.0, 30.0, 40.0, 30.0, 25.0, 25.0],
                [30.0, 50.0, 80.0, 60.0, 50.0, 45.0],
                [50.0, 65.0, 90.0, 70.0, 60.0, 50.0],
                [45.0, 60.0, 85.0, 65.0, 55.0, 45.0],
            ]
        )
        self.orography_cube = set_up_orography_cube(orography)
        self.plugin = OrographicEnhancement()


class Test__regrid_and_populate(DataCubeTest):
    """Test the _regrid_and_populate method"""

    def test_basic(self):
        """Test function populates class instance"""
        self.plugin._regrid_and_populate(
            self.temperature,
            self.humidity,
            self.pressure,
            self.uwind,
            self.vwind,
            self.orography_cube,
        )
        plugin_cubes = [
            self.plugin.temperature,
            self.plugin.humidity,
            self.plugin.pressure,
            self.plugin.uwind,
            self.plugin.vwind,
            self.plugin.topography,
        ]
        for cube in plugin_cubes:
            self.assertIsInstance(cube, iris.cube.Cube)
        self.assertIsInstance(self.plugin.vgradz, np.ndarray)

    def test_variables(self):
        """Test variable values are sensible"""
        expected_temperature = np.array(
            [
                [277.65, 278.15, 278.65, 279.15, 279.65, 280.15],
                [276.15, 276.65, 277.15, 277.65, 278.15, 278.65],
                [274.65, 275.15, 275.65, 276.15, 276.65, 277.15],
                [273.15, 273.65, 274.15, 274.65, 275.15, 275.65],
            ],
            dtype=np.float32,
        )

        expected_humidity = np.array(
            [
                [0.84, 0.85, 0.86, 0.87, 0.88, 0.89],
                [0.81, 0.82, 0.83, 0.84, 0.85, 0.86],
                [0.78, 0.79, 0.80, 0.81, 0.82, 0.83],
                [0.75, 0.76, 0.77, 0.78, 0.79, 0.80],
            ],
            dtype=np.float32,
        )

        expected_pressure = np.array(
            [
                [91000.0, 92000.0, 93000.0, 94000.0, 95000.0, 96000.0],
                [88000.0, 89000.0, 90000.0, 91000.0, 92000.0, 93000.0],
                [85000.0, 86000.0, 87000.0, 88000.0, 89000.0, 90000.0],
                [82000.0, 83000.0, 84000.0, 85000.0, 86000.0, 87000.0],
            ],
            dtype=np.float32,
        )

        expected_uwind = np.full((4, 6), 10.288889, dtype=np.float32)
        expected_vwind = np.full((4, 6), 6.1733336, dtype=np.float32)

        self.plugin._regrid_and_populate(
            self.temperature,
            self.humidity,
            self.pressure,
            self.uwind,
            self.vwind,
            self.orography_cube,
        )

        plugin_cubes = [
            self.plugin.temperature,
            self.plugin.humidity,
            self.plugin.pressure,
            self.plugin.uwind,
            self.plugin.vwind,
        ]
        expected_data = [
            expected_temperature,
            expected_humidity,
            expected_pressure,
            expected_uwind,
            expected_vwind,
        ]

        for cube, array in zip(plugin_cubes, expected_data):
            np.testing.assert_array_almost_equal(cube.data, array)
        np.testing.assert_array_almost_equal(
            self.plugin.topography.data, np.flipud(self.orography_cube.data)
        )

    def test_vgradz(self):
        """Test values of vgradz are sensible"""
        expected_vgradz = np.array(
            [
                [
                    0.20577779,
                    0.29837778,
                    0.10803331,
                    -0.07716664,
                    -0.03086665,
                    -0.03601114,
                ],
                [
                    0.07888144,
                    0.19205923,
                    0.01371852,
                    -0.16976666,
                    -0.10460369,
                    -0.08231109,
                ],
                [
                    0.02229258,
                    0.07030742,
                    -0.10288889,
                    -0.25722224,
                    -0.17148148,
                    -0.12175184,
                ],
                [
                    0.05315927,
                    -0.01543331,
                    -0.22464074,
                    -0.36525553,
                    -0.24864818,
                    -0.17148149,
                ],
            ]
        )
        self.plugin._regrid_and_populate(
            self.temperature,
            self.humidity,
            self.pressure,
            self.uwind,
            self.vwind,
            self.orography_cube,
        )
        np.testing.assert_array_almost_equal(self.plugin.vgradz, expected_vgradz)


class Test__generate_mask(unittest.TestCase):
    """Test the _generate_mask method"""

    def setUp(self):
        """Set up and populate a plugin instance"""
        self.plugin = OrographicEnhancement()
        x_coord = DimCoord(np.arange(5), "projection_x_coordinate", units="km")
        y_coord = DimCoord(np.arange(5), "projection_y_coordinate", units="km")

        # this is neighbourhood-processed as part of mask generation
        topography_data = np.array(
            [
                [0.0, 10.0, 20.0, 50.0, 100.0],
                [10.0, 20.0, 50.0, 100.0, 200.0],
                [25.0, 60.0, 80.0, 160.0, 220.0],
                [50.0, 80.0, 100.0, 200.0, 250.0],
                [50.0, 80.0, 100.0, 200.0, 250.0],
            ]
        )
        self.plugin.topography = iris.cube.Cube(
            topography_data,
            long_name="topography",
            units="m",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)],
        )

        humidity_data = np.full((5, 5), 0.9)
        humidity_data[1, 3] = 0.5
        self.plugin.humidity = iris.cube.Cube(
            humidity_data,
            long_name="relhumidity",
            units="1",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)],
        )

        self.plugin.vgradz = np.full((5, 5), 0.01)
        self.plugin.vgradz[3:, :] = 0.0

    def test_basic(self):
        """Test output is array"""
        result = self.plugin._generate_mask()
        self.assertIsInstance(result, np.ndarray)

    def test_values(self):
        """Test output mask is correct"""
        expected_output = np.full((5, 5), False, dtype=bool)
        expected_output[0, :2] = True  # orography too low
        expected_output[1, 3] = True  # humidity too low
        expected_output[3:, :] = True  # vgradz too low
        result = self.plugin._generate_mask()
        np.testing.assert_array_equal(result, expected_output)


class Test__point_orogenh(unittest.TestCase):
    """Test the _point_orogenh method"""

    def setUp(self):
        """Set up and populate a plugin instance"""
        x_coord = DimCoord(np.arange(3), "projection_x_coordinate", units="km")
        y_coord = DimCoord(np.arange(3), "projection_y_coordinate", units="km")

        temperature = np.array(
            [[277.1, 278.2, 277.7], [278.6, 278.4, 278.9], [278.9, 279.0, 279.6]]
        )
        humidity = np.array(
            [[0.74, 0.85, 0.94], [0.81, 0.82, 0.91], [0.86, 0.93, 0.97]]
        )

        self.plugin = OrographicEnhancement()
        self.plugin.temperature = iris.cube.Cube(
            temperature,
            long_name="temperature",
            units="kelvin",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)],
        )
        self.plugin.humidity = iris.cube.Cube(
            humidity,
            long_name="relhumidity",
            units="1",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)],
        )
        self.plugin.svp = np.array(
            [[813.6, 878.0, 848.3], [903.2, 890.5, 922.2], [922.1, 928.4, 967.9]]
        )

        self.plugin.vgradz = np.array(
            [[0.02, 0.08, 0.2], [-0.06, 0.12, 0.22], [0.08, 0.16, 0.23]]
        )

        topography_data = np.full((3, 3), 50.0, dtype=np.float32)
        self.plugin.topography = iris.cube.Cube(
            topography_data,
            long_name="topography",
            units="m",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)],
        )

    def test_basic(self):
        """Test output is an array"""
        result = self.plugin._point_orogenh()
        self.assertIsInstance(result, np.ndarray)

    def test_values(self):
        """Test output values are as expected"""
        expected_values = np.array(
            [
                [0.0, 1.67372072, 4.47886658],
                [0.0, 2.45468903, 5.1627059],
                [1.77400422, 3.86162901, 6.02323198],
            ]
        )
        result = self.plugin._point_orogenh()
        np.testing.assert_array_almost_equal(result, expected_values)


class Test__get_point_distances(unittest.TestCase):
    """Test the _get_point_distances function"""

    def setUp(self):
        """Define input matrices and plugin"""
        self.wind_speed = np.ones((3, 4), dtype=np.float32)
        sin_wind_dir = np.linspace(0, 1, 12).reshape(3, 4)
        cos_wind_dir = np.sqrt(1.0 - np.square(sin_wind_dir))
        self.max_sin_cos = np.where(
            abs(sin_wind_dir) > abs(cos_wind_dir), abs(sin_wind_dir), abs(cos_wind_dir)
        )
        self.plugin = OrographicEnhancement()
        self.plugin.grid_spacing_km = 3.0

    def test_basic(self):
        """Test the function returns an array of the expected shape"""
        distance = self.plugin._get_point_distances(self.wind_speed, self.max_sin_cos)
        self.assertIsInstance(distance, np.ndarray)
        self.assertSequenceEqual(distance.shape, (5, 3, 4))

    def test_values_with_nans(self):
        """Test for expected values including nans"""
        slice_0 = np.zeros((3, 4), dtype=np.float32)
        slice_1 = np.array(
            [
                [1.0, 1.00415802, 1.01695037, 1.03940225],
                [1.07349002, 1.12268281, 1.1931175, 1.2963624],
                [1.375, 1.22222221, 1.10000002, 1.0],
            ]
        )
        slice_2 = 2.0 * slice_1
        slice_3 = 3.0 * slice_1
        slice_3[1, 3] = np.nan
        slice_3[2, 0] = np.nan
        slice_4 = np.full_like(slice_0, np.nan)
        slice_4[0, 0] = 4.0
        slice_4[-1, -1] = 4.0
        expected_data = np.array([slice_0, slice_1, slice_2, slice_3, slice_4])

        distance = self.plugin._get_point_distances(self.wind_speed, self.max_sin_cos)
        np.testing.assert_allclose(distance, expected_data, equal_nan=True)


class Test__locate_source_points(unittest.TestCase):
    """Test the _locate_source_points method"""

    def setUp(self):
        """Define input matrices and plugin"""
        self.wind_speed = np.ones((3, 4), dtype=np.float32)
        self.sin_wind_dir = np.full((3, 4), 0.4, dtype=np.float32)
        self.cos_wind_dir = np.full((3, 4), np.sqrt(0.84), dtype=np.float32)
        self.plugin = OrographicEnhancement()
        self.plugin.grid_spacing_km = 3.0

    def test_basic(self):
        """Test location of source points"""
        distance = self.plugin._get_point_distances(self.wind_speed, self.cos_wind_dir)
        xsrc, ysrc = self.plugin._locate_source_points(
            self.wind_speed, distance, self.sin_wind_dir, self.cos_wind_dir
        )

        expected_xsrc = np.array(
            [
                [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
                [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
                [[0, 0, 1, 2], [0, 0, 1, 2], [0, 0, 1, 2]],
                [[0, 0, 1, 2], [0, 0, 1, 2], [0, 0, 1, 2]],
            ]
        )

        expected_ysrc = np.array(
            [
                [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ]
        )

        np.testing.assert_array_equal(xsrc, expected_xsrc)
        np.testing.assert_array_equal(ysrc, expected_ysrc)


class Test__compute_weighted_values(unittest.TestCase):
    """Test the _compute_weighted_values method"""

    def setUp(self):
        """Set up plugin and some inputs"""
        self.plugin = OrographicEnhancement()
        self.plugin.grid_spacing_km = 3.0

        self.point_orogenh = np.array(
            [
                [4.1, 4.6, 5.6, 6.8, 5.5],
                [4.4, 4.6, 5.8, 6.2, 5.5],
                [5.2, 3.0, 3.4, 5.1, 3.3],
                [0.6, 2.0, 1.8, 4.2, 2.5],
                [0.0, 0.0, 0.2, 3.2, 1.8],
            ]
        )

        self.wind_speed = np.full((5, 5), 25.0, dtype=np.float32)
        sin_wind_dir = np.full((5, 5), 0.4, dtype=np.float32)
        cos_wind_dir = np.full((5, 5), np.sqrt(0.84), dtype=np.float32)
        self.distance = self.plugin._get_point_distances(self.wind_speed, cos_wind_dir)
        self.xsrc, self.ysrc = self.plugin._locate_source_points(
            self.wind_speed, self.distance, sin_wind_dir, cos_wind_dir
        )

    def test_basic(self):
        """Test output is two arrays"""
        orogenh, weights = self.plugin._compute_weighted_values(
            self.point_orogenh, self.xsrc, self.ysrc, self.distance, self.wind_speed
        )
        self.assertIsInstance(orogenh, np.ndarray)
        self.assertIsInstance(weights, np.ndarray)

    def test_values(self):
        """Test values are as expected"""
        expected_orogenh = np.array(
            [
                [6.0531969, 6.7725644, 8.2301264, 9.9942646, 8.1690931],
                [6.3531971, 6.7725644, 8.4301271, 9.3942642, 8.1690931],
                [7.2848172, 5.1725645, 6.1178742, 8.0310230, 5.9690924],
                [3.0469213, 3.4817038, 3.4649093, 6.6558237, 4.1816435],
                [0.4585612, 1.0727906, 1.1036499, 5.1721582, 3.0895371],
            ]
        )
        expected_weights = np.full((5, 5), 1.4763895, dtype=np.float32)
        orogenh, weights = self.plugin._compute_weighted_values(
            self.point_orogenh, self.xsrc, self.ysrc, self.distance, self.wind_speed
        )
        np.testing.assert_array_almost_equal(orogenh, expected_orogenh)
        np.testing.assert_array_almost_equal(weights, expected_weights)


class Test__add_upstream_component(unittest.TestCase):
    """Test the _add_upstream_component method"""

    def setUp(self):
        """Set up a plugin with wind components"""
        x_coord = DimCoord(3.0 * np.arange(5), "projection_x_coordinate", units="km")
        y_coord = DimCoord(3.0 * np.arange(5), "projection_y_coordinate", units="km")
        uwind = np.full((5, 5), 20.0, dtype=np.float32)
        vwind = np.full((5, 5), 12.0, dtype=np.float32)

        self.plugin = OrographicEnhancement()
        self.plugin.uwind = iris.cube.Cube(
            uwind,
            long_name="grid_eastward_wind",
            units="m s-1",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)],
        )
        self.plugin.vwind = iris.cube.Cube(
            vwind,
            long_name="grid_northward_wind",
            units="m s-1",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)],
        )
        self.plugin.grid_spacing_km = 3.0

        self.point_orogenh = np.array(
            [
                [4.1, 4.6, 5.6, 6.8, 5.5],
                [4.4, 4.6, 5.8, 6.2, 5.5],
                [5.2, 3.0, 3.4, 5.1, 3.3],
                [0.6, 2.0, 1.8, 4.2, 2.5],
                [0.0, 0.0, 0.2, 3.2, 1.8],
            ]
        )

    def test_basic(self):
        """Test output is an array"""
        result = self.plugin._add_upstream_component(self.point_orogenh)
        self.assertIsInstance(result, np.ndarray)

    def test_values(self):
        """Test output values are sensible"""
        expected_values = np.array(
            [
                [0.953865, 1.039876, 1.241070, 1.506976, 1.355637],
                [1.005472, 1.039876, 1.275474, 1.403762, 1.355637],
                [1.161275, 0.782825, 0.863303, 1.226206, 0.942638],
                [0.418468, 0.659300, 0.496544, 0.927728, 0.735382],
                [0.036423, 0.036423, 0.152506, 0.660092, 0.558801],
            ]
        )

        result = self.plugin._add_upstream_component(self.point_orogenh)
        np.testing.assert_array_almost_equal(result, expected_values)


class Test__create_output_cube(unittest.TestCase):
    """Test the _create_output_cube method"""

    def setUp(self):
        """Set up a plugin instance, data array and cubes"""
        self.plugin = OrographicEnhancement()
        topography = set_up_orography_cube(np.zeros((3, 4), dtype=np.float32))
        self.plugin.topography = sort_coord_in_cube(
            topography, topography.coord(axis="y")
        )

        t_attributes = {
            "institution": "Met Office",
            "source": "Met Office Unified Model",
            "mosg__grid_type": "standard",
            "mosg__grid_version": "1.2.0",
            "mosg__grid_domain": "uk_extended",
            "mosg__model_configuration": "uk_det",
        }
        self.temperature = set_up_variable_cube(
            np.full((2, 4), 280.15, dtype=np.float32),
            units="kelvin",
            xo=398000.0,
            attributes=t_attributes,
        )

        self.orogenh = np.array(
            [[1.1, 1.2, 1.5, 1.4], [1.0, 1.3, 1.4, 1.6], [0.8, 0.9, 1.2, 0.9]]
        )

    def test_basic(self):
        """Test that the cube is returned with float32 coords"""
        output = self.plugin._create_output_cube(self.orogenh, self.temperature)

        self.assertIsInstance(output, iris.cube.Cube)
        for coord in output.coords(dim_coords=True):
            self.assertEqual(coord.points.dtype, "float32")

    def test_values(self):
        """Test the cube is changed only in units (to m s-1)"""
        original_converted = 2.7777778e-07 * self.orogenh

        output = self.plugin._create_output_cube(self.orogenh, self.temperature)
        np.testing.assert_array_almost_equal(output.data, original_converted)

    def test_metadata(self):
        """Check output metadata on cube is as expected"""
        expected_attributes = {
            "title": MANDATORY_ATTRIBUTE_DEFAULTS["title"],
            "source": self.temperature.attributes["source"],
            "institution": self.temperature.attributes["institution"],
        }
        for attr in MOSG_GRID_ATTRIBUTES:
            expected_attributes[attr] = self.plugin.topography.attributes[attr]

        output = self.plugin._create_output_cube(self.orogenh, self.temperature)
        for axis in ["x", "y"]:
            self.assertEqual(
                output.coord(axis=axis), self.plugin.topography.coord(axis=axis)
            )

        self.assertEqual(output.name(), "orographic_enhancement")
        self.assertEqual(output.units, "m s-1")
        for t_coord in ["time", "forecast_period", "forecast_reference_time"]:
            self.assertEqual(output.coord(t_coord), self.temperature.coord(t_coord))

        assert set(output.attributes.keys()) == set(expected_attributes.keys())
        for key in output.attributes.keys():
            try:
                assert expected_attributes[key] == output.attributes[key]
            except ValueError:
                np.testing.assert_array_equal(
                    expected_attributes[key], output.attributes[key]
                )


class Test_process(DataCubeTest):
    """Test the process method"""

    def test_basic(self):
        """Test output is float32 cube with float32 coordinates"""
        orogenh = self.plugin.process(
            self.temperature,
            self.humidity,
            self.pressure,
            self.uwind,
            self.vwind,
            self.orography_cube,
        )
        self.assertIsInstance(orogenh, iris.cube.Cube)
        self.assertEqual(orogenh.data.dtype, "float32")
        for coord in orogenh.coords(dim_coords=True):
            self.assertEqual(coord.points.dtype, "float32")

    def test_unmatched_coords(self):
        """Test error thrown if input variable cubes do not match"""
        self.temperature.coord("forecast_reference_time").points = (
            self.temperature.coord("forecast_reference_time").points - 3600
        )
        self.temperature.coord("forecast_period").points = (
            self.temperature.coord("forecast_period").points - 3600
        )
        msg = "Input cube coordinates"
        with self.assertRaisesRegex(ValueError, msg):
            _ = self.plugin.process(
                self.temperature,
                self.humidity,
                self.pressure,
                self.uwind,
                self.vwind,
                self.orography_cube,
            )

    def test_extra_dimensions(self):
        """Test error thrown if input variable cubes have an extra dimension"""
        temperature = set_up_invalid_variable_cube(self.temperature)
        humidity = set_up_invalid_variable_cube(self.humidity)
        pressure = set_up_invalid_variable_cube(self.pressure)
        uwind = set_up_invalid_variable_cube(self.uwind)
        vwind = set_up_invalid_variable_cube(self.vwind)
        msg = "Require 2D fields as input; found 3 dimensions"

        with self.assertRaisesRegex(ValueError, msg):
            _ = self.plugin.process(
                temperature, humidity, pressure, uwind, vwind, self.orography_cube
            )

    def test_inputs_unmodified(self):
        """Test the process method does not modify any of the input cubes"""
        cube_list = [
            self.temperature,
            self.humidity,
            self.pressure,
            self.uwind,
            self.vwind,
            self.orography_cube,
        ]
        copied_cubes = []
        for cube in cube_list:
            copied_cubes.append(cube.copy())

        _ = self.plugin.process(
            self.temperature,
            self.humidity,
            self.pressure,
            self.uwind,
            self.vwind,
            self.orography_cube,
        )

        for cube, copy in zip(cube_list, copied_cubes):
            np.testing.assert_array_almost_equal(cube.data, copy.data)
            self.assertEqual(cube.metadata, copy.metadata)

    def test_values(self):
        """Test values of output"""
        expected_data = np.array(
            [
                [
                    2.6524199e-07,
                    3.4075157e-07,
                    2.5099993e-07,
                    9.1911055e-08,
                    1.7481890e-08,
                    1.5676112e-09,
                ],
                [
                    1.6797775e-07,
                    2.4365076e-07,
                    1.7639361e-07,
                    9.1911055e-08,
                    1.7481890e-08,
                    1.5676112e-09,
                ],
                [
                    4.1531862e-08,
                    4.1531862e-08,
                    8.9591637e-08,
                    2.8731334e-08,
                    5.3441389e-09,
                    1.5676112e-09,
                ],
                [
                    8.5711110e-10,
                    8.5711110e-10,
                    8.5711110e-10,
                    8.5711110e-10,
                    2.1291666e-09,
                    2.4547223e-10,
                ],
            ],
            dtype=np.float32,
        )

        orogenh = self.plugin.process(
            self.temperature,
            self.humidity,
            self.pressure,
            self.uwind,
            self.vwind,
            self.orography_cube,
        )

        np.testing.assert_array_almost_equal(orogenh.data, expected_data)
        self.assertAlmostEqual(self.plugin.grid_spacing_km, 1.0)


if __name__ == "__main__":
    unittest.main()
