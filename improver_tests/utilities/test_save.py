# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for saving functionality."""

import os
import unittest
from tempfile import mkdtemp

import iris
import numpy as np
import pytest
from iris.coords import CellMethod
from iris.tests import IrisTest
from netCDF4 import Dataset

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.load import load_cube
from improver.utilities.save import _order_cell_methods, save_netcdf


def set_up_test_cube():
    """ Set up a temperature cube with additional global attributes. """
    data = np.linspace(-45.0, 45.0, 9, dtype=np.float32).reshape((1, 3, 3)) + 273.15

    attributes = {
        "um_version": "10.4",
        "source": "Met Office Unified Model",
        "Conventions": "CF-1.5",
        "institution": "Met Office",
        "history": "",
    }

    cube = set_up_variable_cube(
        data, attributes=attributes, standard_grid_metadata="uk_ens"
    )

    return cube


class Test_save_netcdf(IrisTest):
    """ Test function to save iris cubes as NetCDF files. """

    def setUp(self):
        """ Set up cube to write, read and check """
        self.global_keys_ref = [
            "title",
            "um_version",
            "grid_id",
            "source",
            "mosg__grid_type",
            "mosg__model_configuration",
            "mosg__grid_domain",
            "mosg__grid_version",
            "Conventions",
            "institution",
            "history",
        ]
        self.directory = mkdtemp()
        self.filepath = os.path.join(self.directory, "temp.nc")
        self.cube = set_up_test_cube()
        self.cell_methods = (
            CellMethod(method="maximum", coords="time", intervals="1 hour"),
            CellMethod(method="mean", coords="realization"),
        )
        self.cube.cell_methods = self.cell_methods

    def tearDown(self):
        """ Remove temporary directories created for testing. """
        try:
            os.remove(self.filepath)
        except FileNotFoundError:
            pass
        os.rmdir(self.directory)

    def test_basic_cube(self):
        """ Test saves file in required location """
        self.assertFalse(os.path.exists(self.filepath))
        save_netcdf(self.cube, self.filepath)
        self.assertTrue(os.path.exists(self.filepath))

    def test_compression(self):
        """ Test data gets compressed with default complevel 1 when saved """
        save_netcdf(self.cube, self.filepath)

        data = Dataset(self.filepath, mode="r")
        filters = data.variables["air_temperature"].filters()

        self.assertTrue(filters["zlib"])
        self.assertEqual(filters["complevel"], 1)

    def test_compression_level(self):
        """ Test data gets compressed with complevel provided by compression_level
        when saved """
        save_netcdf(self.cube, self.filepath, compression_level=3)

        data = Dataset(self.filepath, mode="r")
        filters = data.variables["air_temperature"].filters()

        self.assertTrue(filters["zlib"])
        self.assertEqual(filters["complevel"], 3)

    def test_no_compression(self):
        """ Test data does not get compressed when saved with compression_level 0 """
        save_netcdf(self.cube, self.filepath, compression_level=0)

        data = Dataset(self.filepath, mode="r")
        filters = data.variables["air_temperature"].filters()

        self.assertFalse(filters["zlib"])

    def test_compression_level_invalid(self):
        """ Test ValueError raised when invalid compression_level """
        with self.assertRaises(ValueError):
            save_netcdf(self.cube, self.filepath, compression_level="one")

    def test_compression_level_out_of_range(self):
        """ Test ValueError raised when compression_level out of range """
        with self.assertRaises(ValueError):
            save_netcdf(self.cube, self.filepath, compression_level=10)

    def test_basic_cube_list(self):
        """
        Test functionality for saving iris.cube.CubeList

        Both cubes are saved into one single file which breaks the convention
        of one cube per file. Therefore can't use IMPROVER specific load
        utilities since they don't have the ability to handle multiple
        cubes in one file.

        """
        cube_list = [self.cube, self.cube]
        save_netcdf(cube_list, self.filepath)
        read_cubes = iris.load(self.filepath)
        self.assertIsInstance(read_cubes, iris.cube.CubeList)
        # Length of read_cubes now increased to 3 as Iris 2 saves metadata
        # as separate cube rather than as attributes on other other cubes in
        # the file (Iris 1.13)
        self.assertEqual(len(read_cubes), 2)

    def test_cube_data(self):
        """ Test valid cube can be read from saved file """
        save_netcdf(self.cube, self.filepath)
        cube = load_cube(self.filepath)
        self.assertTrue(isinstance(cube, iris.cube.Cube))
        self.assertArrayEqual(cube.data, self.cube.data)

    def test_cube_dimensions(self):
        """ Test cube dimension coordinates are preserved """
        save_netcdf(self.cube, self.filepath)
        cube = load_cube(self.filepath)
        coord_names = [coord.name() for coord in cube.coords(dim_coords=True)]
        reference_names = [coord.name() for coord in self.cube.coords(dim_coords=True)]
        self.assertCountEqual(coord_names, reference_names)

    def test_cell_method_reordering_in_saved_file(self):
        """ Test cell methods are in the correct order when written out and
        read back in."""
        self.cube.cell_methods = (self.cell_methods[1], self.cell_methods[0])
        save_netcdf(self.cube, self.filepath)
        cube = load_cube(self.filepath)
        self.assertEqual(cube.cell_methods, self.cell_methods)

    def test_cf_global_attributes(self):
        """ Test that a NetCDF file saved from one cube only contains the
        expected global attributes.

        NOTE Loading the file as an iris.cube.Cube does not distinguish global
        from local attributes, and therefore cannot test for the correct
        behaviour here.
        """
        save_netcdf(self.cube, self.filepath)
        global_keys = Dataset(self.filepath, mode="r").ncattrs()
        self.assertTrue(all(key in self.global_keys_ref for key in global_keys))

    def test_cf_data_attributes(self):
        """ Test that forbidden global metadata are saved as data variable
        attributes
        """
        self.cube.attributes["test_attribute"] = np.arange(12)
        save_netcdf(self.cube, self.filepath)
        # cast explicitly to dictionary, as pylint does not recognise
        # OrderedDict as subscriptable
        cf_data_dict = dict(Dataset(self.filepath, mode="r").variables)
        self.assertTrue("test_attribute" in cf_data_dict["air_temperature"].ncattrs())
        self.assertArrayEqual(
            cf_data_dict["air_temperature"].getncattr("test_attribute"), np.arange(12)
        )

    def test_cf_shared_attributes_list(self):
        """ Test that a NetCDF file saved from a list of cubes that share
        non-global attributes does not promote these attributes to global.
        """
        cube_list = [self.cube, self.cube]
        save_netcdf(cube_list, self.filepath)
        global_keys_in_file = Dataset(self.filepath, mode="r").ncattrs()
        self.assertEqual(len(global_keys_in_file), 9)
        self.assertTrue(all(key in self.global_keys_ref for key in global_keys_in_file))

    def test_error_unknown_units(self):
        """Test key error when trying to save a cube with no units"""
        no_units_cube = iris.cube.Cube(np.array([1], dtype=np.float32))
        msg = "has unknown units"
        with self.assertRaisesRegex(ValueError, msg):
            save_netcdf(no_units_cube, self.filepath)

    def test_add_least_significant_digit(self):
        """Test bitshaving adds correct metadata"""
        save_netcdf(self.cube, self.filepath, least_significant_digit=2)
        cube = load_cube(self.filepath)
        self.assertEqual(cube.attributes["least_significant_digit"], 2)

    def test_update_least_significant_digit(self):
        """Test bitshaving updates metadata correctly if already present"""
        self.cube.attributes["least_significant_digit"] = 0
        save_netcdf(self.cube, self.filepath, least_significant_digit=2)
        cube = load_cube(self.filepath)
        self.assertEqual(cube.attributes["least_significant_digit"], 2)

    def test_remove_least_significant_digit(self):
        """Test precision metadata are removed if bitshaving is not applied.
        This is appropriate because if data have been processed since the last read
        this attribute is no longer correct and should be removed."""
        self.cube.attributes["least_significant_digit"] = 0
        save_netcdf(self.cube, self.filepath)
        cube = load_cube(self.filepath)
        self.assertNotIn("least_significant_digit", cube.attributes)


@pytest.fixture(name="bitshaving_cube")
def bitshaving_cube_fixture():
    """ Sets up a cube with a recurring decimal for bitshaving testing """
    cube = set_up_test_cube()
    # 1/9 fractions are recurring decimal and binary fractions
    # good for checking number of digits remaining after bitshaving
    cube.data = (
        np.linspace(1.0 / 9.0, 1.0, 9, dtype=np.float32).reshape((1, 3, 3)) + 273
    )
    return cube


@pytest.mark.parametrize("lsd", (0, 2, 3))
@pytest.mark.parametrize("compress", (0, 2))
def test_least_significant_digit(bitshaving_cube, tmp_path, lsd, compress):
    """ Test the least significant digit for bitshaving output files"""
    filepath = tmp_path / "temp.nc"
    save_netcdf(
        bitshaving_cube,
        filepath,
        compression_level=compress,
        least_significant_digit=lsd,
    )

    # check that netcdf metadata has been set
    data = Dataset(filepath, mode="r")
    assert data.variables["air_temperature"].least_significant_digit == lsd

    file_cube = load_cube(str(filepath))
    abs_diff = np.abs(bitshaving_cube.data.data - file_cube.data.data)
    # check that whole numbers are preserved
    assert np.min(abs_diff) == 0.0
    # check that modified data is accurate to the specified number of digits
    assert 0 < np.median(abs_diff) < 10 ** (-1.0 * (lsd + 0.5))
    assert 0 < np.mean(abs_diff) < 10 ** (-1.0 * (lsd + 0.5))
    assert np.max(abs_diff) < 10 ** (-1.0 * lsd)


class Test__order_cell_methods(IrisTest):
    """ Test function that sorts cube cell_methods before saving. """

    def setUp(self):
        """ Set up cube with cell_methods."""
        self.cube = set_up_test_cube()
        self.cell_methods = (
            CellMethod(method="maximum", coords="time", intervals="1 hour"),
            CellMethod(method="mean", coords="realization"),
        )
        self.cube.cell_methods = self.cell_methods

    def test_no_reordering_cube(self):
        """ Test the order is preserved is no reordering required."""
        _order_cell_methods(self.cube)
        self.assertEqual(self.cube.cell_methods, self.cell_methods)

    def test_reordering_cube(self):
        """ Test the order is changed when reordering is required."""
        self.cube.cell_methods = (self.cell_methods[1], self.cell_methods[0])
        # Test that following the manual reorder above the cube cell methods
        # and the tuple don't match.
        self.assertNotEqual(self.cube.cell_methods, self.cell_methods)

        _order_cell_methods(self.cube)
        # Test that they do match once sorting has occured.
        self.assertEqual(self.cube.cell_methods, self.cell_methods)


if __name__ == "__main__":
    unittest.main()
