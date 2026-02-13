# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the ApplyGriddedLapseRate plugin."""

import unittest

import iris
import numpy as np
import pytest

from improver.constants import DALR
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)
from improver.temperature.lapse_rate import ApplyGriddedLapseRate


class Test_process(unittest.TestCase):
    """Test the ApplyGriddedLapseRate plugin"""

    def setUp(self):
        """Set up some input cubes"""
        source_orog = np.array(
            [
                [400.0, 400.0, 402.0, 402.0],
                [400.0, 400.0, 402.0, 402.0],
                [403.0, 403.0, 405.0, 405.0],
                [403.0, 403.0, 405.0, 405.0],
            ],
            dtype=np.float32,
        )
        self.source_orog = set_up_variable_cube(
            source_orog, name="orography", units="m", spatial_grid="equalarea"
        )

        dest_orog = np.array(
            [
                [400.0, 401.0, 401.0, 402.0],
                [402.0, 402.0, 402.0, 403.0],
                [403.0, 404.0, 405.0, 404.0],
                [404.0, 405.0, 406.0, 405.0],
            ],
            dtype=np.float32,
        )
        self.dest_orog = set_up_variable_cube(
            dest_orog, name="orography", units="m", spatial_grid="equalarea"
        )

        self.lapse_rate = set_up_variable_cube(
            np.full((4, 4), DALR, dtype=np.float32),
            name="lapse_rate",
            units="K m-1",
            spatial_grid="equalarea",
        )

        # specify temperature values ascending in 0.25 K increments
        temp_data = np.array(
            [
                [276.0, 276.25, 276.5, 276.75],
                [277.0, 277.25, 277.5, 277.75],
                [278.0, 278.25, 278.5, 278.75],
                [279.0, 279.25, 279.5, 279.75],
            ],
            dtype=np.float32,
        )
        self.temperature = set_up_variable_cube(
            temp_data, name="screen_temperature", spatial_grid="equalarea"
        )

        self.expected_data = np.array(
            [
                [276.0, 276.2402, 276.5098, 276.75],
                [276.9804, 277.2304, 277.5, 277.7402],
                [278.0, 278.2402, 278.5, 278.7598],
                [278.9902, 279.2304, 279.4902, 279.75],
            ],
            dtype=np.float32,
        )

        self.plugin = ApplyGriddedLapseRate()

    def test_basic(self):
        """Test output is cube with correct name, type and units"""
        result = self.plugin(
            self.temperature, self.lapse_rate, self.source_orog, self.dest_orog
        )
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "screen_temperature")
        self.assertEqual(result.units, "K")
        self.assertEqual(result.dtype, np.float32)

    def test_values(self):
        """Check adjusted temperature values are as expected"""
        result = self.plugin(
            self.temperature, self.lapse_rate, self.source_orog, self.dest_orog
        )

        # test that temperatures are reduced where destination orography
        # is higher than source
        source_lt_dest = np.where(self.source_orog.data < self.dest_orog.data)
        self.assertTrue(
            np.all(result.data[source_lt_dest] < self.temperature.data[source_lt_dest])
        )

        # test that temperatures are increased where destination orography
        # is lower than source
        source_gt_dest = np.where(self.source_orog.data > self.dest_orog.data)
        self.assertTrue(
            np.all(result.data[source_gt_dest] > self.temperature.data[source_gt_dest])
        )

        # test that temperatures are equal where destination orography
        # is equal to source
        source_eq_dest = np.where(
            np.isclose(self.source_orog.data, self.dest_orog.data)
        )
        np.testing.assert_array_almost_equal(
            result.data[source_eq_dest], self.temperature.data[source_eq_dest]
        )

        # match specific values
        np.testing.assert_array_almost_equal(result.data, self.expected_data)

    def test_unit_adjustment(self):
        """Test correct values are retrieved if input cubes have incorrect
        units"""
        self.temperature.units = "degC"
        self.source_orog.convert_units("km")
        result = self.plugin(
            self.temperature, self.lapse_rate, self.source_orog, self.dest_orog
        )
        self.assertEqual(result.units, "degC")
        np.testing.assert_array_almost_equal(result.data, self.expected_data)

    def test_realizations(self):
        """Test processing of a cube with multiple realizations"""
        temp_3d = add_coordinate(self.temperature, [0, 1, 2], "realization")
        lrt_3d = add_coordinate(self.lapse_rate, [0, 1, 2], "realization")
        result = ApplyGriddedLapseRate()(
            temp_3d, lrt_3d, self.source_orog, self.dest_orog
        )
        np.testing.assert_array_equal(
            result.coord("realization").points, np.array([0, 1, 2])
        )
        for subcube in result.slices_over("realization"):
            np.testing.assert_array_almost_equal(subcube.data, self.expected_data)

    def test_unmatched_realizations(self):
        """Test error if realizations on temperature and lapse rate do not
        match"""
        temp_3d = add_coordinate(self.temperature, [0, 1, 2], "realization")
        lrt_3d = add_coordinate(self.lapse_rate, [2, 3, 4], "realization")
        msg = 'Lapse rate cube coordinate "realization" does not match '
        with self.assertRaisesRegex(ValueError, msg):
            ApplyGriddedLapseRate()(temp_3d, lrt_3d, self.source_orog, self.dest_orog)

    def test_missing_coord(self):
        """Test error if temperature cube has realizations but lapse rate
        does not"""
        temp_3d = add_coordinate(self.temperature, [0, 1, 2], "realization")
        msg = 'Lapse rate cube has no coordinate "realization"'
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin(temp_3d, self.lapse_rate, self.source_orog, self.dest_orog)

    def test_spatial_mismatch(self):
        """Test error if source orography grid is not matched to temperature"""
        new_y_points = self.source_orog.coord(axis="y").points + 100.0
        self.source_orog.coord(axis="y").points = new_y_points
        msg = "Source orography spatial coordinates do not match"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin(
                self.temperature, self.lapse_rate, self.source_orog, self.dest_orog
            )

    def test_spatial_mismatch_2(self):
        """Test error if destination orography grid is not matched to
        temperature"""
        new_y_points = self.dest_orog.coord(axis="y").points + 100.0
        self.dest_orog.coord(axis="y").points = new_y_points
        msg = "Destination orography spatial coordinates do not match"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin(
                self.temperature, self.lapse_rate, self.source_orog, self.dest_orog
            )


@pytest.mark.parametrize(
    "data_limits,data_limits_from_nbhood,expected_local_min,expected_local_max,expected_nbhood",
    [
        # Test case 1: No limits specified
        ((None, None), None, None, None, None),
        # Test case 2: Only minimum limit specified
        ((0.0, None), None, 0.0, None, None),
        # Test case 3: Only maximum limit specified
        ((None, 100.0), None, None, 100.0, None),
        # Test case 4: Both limits specified
        ((0.0, 100.0), None, 0.0, 100.0, None),
        # Test case 5: Neighbourhood-based limits
        ((None, None), 5, None, None, 5),
        # Test case 6: Neighbourhood overrides static limits
        ((0.0, 100.0), 7, None, None, 7),
    ],
)
def test_initialization_valid(
    data_limits,
    data_limits_from_nbhood,
    expected_local_min,
    expected_local_max,
    expected_nbhood,
):
    """Test valid initialization combinations"""
    kwargs = {}
    if data_limits is not None:
        kwargs["data_limits"] = data_limits
    if data_limits_from_nbhood is not None:
        kwargs["data_limits_from_nbhood"] = data_limits_from_nbhood
    plugin = ApplyGriddedLapseRate(**kwargs)

    assert plugin.local_min == expected_local_min
    assert plugin.local_max == expected_local_max
    assert plugin.data_limits_from_nbhood == expected_nbhood


@pytest.mark.parametrize(
    "data_limits_from_nbhood",
    [0, -1, np.nan, -np.inf],
)
def test_initialization_invalid_nbhood(data_limits_from_nbhood):
    """Test that invalid neighbourhood radius raises ValueError"""
    with pytest.raises(ValueError, match="Neighbourhood radius must be at least 1"):
        ApplyGriddedLapseRate(data_limits_from_nbhood=data_limits_from_nbhood)


@pytest.mark.parametrize(
    "min_from_nbhood_uplift, max_from_nbhood_uplift",
    ((-1.0, None), (None, -1.0), (-1.0, -1.0)),
)
def test_initialization_invalid_uplift(min_from_nbhood_uplift, max_from_nbhood_uplift):
    """Test that invalid min and/or max neighbourhood uplifts raise ValueErrors"""
    kwargs = {}
    if min_from_nbhood_uplift is not None:
        kwargs["min_from_nbhood_uplift"] = min_from_nbhood_uplift
    if max_from_nbhood_uplift is not None:
        kwargs["max_from_nbhood_uplift"] = max_from_nbhood_uplift
    with pytest.raises(
        ValueError,
        match="max_from_nbhood_uplift and min_from_nbhood_uplift should be greater than 0.",
    ):
        ApplyGriddedLapseRate(data_limits_from_nbhood=1, **kwargs)


def test_initialization_invalid_uplift_pair():
    """Test that incompatible neighbourhood uplifts raises ValueError"""
    with pytest.raises(
        ValueError,
        match="max_from_nbhood_uplift should be greater than or equal to min_from_nbhood_uplift",
    ):
        ApplyGriddedLapseRate(
            data_limits_from_nbhood=1,
            min_from_nbhood_uplift=1.1,
            max_from_nbhood_uplift=1.0,
        )


@pytest.mark.parametrize("as_array", (False, True))
def test__apply_limits(as_array):
    """Test the _apply_limits method with limits as scalars and arrays"""
    data = np.array([[-10.0, 0.0, 10.0], [50.0, 100.0, 150.0]], dtype=np.float32)
    cube = set_up_variable_cube(data, name="test_data", units="K")
    expected = np.array([[0.0, 0.0, 10.0], [50.0, 100.0, 100.0]])
    plugin = ApplyGriddedLapseRate()
    plugin.local_min = 0.0 if not as_array else np.zeros_like(cube.data)
    plugin.local_max = 100.0 if not as_array else np.full_like(cube.data, 100.0)
    plugin._apply_limits(cube)
    np.testing.assert_array_equal(cube.data, expected)


@pytest.mark.parametrize("min_from_nbhood_uplift", (None, 1.0, 0.9))
@pytest.mark.parametrize("max_from_nbhood_uplift", (None, 1.0, 1.1))
def test__calc_local_limits(min_from_nbhood_uplift, max_from_nbhood_uplift):
    """Test the _calc_local_limits method with a simple input cube"""
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    cube = set_up_variable_cube(data, name="test_data", units="K")
    kwargs = {}
    if min_from_nbhood_uplift is not None:
        kwargs["min_from_nbhood_uplift"] = min_from_nbhood_uplift
        min_uplift = min_from_nbhood_uplift
    else:
        min_uplift = 1.0  # default value
    if max_from_nbhood_uplift is not None:
        kwargs["max_from_nbhood_uplift"] = max_from_nbhood_uplift
        max_uplift = max_from_nbhood_uplift
    else:
        max_uplift = 1.1  # default value
    plugin = ApplyGriddedLapseRate(data_limits_from_nbhood=1, **kwargs)
    plugin._calc_local_limits(cube)
    expected_min = np.array([[1.0, 1.0], [1.0, 2.0]], dtype=np.float32) * min_uplift
    expected_max = np.array([[3.0, 4.0], [4.0, 4.0]], dtype=np.float32) * max_uplift
    np.testing.assert_array_equal(plugin.local_min, expected_min)
    np.testing.assert_array_equal(plugin.local_max, expected_max)


if __name__ == "__main__":
    unittest.main()
