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
"""Unit tests for the threshold.Threshold plugin."""


import pytest
import unittest

import numpy as np
from iris.coords import CellMethod, DimCoord
from iris.cube import Cube, CubeList
from iris.tests import IrisTest

from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)
from improver.threshold import Threshold as Threshold


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        # a threshold of zero is used with a multiplicative fuzzy factor
        (
            {"threshold_values": 0.0, "fuzzy_factor": 0.6},
            "Invalid threshold with fuzzy factor",
        ),
        # a fuzzy factor of minus 1 is given
        (
            {"threshold_values": 0.6, "fuzzy_factor": -1.0},
            "Invalid fuzzy_factor: must be >0 and <1: -1.0",
        ),
        # a fuzzy factor of zero is given
        (
            {"threshold_values": 0.6, "fuzzy_factor": 0.0},
            "Invalid fuzzy_factor: must be >0 and <1: 0.0",
        ),
        # a fuzzy factor of unity is given
        (
            {"threshold_values": 0.6, "fuzzy_factor": 1.0},
            "Invalid fuzzy_factor: must be >0 and <1: 1.0",
        ),
        # a fuzzy factor of 2 is given
        (
            {"threshold_values": 0.6, "fuzzy_factor": 2.0},
            "Invalid fuzzy_factor: must be >0 and <1: 2.0",
        ),
        # fuzzy_factor and fuzzy_bounds both set
        (
            {"threshold_config": {"0.6": [0.4, 0.8]}, "fuzzy_factor": 2.0},
            "Invalid combination of keywords",
        ),
        # fuzzy_bounds contains one value
        (
            {"threshold_config": {"0.6": [0.4]}},
            "Invalid bounds for one threshold: \\(0.4,\\).",
        ),
        # fuzzy_bounds contains three values
        (
            {"threshold_config": {"0.6": [0.4, 0.8, 1.2]}},
            "Invalid bounds for one threshold: \\(0.4, 0.8, 1.2\\).",
        ),
        # fuzzy_bounds do not bound threshold - upper bound too low
        (
            {"threshold_config": {"0.6": [0.4, 0.5]}},
            "Threshold must be within bounds: !\\( 0.4 <= 0.6 <= 0.5 \\)",
        ),
        # fuzzy_bounds do not bound threshold - lower bound too high
        (
            {"threshold_config": {"0.6": [0.7, 0.8]}},
            "Threshold must be within bounds: !\\( 0.7 <= 0.6 <= 0.8 \\)",
        ),
        # comparison_operator is invalid
        (
            {"threshold_values": 0.6, "comparison_operator": "invalid"},
            'String "invalid" does not match any known comparison_operator',
        ),
    ],
)
def test_init(kwargs, exception):
    with pytest.raises(ValueError, match=exception):
        Threshold(**kwargs)


@pytest.mark.parametrize(
    "diagnostic,units", [("precipitation_rate", "mm/hr"), ("air_temperature", "K")]
)
def test__add_threshold_coord(default_cube, diagnostic, units):
    """Test the _add_threshold_coord method for diagnostics with
    different units."""

    ref_cube = default_cube.copy()
    default_cube.rename(diagnostic)
    default_cube.units = units
    plugin = Threshold(threshold_values=1)
    plugin.threshold_coord_name = default_cube.name()
    plugin._add_threshold_coord(default_cube, 1)

    assert default_cube.ndim == ref_cube.ndim
    if diagnostic == "air_temperature":
        assert diagnostic in [
            coord.standard_name for coord in default_cube.coords(dim_coords=False)
        ]
    else:
        assert diagnostic in [
            coord.long_name for coord in default_cube.coords(dim_coords=False)
        ]

    threshold_coord = default_cube.coord(diagnostic)
    assert threshold_coord.var_name == "threshold"
    assert threshold_coord.points[0] == 1
    assert threshold_coord.units == default_cube.units
    assert threshold_coord.dtype == np.float64


@pytest.mark.parametrize(
    "n_realizations,data",
        [
            # A typical case with float inputs
            (1, np.zeros((25), dtype=np.float32).reshape(5, 5)),
            # A case with integer inputs, where the data is converted to
            # float32 type, allowing for non-integer thresholded values,
            # i.e. due to the application of fuzzy thresholds.
            (1, np.zeros((25), dtype=np.int8).reshape(5, 5)),
        ]
)
def test_attributes_and_types(custom_cube, n_realizations, data):
    """Test that the returned cube has the expected type and attributes."""

    expected_attributes = {
        "source": "Unit test",
        "institution": "Met Office",
        "title": "Post-Processed IMPROVER unit test",
    }
    plugin = Threshold(threshold_values=12, fuzzy_factor=(5/6))
    result = plugin(custom_cube)

    assert isinstance(result, Cube)
    assert result.dtype == np.float32
    for key, attribute in expected_attributes.items():
        assert result.attributes[key] == attribute


@pytest.mark.parametrize("comparison_operator", ["gt", "lt", "ge", "le", ">", "<", ">=", "<=", "GT", "LT", "GE", "LE"])
@pytest.mark.parametrize("vicinity", [None, [4000]])
@pytest.mark.parametrize("threshold_values", [0.4])
@pytest.mark.parametrize("threshold_units", ["mm/hr", "mm/day"])
def test_threshold_metadata(
    default_cube,
    threshold_coord,
    expected_cube_name,
    comparison_operator,
    vicinity,
    threshold_values,
    threshold_units,
):
    """"Test that the metadata relating to the thresholding options, on both
    the cube and threshold coordinate is as expected. Many combinations of
    options are tested."""

    kwargs = {
        "threshold_values": threshold_values,
        "comparison_operator": comparison_operator,
        "vicinity": vicinity,
        "threshold_units": threshold_units,
    }
    ref_cube_name = default_cube.name()
    plugin = Threshold(**kwargs)
    result = plugin(default_cube)

    assert result.name() == expected_cube_name.format(cube_name=ref_cube_name)
    assert result.coord(var_name="threshold") == threshold_coord

    if vicinity is not None:
        expected_vicinity = DimCoord(vicinity, long_name="radius_of_vicinity", units="m")
        assert result.coord("radius_of_vicinity") == expected_vicinity


@pytest.mark.parametrize("collapse", (False, True))
@pytest.mark.parametrize("comparator", ("gt", "lt", "le", "ge"))
@pytest.mark.parametrize(
    "kwargs,expected_single_value,expected_multi_value",
    [
        # Note that the expected values given here are for
        # thresholding with a ">" or ">=" comparator. If the comparator
        # is "<" or "<=", the values are inverted as (1 - expected_value).

        # diagnostic value(s) above threshold value
        ({"threshold_values": 0.1}, 1.0, [1., 1.]),
        # diagnostic value(s) below threshold value
        ({"threshold_values": 1.0}, 0.0, [0., 0.]),
        # diagnostic value at threshold value, multi-realization values either side, fuzziness applied
        ({"threshold_values": 0.5, "fuzzy_factor": 0.5}, 0.5, [0.4, 0.6]),
        # diagnostic value(s) above threshold value, fuzziness applied
        ({"threshold_values": 0.4, "fuzzy_factor": 0.5}, 0.75, [0.625, 0.875]),
        # diagnostic value(s) below threshold value, fuzziness applied
        ({"threshold_values": 0.8, "fuzzy_factor": 0.5}, 0.125, [0.0625, 0.1875]),
        # diagnostic value(s) below the fuzzy bounds
        ({"threshold_values": 2.0, "fuzzy_factor": 0.5}, 0.0, [0., 0.]),
        # diagnostic value(s) above the fuzzy bounds
        ({"threshold_values": 0.2, "fuzzy_factor": 0.8}, 1.0, [1., 1.]),
        # asymmetric fuzzy bounds applied, diagnostic value(s) below threshold value
        ({"threshold_config": {"0.6": [0.4, 0.7]}}, 0.25, [0.125, 0.375]),
        # asymmetric fuzzy bounds applied, diagnostic value(s) above threshold value
        ({"threshold_config": {"0.4": [0.0, 0.6]}}, 0.75, [0.625, 0.875]),
        # asymmetric fuzzy bounds applied, diagnostic value(s) at threshold value
        ({"threshold_config": {"0.5": [0.4, 0.7]}}, 0.5, [0.25, 0.625]),
    ],
)
def test_expected_values(default_cube, kwargs, collapse, comparator, expected_result):
    """Test that thresholding yields the expected data values for
    different configurations. Variations tried here are:

      - Different threshold values relative to the diagnostic value(s)
      - Use of fuzzy thresholds, specified in different ways.
      - Deterministic, single realization, and multi-realization inputs
      - Different threshold comparators. Note that the tests have been
        engineered such that there are no cases where the difference
        between "ge" and "gt" is signficant, these are tested elsewhere.
      - Collapsing and not collapsing the realization coordinate when
        present.
    """

    local_kwargs = kwargs.copy()

    if collapse and default_cube.coords("realization", dim_coords=True):
        local_kwargs.update({"collapse_coord": "realization"})

    local_kwargs.update({"comparison_operator": comparator})
    plugin = Threshold(**local_kwargs)
    result = plugin(default_cube)

    assert result.data.shape == expected_result.shape
    assert np.allclose(result.data, expected_result)


class Test_process(IrisTest):

    """Test the thresholding plugin."""

    def test_masked_array(self):
        """Test masked array are handled correctly.
        Masked values are preserved following thresholding."""
        plugin = Threshold(threshold_values=0.1)
        result = plugin(self.masked_cube)
        expected_result_array = np.zeros_like(self.masked_cube.data)
        expected_result_array[2][2] = 1.0
        self.assertArrayAlmostEqual(result.data.data, expected_result_array)
        self.assertArrayEqual(result.data.mask, self.masked_cube.data.mask)

    def test_fill_masked(self):
        """Test plugin when masked points are replaced with fill value"""
        plugin = Threshold(0.6, fill_masked=np.inf)
        result = plugin(self.masked_cube)
        expected_result = np.zeros_like(self.masked_cube.data)
        expected_result[0][0] = 1.0
        self.assertArrayEqual(result.data, expected_result)

    def test_masked_array_fuzzybounds(self):
        """Test masked array are handled correctly when using fuzzy bounds.
        Masked values are preserved following thresholding."""
        bounds = (0.6 * self.fuzzy_factor, 0.6 * (2.0 - self.fuzzy_factor))
        threshold_config = {"0.6": bounds}
        plugin = Threshold(threshold_config=threshold_config)
        result = plugin.process(self.masked_cube)
        expected_result_array = np.zeros_like(self.masked_cube.data)
        expected_result_array[2][2] = 1.0 / 3.0
        self.assertArrayAlmostEqual(result.data.data, expected_result_array)
        self.assertArrayEqual(
            result.data.mask, self.masked_cube.data.mask.reshape((5, 5))
        )

    def test_threshold_boundingzero(self):
        """Test fuzzy threshold of zero."""
        threshold_config = {"0.": [-1.0, 1.0]}
        plugin = Threshold(threshold_config=threshold_config)
        result = plugin(self.cube)
        expected_result_array = np.full_like(self.cube.data, fill_value=0.5)
        expected_result_array[2][2] = 0.75
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_boundingzero_above(self):
        """Test fuzzy threshold of zero where data are above upper-bound."""
        threshold_config = {"0.": [-0.1, 0.1]}
        plugin = Threshold(threshold_config=threshold_config)
        result = plugin(self.cube)
        expected_result_array = np.full_like(self.cube.data, fill_value=0.5)
        expected_result_array[2][2] = 1.0
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_boundingbelowzero(self):
        """Test fuzzy threshold of below-zero."""
        threshold_config = {"0.": [-1.0, 1.0]}
        plugin = Threshold(threshold_config=threshold_config, comparison_operator="<")
        result = plugin(self.cube)
        expected_result_array = np.full_like(self.cube.data, fill_value=0.5)
        expected_result_array[2][2] = 0.25
        self.assertArrayAlmostEqual(result.data, expected_result_array)


    def test_threshold_negative(self):
        """Test a point when the threshold is negative."""
        self.cube.data[2][2] = -0.75
        plugin = Threshold(
            threshold_values=-1.0,
            fuzzy_factor=self.fuzzy_factor,
            comparison_operator="<",
        )
        result = plugin(self.cube)
        expected_result_array = np.zeros_like(self.cube.data)
        expected_result_array[2][2] = 0.25
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_multiple_thresholds(self):
        """Test multiple thresholds applied to a multi-realization cube return a
        single cube arrays corresponding to each realization and threshold."""
        multi_realization_cube = add_coordinate(
            self.cube, [0, 1, 2], "realization", dtype=np.int32
        )
        all_zeroes = np.zeros_like(multi_realization_cube.data)
        one_exceed_point = all_zeroes.copy()
        one_exceed_point[:, 2, 2] = 1.0
        expected_result_array = np.array(
            [one_exceed_point, one_exceed_point, all_zeroes]
        )
        # transpose array so that realization is leading coordinate
        expected_result_array = np.transpose(expected_result_array, [1, 0, 2, 3])
        thresholds = [0.2, 0.4, 0.6]
        plugin = Threshold(threshold_values=thresholds)
        result = plugin(multi_realization_cube)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_unit_conversion(self):
        """Test data are correctly thresholded when the threshold is given in
        units different from that of the input cube.  In this test two
        thresholds (of 4 and 6 mm/h) are used on a 5x5 cube where the
        central data point value is 1.39e-6 m/s (~ 5 mm/h)."""
        expected_result_array = np.zeros((2, 5, 5))
        expected_result_array[0][2][2] = 1.0
        plugin = Threshold(threshold_values=[4.0, 6.0], threshold_units="mm h-1")
        result = plugin(self.rate_cube)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_unit_conversion_2(self):
        """Test threshold coordinate points after undergoing unit conversion.
        Specifically ensuring that small floating point values have no floating
        point precision errors after the conversion (float equality check with no
        tolerance)."""
        plugin = Threshold(threshold_values=[0.03, 0.09, 0.1], threshold_units="mm s-1")
        result = plugin(self.rate_cube)
        self.assertArrayEqual(
            result.coord(var_name="threshold").points,
            np.array([3e-5, 9.0e-05, 1e-4], dtype="float32"),
        )

    def test_threshold_unit_conversion_fuzzy_factor(self):
        """Test for sensible fuzzy factor behaviour when units of threshold
        are different from input cube.  A fuzzy factor of 0.75 is equivalent
        to bounds +/- 25% around the threshold in the given units.  So for a
        threshold of 4 (6) mm/h, the thresholded exceedance probabilities
        increase linearly from 0 at 3 (4.5) mm/h to 1 at 5 (7.5) mm/h."""
        expected_result_array = np.zeros((2, 5, 5))
        expected_result_array[0][2][2] = 1.0
        expected_result_array[1][2][2] = 0.168
        plugin = Threshold(
            threshold_values=[4.0, 6.0], threshold_units="mm h-1", fuzzy_factor=0.75
        )
        result = plugin(self.rate_cube)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_point_nan(self):
        """Test behaviour for a single NaN grid cell."""
        # Need to copy the cube as we're adjusting the data.
        self.cube.data[2][2] = np.NAN
        msg = "NaN detected in input cube data"
        plugin = Threshold(
            threshold_values=2.0,
            fuzzy_factor=self.fuzzy_factor,
            comparison_operator="<",
        )
        with self.assertRaisesRegex(ValueError, msg):
            plugin(self.cube)

    def test_cell_method_updates(self):
        """Test plugin adds correct information to cell methods"""
        self.cube.add_cell_method(CellMethod("max", coords="time"))
        plugin = Threshold(threshold_values=2.0, comparison_operator=">")
        result = plugin(self.cube)
        (cell_method,) = result.cell_methods
        self.assertEqual(cell_method.method, "max")
        self.assertEqual(cell_method.coord_names, ("time",))
        self.assertEqual(cell_method.comments, ("of precipitation_amount",))

    def test_threshold_vicinity(self):
        """Test the thresholding with application of maximum in vicinity
        processing."""
        vicinity = 2000
        self.plugin = Threshold(threshold_values=0.1, vicinity=vicinity)
        expected_result_array = np.zeros_like(self.cube.data)
        expected_result_array[1:4, 1:4] = 1.0

        result = self.plugin(self.cube)
        self.assertArrayAlmostEqual(result.data, expected_result_array)
        self.assertEqual(result.coord(var_name="threshold").shape[0], 1)
        self.assertTrue(result.coord("radius_of_vicinity"))
        self.assertEqual(result.coord("radius_of_vicinity").points, vicinity)

    def test_multi_threshold_vicinity(self):
        """Test the thresholding with application of maximum in vicinity
        processing with multiple thresholds."""
        cube = self.cube.copy()
        cube.data[2, 1] = 0.7
        vicinity = 2000
        self.plugin = Threshold(threshold_values=[0.1, 0.6], vicinity=vicinity)
        threshold1, threshold2 = np.zeros((2, *self.cube.shape))
        threshold1[1:4, 0:4] = 1.0
        threshold2[1:4, 0:3] = 1.0
        expected_result_array = np.stack([threshold1, threshold2])

        result = self.plugin(cube)
        self.assertArrayAlmostEqual(result.data, expected_result_array)
        self.assertEqual(result.coord(var_name="threshold").shape[0], 2)
        self.assertTrue(result.coord("radius_of_vicinity"))
        self.assertEqual(result.coord("radius_of_vicinity").points, vicinity)

    def test_threshold_multi_vicinity(self):
        """Test the thresholding with application of maximum in vicinity
        processing with multiple vicinity radii."""
        vicinity = [2000, 4000]
        self.plugin = Threshold(threshold_values=0.1, vicinity=vicinity)
        vicinity1 = np.zeros_like(self.cube.data)
        vicinity1[1:4, 1:4] = 1.0
        vicinity2 = np.ones_like(self.cube.data)
        expected_result_array = np.stack([vicinity1, vicinity2])

        result = self.plugin(self.cube)
        self.assertArrayAlmostEqual(result.data, expected_result_array)
        self.assertEqual(result.coord(var_name="threshold").shape[0], 1)
        self.assertTrue(result.coord("radius_of_vicinity"))
        self.assertArrayEqual(result.coord("radius_of_vicinity").points, vicinity)

    def test_multi_threshold_multi_vicinity(self):
        """Test the thresholding with application of maximum in vicinity
        processing with multiple thresholds and multiple vicinity radii."""
        cube = self.cube.copy()
        cube.data[2, 1] = 0.7
        vicinity = [2000, 4000]
        self.plugin = Threshold(threshold_values=[0.1, 0.6], vicinity=vicinity)
        t1v1, t1v2, t2v1, t2v2 = np.zeros((4, *self.cube.shape))
        t1v1[1:4, 0:4] = 1.0
        t1v2[:] = 1.0
        t2v1[1:4, 0:3] = 1.0
        t2v2[:, 0:4] = 1.0
        expected_result_array = np.stack([[t1v1, t1v2], [t2v1, t2v2]])

        result = self.plugin(cube)
        self.assertArrayAlmostEqual(result.data, expected_result_array)
        self.assertEqual(result.coord(var_name="threshold").shape[0], 2)
        self.assertTrue(result.coord("radius_of_vicinity"))
        self.assertArrayEqual(result.coord("radius_of_vicinity").points, vicinity)


    def test_percentile_collapse(self):
        """Test the collapsing of the percentile coordinate when thresholding."""
        expected_result_array = np.zeros_like(self.multi_realization_cube.data[0])
        expected_result_array[2][2] = 0.5

        self.multi_realization_cube.coord("realization").rename("percentile")
        self.multi_realization_cube.coord("percentile").points = [0, 100]

        plugin = Threshold(threshold_values=1.0, collapse_coord="percentile")
        result = plugin(self.multi_realization_cube)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_coord_collapse_exception(self):
        """Test that an exception is raised when requesting collapse an
        unsupported coordinate."""
        plugin = Threshold(threshold_values=1.0, collapse_coord="kittens")
        msg = "Can only collapse over a realization coordinate or a percentile"
        with self.assertRaisesRegex(ValueError, msg):
            plugin(self.multi_realization_cube)


if __name__ == "__main__":
    unittest.main()
