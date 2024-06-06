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
"""Unit tests for the ConstructReliabilityCalibrationTables plugin."""

import iris
import numpy as np
import pytest
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from numpy.testing import assert_allclose, assert_array_equal

from improver.calibration.reliability_calibration import (
    AggregateReliabilityCalibrationTables,
)
from improver.calibration.reliability_calibration import (
    ConstructReliabilityCalibrationTables as Plugin,
)

"""Create forecast and truth cubes for use in testing the reliability
calibration plugin. Two forecast and two truth cubes are created, each
pair containing the same data but given different forecast reference
times and validity times. These times maintain the same forecast period
for each forecast cube.

The truth data for reliability calibration is thresholded data, giving
fields of zeroes and ones.

Each forecast cube in conjunction with the contemporaneous truth cube
will be used to produce a reliability calibration table. When testing
the process method here we expect the final reliability calibration
table for a given threshold (we are only using 283K in the value
comparisons) to be the sum of two of these identical tables."""


def test_init_using_defaults():
    """Test init method without providing any arguments."""
    plugin = Plugin()
    assert len(plugin.probability_bins) == 5
    assert plugin.expected_table_shape == (3, 5)


def test_init_with_arguments():
    """Test init method with specified arguments."""
    plugin = Plugin(
        n_probability_bins=4,
        single_value_lower_limit=False,
        single_value_upper_limit=False,
    )
    assert len(plugin.probability_bins) == 4
    assert plugin.expected_table_shape, (3, 4)


def test_dpb_without_single_value_limits():
    """Test the generation of probability bins without single value end
    bins. The range 0 to 1 will be divided into 4 equally sized bins."""
    expected = np.array(
        [[0.0, 0.24999999], [0.25, 0.49999997], [0.5, 0.74999994], [0.75, 1.0]]
    )
    result = Plugin()._define_probability_bins(
        n_probability_bins=4,
        single_value_lower_limit=False,
        single_value_upper_limit=False,
    )
    assert_allclose(result, expected)


def test_dpb_with_both_single_value_limits():
    """Test the generation of probability bins with both upper and lower
    single value end bins. The range 0 to 1 will be divided into 2 equally
    sized bins, with 2 end bins holding values approximately equal to 0 and 1."""
    expected = np.array(
        [
            [0.0000000e00, 1.0000000e-06],
            [1.0000001e-06, 4.9999997e-01],
            [5.0000000e-01, 9.9999893e-01],
            [9.9999899e-01, 1.0000000e00],
        ]
    )
    result = Plugin()._define_probability_bins(
        n_probability_bins=4,
        single_value_lower_limit=True,
        single_value_upper_limit=True,
    )
    assert_allclose(result, expected)


def test_dpb_with_lower_single_value_limit():
    """Test the generation of probability bins with only the lower single value
    limit bin. The range 0 to 1 will be divided into 4 equally sized bins,
    with 1 lower bin holding values approximately equal to 0."""
    expected = np.array(
        [
            [0.0000000e00, 1.0000000e-06],
            [1.0000001e-06, 3.3333331e-01],
            [3.3333334e-01, 6.6666663e-01],
            [6.6666669e-01, 1.0000000e00],
        ],
        dtype=np.float32,
    )

    result = Plugin()._define_probability_bins(
        n_probability_bins=4,
        single_value_lower_limit=True,
        single_value_upper_limit=False,
    )
    assert_allclose(result, expected)


def test_dpb_with_upper_single_value_limit():
    """Test the generation of probability bins with only the upper single value
    limit bin. The range 0 to 1 will be divided into 4 equally sized bins,
    with 1 upper bin holding values approximately equal to 1."""
    expected = np.array(
        [
            [0.0, 0.3333333],
            [0.33333334, 0.6666666],
            [0.6666667, 0.9999989],
            [0.999999, 1.0],
        ],
        dtype=np.float32,
    )

    result = Plugin()._define_probability_bins(
        n_probability_bins=4,
        single_value_lower_limit=False,
        single_value_upper_limit=True,
    )
    assert_allclose(result, expected)


def test_dpb_with_both_single_value_limits_too_few_bins():
    """In this test both lower and uppper single_value_limits are requested
    whilst also trying to use 2 bins. This would leave no bins to cover the
    range 0 to 1, so an error is raised."""

    msg = (
        "Cannot use both single_value_lower_limit and "
        "single_value_upper_limit with 2 or fewer probability bins."
    )
    with pytest.raises(ValueError, match=msg):
        Plugin()._define_probability_bins(
            n_probability_bins=2,
            single_value_lower_limit=True,
            single_value_upper_limit=True,
        )


def test_cpb_coordinate_no_single_value_bins():
    """Test the probability_bins coordinate has the expected values and
    type with no single value lower and upper bins."""
    expected_bounds = np.array([[0, 0.5], [0.5, 1]])
    expected_points = np.mean(expected_bounds, axis=1)
    plugin = Plugin(n_probability_bins=2,)
    result = plugin._create_probability_bins_coord()

    assert isinstance(result, iris.coords.DimCoord)
    assert_allclose(result.points, expected_points)
    assert_allclose(result.bounds, expected_bounds)


def test_cpb_coordinate_single_value_bins():
    """Test the probability_bins coordinate has the expected values and
    type when using the single value lower and upper bins."""
    expected_bounds = np.array(
        [
            [0.0000000e00, 1.0000000e-06],
            [1.0000001e-06, 4.9999997e-01],
            [5.0000000e-01, 9.9999893e-01],
            [9.9999899e-01, 1.0000000e00],
        ]
    )
    expected_points = np.mean(expected_bounds, axis=1)
    plugin = Plugin(
        n_probability_bins=4,
        single_value_lower_limit=True,
        single_value_upper_limit=True,
    )
    result = plugin._create_probability_bins_coord()

    assert isinstance(result, DimCoord)
    assert_allclose(result.points, expected_points)
    assert_allclose(result.bounds, expected_bounds)


def test_crt_coordinates():
    """Test the reliability table coordinates have the expected values and
    type."""
    expected_indices = np.array([0, 1, 2], dtype=np.int32)
    expected_names = np.array(
        ["observation_count", "sum_of_forecast_probabilities", "forecast_count"]
    )
    index_coord, name_coord = Plugin()._create_reliability_table_coords()

    assert isinstance(index_coord, DimCoord)
    assert isinstance(name_coord, AuxCoord)
    assert_array_equal(index_coord.points, expected_indices)
    assert_array_equal(name_coord.points, expected_names)


def test_metadata_with_complete_inputs(forecast_grid, expected_attributes):
    """Test the metadata returned is complete and as expected when the
    forecast cube contains the required metadata to copy."""
    forecast_1 = forecast_grid[0]
    forecast_1.attributes["institution"] = "Kitten Inc"
    expected_attributes["institution"] = "Kitten Inc"
    result = Plugin._define_metadata(forecast_1)
    assert isinstance(result, dict)
    assert result == expected_attributes


def test_metadata_with_incomplete_inputs(forecast_grid, expected_attributes):
    """Test the metadata returned is complete and as expected when the
    forecast cube does not contain all the required metadata to copy."""
    forecast_1 = forecast_grid[0]
    result = Plugin._define_metadata(forecast_1)
    assert isinstance(result, dict)
    assert result == expected_attributes


def test_valid_inputs(create_rel_table_inputs, expected_attributes):
    """Tests correct reliability cube generated. Parameterized using
    `create_rel_table_inputs` fixture."""
    forecast_1 = create_rel_table_inputs.forecast[0]
    forecast_slice = next(forecast_1.slices_over("air_temperature"))
    result = Plugin()._create_reliability_table_cube(
        forecast_slice, forecast_slice.coord(var_name="threshold")
    )
    assert isinstance(result, Cube)
    assert result.shape == create_rel_table_inputs.expected_shape
    assert result.name() == "reliability_calibration_table"
    assert result.attributes == expected_attributes


def test_prb_table_values(create_rel_table_inputs, expected_table):
    """Test the reliability table returned has the expected values for the
    given inputs. Parameterized using `create_rel_table_inputs` fixture."""
    forecast_1 = create_rel_table_inputs.forecast[0]
    truth_1 = create_rel_table_inputs.truth[0]
    forecast_slice = next(forecast_1.slices_over("air_temperature"))
    truth_slice = next(truth_1.slices_over("air_temperature"))
    result = Plugin(
        single_value_lower_limit=True, single_value_upper_limit=True
    )._populate_reliability_bins(forecast_slice.data, truth_slice.data)

    expected_table_shape = create_rel_table_inputs.expected_shape
    assert result.shape == expected_table_shape
    assert_array_equal(result, expected_table.reshape(expected_table_shape))


def test_pmrb_table_values_masked_truth(
    forecast_grid, masked_truths, expected_table_for_mask, expected_table_shape_grid
):
    """Test the reliability table returned has the expected values when a
    masked truth is input."""
    forecast_1 = forecast_grid[0]
    masked_truth_1 = masked_truths[0]
    forecast_slice = next(forecast_1.slices_over("air_temperature"))
    truth_slice = next(masked_truth_1.slices_over("air_temperature"))
    result = Plugin(
        single_value_lower_limit=True, single_value_upper_limit=True
    )._populate_masked_reliability_bins(forecast_slice.data, truth_slice.data)

    assert result.shape == expected_table_shape_grid
    assert np.ma.is_masked(result)
    assert_array_equal(result.data, expected_table_for_mask)
    expected_mask = np.zeros(expected_table_for_mask.shape, dtype=bool)
    expected_mask[:, :, 0, :2] = True
    assert_array_equal(result.mask, expected_mask)


def test_process_return_type(forecast_grid, truth_grid):
    """Test the process method returns a reliability table cube."""
    result = Plugin().process(forecast_grid, truth_grid)
    assert isinstance(result, iris.cube.Cube)
    assert result.name() == "reliability_calibration_table"
    assert result.coord("air_temperature") == forecast_grid.coord(var_name="threshold")
    assert result.coord_dims("air_temperature")[0] == 0


def test_process_table_values(create_rel_table_inputs, expected_table):
    """Test that cube values are as expected, when process has
    sliced the inputs up for processing and then summed the contributions
    from the two dates. Note that the values tested here are for only one
    of the two processed thresholds (283K). The results contain
    contributions from two forecast/truth pairs.

    Parameterized using `create_rel_table_inputs` fixture."""
    expected = np.sum([expected_table, expected_table], axis=0)
    result = Plugin(
        single_value_lower_limit=True, single_value_upper_limit=True
    ).process(create_rel_table_inputs.forecast, create_rel_table_inputs.truth)
    assert_array_equal(
        result[0].data, expected.reshape(create_rel_table_inputs.expected_shape)
    )


def test_process_table_values_nan_or_masked_forecast(
    create_rel_table_inputs, expected_table
):
    """Test that nan or masked values in the forecast are not counted."""

    forecast, truth = create_rel_table_inputs.forecast, create_rel_table_inputs.truth
    nan_ind = list(range(0, forecast.data.size, 2))
    nan_ind_bool = np.zeros_like(forecast.data).astype(bool)
    nan_ind_bool.flat[nan_ind] = 1
    # split the forecast into 2 parts, which have inverse patterns of nans
    forecast_1 = forecast.copy(data=np.where(nan_ind_bool, np.nan, forecast.data))
    forecast_2 = forecast.copy(data=np.where(nan_ind_bool, forecast.data, np.nan))
    expected = np.reshape(
        np.sum([expected_table, expected_table], axis=0),
        create_rel_table_inputs.expected_shape,
    )
    plugin = Plugin(single_value_lower_limit=True, single_value_upper_limit=True)
    result_1 = plugin.process(forecast_1, truth)[0]
    result_2 = plugin.process(forecast_2, truth)[0]
    sum_result = result_1 + result_2
    assert_array_equal(sum_result.data, expected)
    # mask nan values and fill with random data
    forecast_masked = forecast.copy(data=np.ma.masked_invalid(forecast_1.data))
    forecast_masked.data = np.where(
        np.ma.getmask(forecast_1.data),
        np.random.random(forecast_1.data.shape),
        forecast_1.data,
    )
    result = plugin.process(forecast_masked, truth)[0]
    # check masks are preserved
    assert_array_equal(
        np.ma.getmask(result.data),
        np.broadcast_to(np.ma.getmask(forecast_masked.data), result.shape),
    )
    expected = result_1.copy(
        data=np.ma.masked_where(
            result_1.data,
            np.broadcast_to(np.ma.getmask(forecast_masked), result_1.shape),
        )
    )
    # check masked data is not counted in reliability table
    assert_array_equal(result.data, expected.data)


def test_table_values_masked_truth(
    forecast_grid, masked_truths, expected_table_for_mask
):
    """Test, similar to test_table_values, using masked arrays. The
    mask is different for different timesteps, reflecting the potential
    for masked areas in e.g. a radar truth to differ between timesteps.
    At timestep 1, two grid points are masked. At timestep 2, two
    grid points are also masked with one masked grid point in common
    between timesteps. As a result, only one grid point is masked (
    within the upper left corner) within the resulting reliability table."""
    expected_table_for_second_mask = np.array(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.125, 0.25], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.5, 0.625], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.75, 0.875, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ],
        ],
        dtype=np.float32,
    )
    expected = np.sum([expected_table_for_mask, expected_table_for_second_mask], axis=0)
    expected_mask = np.zeros(expected.shape, dtype=bool)
    expected_mask[:, :, 0, 0] = True
    result = Plugin(
        single_value_lower_limit=True, single_value_upper_limit=True
    ).process(forecast_grid, masked_truths)
    assert isinstance(result.data, np.ma.MaskedArray)
    assert_array_equal(result[0].data.data, expected)
    assert_array_equal(result[0].data.mask, expected_mask)
    # Different thresholds must have the same mask.
    assert_array_equal(result[0].data.mask, result[1].data.mask)


def test_process_mismatching_threshold_coordinates(truth_grid, forecast_grid):
    """Test that an exception is raised if the forecast and truth cubes
    have differing threshold coordinates."""
    truths_grid = truth_grid[:, 0, ...]
    msg = "Threshold coordinates differ between forecasts and truths."
    with pytest.raises(ValueError, match=msg):
        Plugin().process(forecast_grid, truths_grid)


def test_process_and_aggregate(create_rel_table_inputs):
    """Test that aggregation during construction produces the same result as
    applying the two plugins sequentially."""
    # use the spatial coordinates for aggregation - input is a parameterised fixture
    if create_rel_table_inputs.forecast.coords("spot_index"):
        agg_coords = ["spot_index"]
    else:
        agg_coords = ["longitude", "latitude"]

    # construct and aggregate as two separate plugins
    constructed = Plugin(
        single_value_lower_limit=True, single_value_upper_limit=True
    ).process(create_rel_table_inputs.forecast, create_rel_table_inputs.truth)
    aggregated = AggregateReliabilityCalibrationTables().process(
        [constructed], agg_coords
    )

    # construct plugin with aggregate_coords option
    constructed_with_agg = Plugin(
        single_value_lower_limit=True, single_value_upper_limit=True
    ).process(
        create_rel_table_inputs.forecast, create_rel_table_inputs.truth, agg_coords
    )

    # check that the two cubes are identical
    assert constructed_with_agg == aggregated
