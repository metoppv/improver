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

from datetime import datetime, timedelta

import numpy as np
import pytest
from iris import Constraint
from iris.cube import Cube, CubeList
from iris.util import new_axis
from numpy import ndarray

from improver.calibration.simple_bias_correction import (
    CalculateForecastBias,
    evaluate_additive_error,
)
from improver.cli import generate_percentiles, threshold
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import get_coord_names, get_dim_coord_names

ATTRIBUTES = {
    "title": "Test forecast dataset",
    "model_configuration": "fcst_model",
    "source": "IMPROVER",
    "institution": "Australian Bureau of Meteorology",
}

VALID_TIME = datetime(2022, 12, 6, 3, 0)


def generate_dataset(
    num_frts: int = 1,
    truth_dataset: bool = False,
    data: ndarray = None,
    masked: bool = False,
) -> Cube:
    """Generate sample input datasets.

    Args:
        num_frts:
            Number of forecast_reference_times to use in constructing the dataset.
        truth_dataset:
            Flag to specify whether the dataset represents truth_dataset.
        data:
            Data values to pass into the resultant cube.

    Returns:
        Cube containing input dataset for calculating bias.
    """
    # Setup values around
    attributes = ATTRIBUTES.copy()
    times = [VALID_TIME - i * timedelta(days=1) for i in range(num_frts)]
    if truth_dataset:
        period = timedelta(hours=0)
        attributes["title"] = "Test truth dataset"
        attributes["model_configuration"] = "truth_data"
    else:
        period = timedelta(hours=3)
    forecast_ref_times = {time: time - period for time in times}
    # Initialise random number generator for adding noise around data.
    rng = np.random.default_rng(0)
    if data is None:
        data_shape = (4, 3)
        data = np.ones(shape=data_shape, dtype=np.float32)
    else:
        data_shape = data.shape
    # Construct the cubes.
    if masked:
        data = np.ma.masked_array(data)
        data.mask = np.zeros(shape=data_shape, dtype=bool)
        if truth_dataset:
            data.mask[:, -1] = True
        else:
            data.mask[0, :] = True
    ref_forecast_cubes = CubeList()
    data_mask = data.mask if isinstance(data, np.ma.MaskedArray) else False
    for time in times:
        if (num_frts > 1) and (not truth_dataset):
            noise = rng.normal(0.0, 0.1, data_shape).astype(np.float32)
            data_slice = data + noise
        else:
            data_slice = data

        ref_forecast_cubes.append(
            set_up_variable_cube(
                data=data_slice,
                time=time,
                frt=forecast_ref_times[time],
                attributes=attributes,
            )
        )
    ref_forecast_cube = ref_forecast_cubes.merge_cube()

    return ref_forecast_cube, data_mask


@pytest.mark.parametrize("num_frt", (1, 30))
@pytest.mark.parametrize("mask_truth", (False, True))
@pytest.mark.parametrize("mask_forecast", (False, True))  # , True))
def test_evaluate_additive_error(num_frt, mask_truth, mask_forecast):
    """test additive error evaluation gives expected value (within tolerance)."""
    data = 273.0 + np.array(
        [[1.0, 2.0, 2.0], [2.0, 1.0, 3.0], [3.0, 3.0, 3.0]], dtype=np.float32
    )
    diff = np.array(
        [[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [-2.0, 0.0, 1.0]], dtype=np.float32
    )
    truth_data = data - diff

    historic_forecasts, forecasts_mask = generate_dataset(
        num_frt, data=data, masked=mask_forecast
    )
    truths, truths_mask = generate_dataset(
        num_frt, truth_dataset=True, data=truth_data, masked=mask_truth
    )
    truths.remove_coord("forecast_reference_time")

    result = evaluate_additive_error(historic_forecasts, truths, collapse_dim="time")

    assert np.allclose(result, diff, atol=0.05)
    if mask_forecast or mask_truth:
        assert np.all(result.mask == np.ma.mask_or(truths_mask, forecasts_mask))


# Test case where we have a single or multiple reference forecasts.
@pytest.mark.parametrize("num_frt", (1, 4))
def test__define_metadata(num_frt):
    """Test the resultant metadata is as expected."""
    reference_forecast_cube, _ = generate_dataset(num_frt)

    expected = ATTRIBUTES.copy()
    expected["title"] = "Forecast bias data"
    # Don't expect this attribute to be carried over to forecast bias data.
    del expected["model_configuration"]

    actual = CalculateForecastBias()._define_metadata(reference_forecast_cube)

    assert actual == expected


# Test case where we have a single or multiple reference forecasts.
@pytest.mark.parametrize("num_frt", (1, 4))
def test__create_bias_cube(num_frt):
    """Test that the bias cube has the expected structure."""
    reference_forecast_cube, _ = generate_dataset(num_frt)
    result = CalculateForecastBias()._create_bias_cube(reference_forecast_cube)

    # Check all but the time dim coords are consistent
    expected_dim_coords = set(get_dim_coord_names(reference_forecast_cube))
    actual_dim_coords = set(get_dim_coord_names(result))
    if num_frt > 1:
        assert expected_dim_coords - actual_dim_coords == set(["time"])
    else:
        assert actual_dim_coords == expected_dim_coords

    # dtypes are consistent
    assert reference_forecast_cube.dtype == result.dtype
    # Check that frt coord has expected bounds and values (dependent on whether
    # single or multiple historic forecasts are present).
    if num_frt > 1:
        assert (
            result.coord("forecast_reference_time").points
            == reference_forecast_cube.coord("forecast_reference_time").points[-1]
        )
        assert np.all(
            result.coord("forecast_reference_time").bounds
            == [
                reference_forecast_cube.coord("forecast_reference_time").points[0],
                reference_forecast_cube.coord("forecast_reference_time").points[-1],
            ]
        )
    else:
        assert result.coord("forecast_reference_time") == reference_forecast_cube.coord(
            "forecast_reference_time"
        )

    # Check that time coord has been removed
    assert "time" not in get_coord_names(result)

    # Check variable name is as expected
    assert result.long_name == f"forecast_error_of_{reference_forecast_cube.name()}"


# Test case where we have a single or multiple reference forecasts, and single or multiple
# truth values including case where num_truth_frt != num_fcst_frt.
@pytest.mark.parametrize("num_fcst_frt", (1, 50))
@pytest.mark.parametrize("num_truth_frt", (1, 48, 50))
@pytest.mark.parametrize("mask_truth", (False, True))
@pytest.mark.parametrize("mask_forecast", (False, True))  # , True))
def test_process(num_fcst_frt, num_truth_frt, mask_truth, mask_forecast):
    """Test process function over a variations in number of historical forecasts and
    truth values passed in."""
    reference_forecast_cube, forecasts_mask = generate_dataset(
        num_fcst_frt, masked=mask_forecast
    )
    truth_cube, truth_mask = generate_dataset(
        num_truth_frt, truth_dataset=True, masked=mask_truth
    )

    result = CalculateForecastBias().process(reference_forecast_cube, truth_cube)
    # Check that the values used in calculate mean bias are expected based on
    # alignment of forecast/truth values. For this we will consider the bounds
    # on the forecast_reference_time_coordinate.
    if (num_fcst_frt == 1) or (num_truth_frt == 1):
        expected_bounds = None
    elif num_truth_frt != num_fcst_frt:
        expected_bounds = [
            reference_forecast_cube.coord("forecast_reference_time").points[
                num_fcst_frt - num_truth_frt
            ],
            reference_forecast_cube.coord("forecast_reference_time").points[-1],
        ]
    else:
        expected_bounds = [
            reference_forecast_cube.coord("forecast_reference_time").points[0],
            reference_forecast_cube.coord("forecast_reference_time").points[-1],
        ]
    assert np.all(result.coord("forecast_reference_time").bounds == expected_bounds)
    # Check that dtypes match for input/output
    assert result.dtype == reference_forecast_cube.dtype
    # Check that results are near zero
    # Note: case of single truth value and multiple forecasts will have larger deviation
    # from expected value, so here we use a larger tolerance.
    expected_tol = 0.2 if (num_truth_frt == 1 and num_fcst_frt > 1) else 0.05
    assert np.allclose(result.data, 0.0, atol=expected_tol)
    if mask_forecast or mask_truth:
        assert np.all(result.data.mask == np.ma.mask_or(truth_mask, forecasts_mask))


@pytest.mark.parametrize("num_fcst_frt", (1, 5))
@pytest.mark.parametrize("single_value_as_dim_coord", (True, False))
def test_ensure_single_valued_forecast(num_fcst_frt, single_value_as_dim_coord):
    """Test the check to ensure that forecasts are single-valued realisable
    forecasts."""
    # Test the case where inputs have an associated ensemble dimension with
    # length > 1.
    data = np.ones(shape=(4, 3, 3), dtype=np.float32)
    # Test realization data
    realization_cube, _ = generate_dataset(num_frts=num_fcst_frt, data=data)
    with pytest.raises(ValueError, match="Multiple realization values"):
        CalculateForecastBias()._ensure_single_valued_forecast(realization_cube)
    # Test percentile data
    percentile_cube = generate_percentiles.process(
        realization_cube, coordinates="realization", percentiles=[25, 50, 75]
    )
    with pytest.raises(ValueError, match="Multiple percentile values"):
        CalculateForecastBias()._ensure_single_valued_forecast(percentile_cube)
    # Test threshold data; here we test case of thresholds as scalar coord (single-valued)
    # which should also raise exception as probability data is not supported.
    for threshold_set in ([273], [273, 275]):
        threshold_cube = threshold.process(
            realization_cube,
            threshold_values=threshold_set,
            threshold_units="K",
            comparison_operator="ge",
            collapse_coord="realization",
        )
        with pytest.raises(ValueError, match="provided as probability data"):
            CalculateForecastBias()._ensure_single_valued_forecast(threshold_cube)

    # Test the case where inputs have an associated ensemble coord of length 1.
    # Test realization data.
    expected_realization = realization_cube.extract(
        constraint=Constraint(realization=0)
    )
    single_realization_cube = (
        new_axis(expected_realization, "realization")
        if single_value_as_dim_coord
        else expected_realization
    )
    result = CalculateForecastBias()._ensure_single_valued_forecast(
        single_realization_cube
    )
    assert result == expected_realization
    expected_percentile = generate_percentiles.process(
        realization_cube, coordinates="realization", percentiles=[50]
    )
    # Test percentile data.
    single_percentile_cube = (
        new_axis(expected_percentile, "percentile")
        if single_value_as_dim_coord
        else expected_percentile
    )
    result = CalculateForecastBias()._ensure_single_valued_forecast(
        single_percentile_cube
    )
    assert result == expected_percentile

    # Test the case where the input data does not have an associated ensemble coord.
    cube_without_ens_coord, _ = generate_dataset(num_frts=num_fcst_frt)
    result = CalculateForecastBias()._ensure_single_valued_forecast(
        cube_without_ens_coord
    )
    assert result == cube_without_ens_coord
