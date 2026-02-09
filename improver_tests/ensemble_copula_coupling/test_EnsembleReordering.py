# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Pytest-based tests for the
`ensemble_copula_coupling.EnsembleReordering` plugin.
"""

import itertools

import numpy as np
import pytest
from iris.cube import Cube

from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    EnsembleReordering as Plugin,
)
from improver.synthetic_data.set_up_test_cubes import (
    set_up_percentile_cube,
    set_up_variable_cube,
)

from .ecc_test_data import ECC_TEMPERATURE_REALIZATIONS


@pytest.fixture
def cubes_for_recycle():
    data = np.tile(np.linspace(5, 10, 9), 3).reshape(3, 3, 3)
    data[0] -= 1
    data[1] += 1
    data[2] += 3
    realization_cube = set_up_variable_cube(
        data.astype(np.float32), name="air_temperature", units="degC"
    )
    percentile_cube = set_up_percentile_cube(
        np.sort(data.astype(np.float32), axis=0),
        np.array([10, 50, 90], dtype=np.float32),
        name="air_temperature",
        units="degC",
    )
    return realization_cube, percentile_cube


@pytest.fixture
def cubes_for_rank():
    cube = set_up_variable_cube(ECC_TEMPERATURE_REALIZATIONS.copy())
    cube2d = cube[:, :2, 0].copy()
    return cube, cube2d


@pytest.fixture
def process_cubes():
    raw = set_up_variable_cube(
        ECC_TEMPERATURE_REALIZATIONS.copy(), realizations=[10, 11, 12]
    )
    perc = set_up_percentile_cube(
        np.sort(ECC_TEMPERATURE_REALIZATIONS.copy(), axis=0),
        np.array([25, 50, 75], dtype=np.float32),
    )
    return raw, perc


# Tests for _recycle_raw_ensemble_realizations


def test_recycle_equal(cubes_for_recycle):
    """Test recycling when the number of percentiles and realizations are equal."""
    realization_cube, percentile_cube = cubes_for_recycle
    expected = np.array(
        [
            [[4.0, 4.625, 5.25], [5.875, 6.5, 7.125], [7.75, 8.375, 9.0]],
            [[6.0, 6.625, 7.25], [7.875, 8.5, 9.125], [9.75, 10.375, 11.0]],
            [[8.0, 8.625, 9.25], [9.875, 10.5, 11.125], [11.75, 12.375, 13.0]],
        ]
    )
    result = Plugin()._recycle_raw_ensemble_realizations(
        percentile_cube, realization_cube, "percentile"
    )
    assert isinstance(result, Cube)
    np.testing.assert_array_equal(result.coord("realization").points, [0, 1, 2])
    np.testing.assert_array_almost_equal(result.data, expected)


def test_recycle_more_percentiles(cubes_for_recycle):
    """Test recycling when there are more percentiles than realizations."""
    realization_cube, percentile_cube = cubes_for_recycle
    expected = np.array(
        [
            [[4.0, 4.625, 5.25], [5.875, 6.5, 7.125], [7.75, 8.375, 9.0]],
            [[6.0, 6.625, 7.25], [7.875, 8.5, 9.125], [9.75, 10.375, 11.0]],
            [[4.0, 4.625, 5.25], [5.875, 6.5, 7.125], [7.75, 8.375, 9.0]],
        ]
    )
    raw = realization_cube[:2, :, :]
    raw.coord("realization").points = [12, 13]
    result = Plugin()._recycle_raw_ensemble_realizations(
        percentile_cube, raw, "percentile"
    )
    np.testing.assert_array_equal(result.coord("realization").points, [12, 13, 14])
    np.testing.assert_array_almost_equal(result.data, expected)


def test_recycle_fewer_percentiles(cubes_for_recycle):
    """Test recycling when there are fewer percentiles than realizations."""
    realization_cube, percentile_cube = cubes_for_recycle
    expected = np.array(
        [
            [[4.0, 4.625, 5.25], [5.875, 6.5, 7.125], [7.75, 8.375, 9.0]],
            [[6.0, 6.625, 7.25], [7.875, 8.5, 9.125], [9.75, 10.375, 11.0]],
        ]
    )
    perc = percentile_cube[:2, :, :]
    result = Plugin()._recycle_raw_ensemble_realizations(
        perc, realization_cube, "percentile"
    )
    np.testing.assert_array_equal(result.coord("realization").points, [0, 1])
    np.testing.assert_array_almost_equal(result.data, expected)


# Tests for rank_ecc


def test_rank_ecc_returns_cube(cubes_for_rank):
    """Test that rank_ecc returns a Cube."""
    cube, _ = cubes_for_rank
    raw = np.ones_like(cube.data)
    for i in range(raw.shape[0]):
        raw[i] *= i + 1
    cal = np.copy(raw)
    raw_cube = cube.copy(data=raw)
    cal_cube = cube.copy(data=cal)
    result = Plugin().rank_ecc(cal_cube, raw_cube)
    assert isinstance(result, Cube)


def test_rank_ecc_preserves_order(cubes_for_rank):
    """Test that rank_ecc preserves order when input is already sorted."""
    cube, _ = cubes_for_rank
    arr = np.ones_like(cube.data)
    for i in range(arr.shape[0]):
        arr[i] *= i + 1
    raw_cube = cube.copy(data=arr)
    cal_cube = cube.copy(data=arr)
    result = Plugin().rank_ecc(cal_cube, raw_cube)
    np.testing.assert_array_almost_equal(result.data, arr)


def test_rank_ecc_reorders(cubes_for_rank):
    """Test that rank_ecc reorders the raw data to match the ranks of the calibrated data."""
    cube, _ = cubes_for_rank
    raw = np.array(
        [
            [[5, 5, 5], [7, 5, 5], [5, 5, 5]],
            [[4, 4, 4], [4, 4, 4], [4, 4, 4]],
            [[6, 6, 6], [6, 6, 6], [6, 6, 6]],
        ]
    )
    cal = np.array(
        [
            [[4, 5, 4], [4, 5, 4], [4, 5, 4]],
            [[5, 6, 5], [5, 6, 5], [5, 6, 5]],
            [[6, 7, 6], [6, 7, 6], [6, 7, 6]],
        ]
    )
    expected = np.array(
        [
            [[5, 6, 5], [6, 6, 5], [5, 6, 5]],
            [[4, 5, 4], [4, 5, 4], [4, 5, 4]],
            [[6, 7, 6], [5, 7, 6], [6, 7, 6]],
        ]
    )
    raw_cube = cube.copy(data=raw)
    cal_cube = cube.copy(data=cal)
    result = Plugin().rank_ecc(cal_cube, raw_cube)
    np.testing.assert_array_almost_equal(result.data, expected)


def test_rank_ecc_2d(cubes_for_rank):
    """Test rank_ecc with 2D data."""
    _, cube2d = cubes_for_rank
    raw = np.array([[1, 1], [3, 2], [2, 3]])
    cal = np.array([[1, 1], [2, 2], [3, 3]])
    raw_cube = cube2d.copy(data=raw)
    cal_cube = cube2d.copy(data=cal)
    result = Plugin().rank_ecc(cal_cube, raw_cube)
    np.testing.assert_array_almost_equal(result.data, raw)


def test_rank_ecc_2d_masked(cubes_for_rank):
    """Test rank_ecc with masked 2D data."""
    _, cube2d = cubes_for_rank
    mask = np.array([[True, False], [True, False], [True, False]])
    raw = np.array([[1, 9], [3, 5], [2, 7]])
    cal = np.ma.MaskedArray([[1, 6], [2, 8], [3, 10]], mask=mask, dtype=np.float32)
    expected = np.array([[np.nan, 10], [np.nan, 6], [np.nan, 8]], dtype=np.float32)
    raw_cube = cube2d.copy(data=raw)
    cal_cube = cube2d.copy(data=cal)
    result = Plugin().rank_ecc(cal_cube, raw_cube)
    np.testing.assert_array_almost_equal(result.data.data, expected)
    np.testing.assert_array_equal(result.data.mask, mask)
    assert result.data.dtype == np.float32


def test_rank_ecc_2d_masked_nans(cubes_for_rank):
    """Test rank_ecc with masked 2D data containing NaNs."""
    _, cube2d = cubes_for_rank
    mask = np.array([[True, False], [True, False], [True, False]])
    raw = np.array([[1, 9], [3, 5], [2, 7]])
    cal = np.ma.MaskedArray(
        [[np.nan, 6], [np.nan, 8], [np.nan, 10]], mask=mask, dtype=np.float32
    )
    expected = np.array([[np.nan, 10], [np.nan, 6], [np.nan, 8]], dtype=np.float32)
    raw_cube = cube2d.copy(data=raw)
    cal_cube = cube2d.copy(data=cal)
    result = Plugin().rank_ecc(cal_cube, raw_cube)
    np.testing.assert_array_almost_equal(result.data.data, expected)
    np.testing.assert_array_equal(result.data.mask, mask)
    assert result.data.dtype == np.float32


def test_rank_ecc_2d_tied_random(cubes_for_rank):
    """Test rank_ecc with tied values and random tie-breaking."""
    _, cube2d = cubes_for_rank
    raw = np.array([[1, 1], [3, 2], [2, 2]])
    cal = np.array([[1, 1], [2, 2], [3, 3]])
    possible = [
        np.array([[1, 1], [3, 2], [2, 3]]),
        np.array([[1, 1], [3, 3], [2, 2]]),
    ]
    raw_cube = cube2d.copy(data=raw)
    cal_cube = cube2d.copy(data=cal)
    result = Plugin().rank_ecc(cal_cube, raw_cube)
    assert any(np.array_equal(result.data, p) for p in possible)


def test_rank_ecc_2d_tied_random_seed(cubes_for_rank):
    """Test rank_ecc with tied values and a fixed random seed."""
    _, cube2d = cubes_for_rank
    raw = np.array([[1, 1], [3, 2], [2, 2]])
    cal = np.array([[1, 1], [2, 2], [3, 3]])
    expected = np.array([[1, 1], [3, 3], [2, 2]])
    raw_cube = cube2d.copy(data=raw)
    cal_cube = cube2d.copy(data=cal)
    result = Plugin(random_seed=1).rank_ecc(cal_cube, raw_cube)
    np.testing.assert_array_almost_equal(result.data, expected)


def test_rank_ecc_2d_tied_realization(cubes_for_rank):
    """Test rank_ecc with tied values and realization tie-breaking."""
    _, cube2d = cubes_for_rank
    raw = np.array([[1, 1], [3, 2], [2, 2]])
    cal = np.array([[1, 1], [2, 2], [3, 3]])
    expected = np.array([[1, 1], [3, 2], [2, 3]])
    raw_cube = cube2d.copy(data=raw)
    cal_cube = cube2d.copy(data=cal)
    result = Plugin(tie_break="realization").rank_ecc(cal_cube, raw_cube)
    np.testing.assert_array_almost_equal(result.data, expected)


def test_rank_ecc_1d(cubes_for_rank):
    """Test rank_ecc with 1D data."""
    cube, _ = cubes_for_rank
    raw = np.array([3, 2, 1])
    cal = np.array([1, 2, 3])
    raw_cube = cube[:, 0, 0].copy(data=raw)
    cal_cube = cube[:, 0, 0].copy(data=cal)
    result = Plugin().rank_ecc(cal_cube, raw_cube)
    np.testing.assert_array_almost_equal(result.data, raw)


def test_rank_ecc_1d_random(cubes_for_rank):
    """Test rank_ecc with 1D data and random ordering."""
    cube, _ = cubes_for_rank
    raw = np.array([3, 2, 1])
    cal = np.array([1, 2, 3])
    raw_cube = cube[:, 0, 0].copy(data=raw)
    cal_cube = cube[:, 0, 0].copy(data=cal)
    result = Plugin(random_ordering=True).rank_ecc(cal_cube, raw_cube)
    perms = [np.array(p) for p in itertools.permutations(raw)]
    assert any(np.array_equal(result.data, p) for p in perms)


def test_rank_ecc_bad_tie_break(cubes_for_rank):
    """Test rank_ecc raises ValueError for invalid tie_break argument."""
    _, cube2d = cubes_for_rank
    raw = np.array([[1, 1], [3, 2], [2, 2]])
    cal = np.array([[1, 1], [2, 2], [3, 3]])
    raw_cube = cube2d.copy(data=raw)
    cal_cube = cube2d.copy(data=cal)
    with pytest.raises(
        ValueError,
        match='Input tie_break must be either "random", or "realization", not "kittens".',
    ):
        Plugin(tie_break="kittens").rank_ecc(cal_cube, raw_cube)


# Tests for _check_input_cube_masks


def test_check_input_cube_masks_unmasked(process_cubes):
    """Test _check_input_cube_masks with unmasked input cubes."""
    raw, perc = process_cubes
    Plugin._check_input_cube_masks(perc, raw)


def test_check_input_cube_masks_only_post_processed_masked(process_cubes):
    """Test _check_input_cube_masks with only post-processed cube masked."""
    raw, perc = process_cubes
    perc.data[:, 0, 0] = np.nan
    perc.data = np.ma.masked_invalid(perc.data)
    Plugin._check_input_cube_masks(perc, raw)


def test_check_input_cube_masks_only_raw_masked(process_cubes):
    """Test _check_input_cube_masks raises if only raw cube is masked."""
    raw, perc = process_cubes
    raw.data[:, 0, 0] = np.nan
    raw.data = np.ma.masked_invalid(raw.data)
    with pytest.raises(
        ValueError,
        match="The raw_forecast provided has a mask, but the post_processed_forecast isn't masked.",
    ):
        Plugin._check_input_cube_masks(perc, raw)


def test_check_input_cube_masks_post_processed_inconsistent(process_cubes):
    """Test _check_input_cube_masks raises if post-processed mask is inconsistent."""
    raw, perc = process_cubes
    perc.data[2, 0, 0] = np.nan
    perc.data = np.ma.masked_invalid(perc.data)
    raw.data[:, 0, 0] = np.nan
    raw.data = np.ma.masked_invalid(raw.data)
    with pytest.raises(
        ValueError,
        match="The post_processed_forecast does not have same mask on all x-y slices",
    ):
        Plugin._check_input_cube_masks(perc, raw)


def test_check_input_cube_masks_raw_inconsistent(process_cubes):
    """Test _check_input_cube_masks raises if raw mask is inconsistent."""
    raw, perc = process_cubes
    perc.data[:, 0, 0] = np.nan
    perc.data = np.ma.masked_invalid(perc.data)
    raw.data[2, 0, 0] = np.nan
    raw.data = np.ma.masked_invalid(raw.data)
    with pytest.raises(
        ValueError,
        match="The raw_forecast x-y slices do not all have the same mask as the post_processed_forecast.",
    ):
        Plugin._check_input_cube_masks(perc, raw)


def test_check_input_cube_masks_consistent(process_cubes):
    """Test _check_input_cube_masks with consistent masks."""
    raw, perc = process_cubes
    perc.data[:, 0, 0] = np.nan
    perc.data = np.ma.masked_invalid(perc.data)
    raw.data[:, 0, 0] = np.nan
    raw.data = np.ma.masked_invalid(raw.data)
    Plugin._check_input_cube_masks(perc, raw)


# Tests for process method


def test_process_basic(process_cubes):
    """Test process returns expected cube with realizations."""
    raw, perc = process_cubes
    expected = raw.data.copy()
    result = Plugin().process(perc, raw)
    assert isinstance(result, Cube)
    assert result.coords("realization")
    np.testing.assert_array_almost_equal(result.data, expected)


def test_process_percentile_index(process_cubes):
    """Test process works with percentile_index coordinate."""
    raw, perc = process_cubes
    expected = raw.data.copy()
    perc.coord("percentile").rename("percentile_index")
    perc.coord("percentile_index").points = np.array([0, 1, 2], dtype=np.float32)
    perc.coord("percentile_index").units = "1"
    result = Plugin().process(perc, raw)
    assert isinstance(result, Cube)
    assert result.coords("realization")
    np.testing.assert_array_almost_equal(result.data, expected)


def test_process_masked_input_data(process_cubes):
    """Test process with masked input data containing NaNs."""
    raw, perc = process_cubes
    raw.data[:, 0, 0] = np.nan
    raw.data = np.ma.masked_invalid(raw.data)
    perc.data[:, 0, 0] = np.nan
    perc.data = np.ma.masked_invalid(perc.data)
    expected = raw.data.copy()
    result = Plugin().process(perc, raw)
    assert isinstance(result, Cube)
    assert result.coords("realization")
    np.testing.assert_array_almost_equal(result.data, expected)
    np.testing.assert_array_equal(result.data.mask, expected.mask)


def test_process_masked_input_data_not_nans(process_cubes):
    """Test process with masked input data using a non-NaN mask value."""
    raw, perc = process_cubes
    raw.data[:, 0, 0] = 1000
    raw.data = np.ma.masked_equal(raw.data, 1000)
    perc.data[:, 0, 0] = 1000
    perc.data = np.ma.masked_equal(perc.data, 1000)
    expected = raw.data.copy()
    result = Plugin().process(perc, raw)
    assert isinstance(result, Cube)
    assert result.coords("realization")
    np.testing.assert_array_almost_equal(result.data, expected)
    np.testing.assert_array_equal(result.data.mask, expected.mask)


def test_process_1d_random_ordering(process_cubes):
    """Test process with 1D data and random ordering."""
    raw, perc = process_cubes
    raw_data = np.array([3, 2, 1])
    perc_data = np.array([1, 2, 3])
    raw_cube = raw[:, 0, 0]
    raw_cube.data = raw_data
    perc_cube = perc[:, 0, 0]
    perc_cube.data = perc_data
    result = Plugin(random_ordering=True).process(perc_cube, raw_cube)
    perms = [np.array(p) for p in itertools.permutations(raw_data)]
    assert any(np.array_equal(result.data, p) for p in perms)


def test_process_1d_recycling(process_cubes):
    """Test process with 1D data and recycling realizations."""
    raw, perc = process_cubes
    raw_data = np.array([1, 2])
    perc_data = np.array([1, 2, 3])
    expected1 = np.array([1, 3, 2])
    expected2 = np.array([2, 3, 1])
    raw_cube = raw[:2, 0, 0]
    raw_cube.data = raw_data
    perc_cube = perc[:, 0, 0]
    perc_cube.data = perc_data
    result = Plugin().process(perc_cube, raw_cube)
    assert any(np.array_equal(result.data, e) for e in [expected1, expected2])
