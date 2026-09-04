# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for DurationSubdivision plugin."""

from datetime import datetime as dt
from datetime import timedelta
from typing import List, Optional, Tuple

import numpy as np
import pytest
from iris.cube import Cube, CubeList
from numpy.testing import assert_array_equal

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.temporal_interpolation import DurationSubdivision


def _grid_params(spatial_grid: str, npoints: int) -> Tuple[Tuple[float, float], float]:
    """Set domain corner and grid spacing for lat-lon or equal area
    projections.

    Args:
        spatial_grid:
            "latlon" or "equalarea" to determine the type of projection.
        npoints:
            The number of grid points to use in both x and y.
    Returns:
        A tuple containing a further tuple that includes the grid corner
        coordinates, and a single value specifying the grid spacing.
    """

    domain_corner = None
    grid_spacing = None
    if spatial_grid == "latlon":
        domain_corner = (40, -20)
        grid_spacing = 40 / (npoints - 1)
    elif spatial_grid == "equalarea":
        domain_corner = (-100000, -400000)
        grid_spacing = np.around(1000000.0 / npoints)
    return domain_corner, grid_spacing


def diagnostic_cube(
    data: np.ndarray,
    time: dt = dt(2024, 6, 15, 12),
    frt: dt = dt(2024, 6, 15, 6),
    period: int = 3600,
    spatial_grid: str = "latlon",
    realizations: Optional[List] = None,
) -> Cube:
    """Return a diagnostic cube containing the provided data.

    Args:
        data:
            The data to be contained in the cube.
        time:
            A datetime object that gives the validity time of the cube.
        frt:
            The forecast reference time for the cube.
        period:
            The length of the period represented by the cube in seconds.
        spatial_grid:
            Whether this is a lat-lon or equal areas projection.
        realizations:
            An optional list of realizations identifiers. The length of this
            list will determine how many realizations are created.
    Returns:
        A diagnostic cube for use in testing.
    """
    npoints = data.shape[-1]
    if npoints != data.shape[-2]:
        raise ValueError("diagnostic_cube function assumes square grid.")
    domain_corner, grid_spacing = _grid_params(spatial_grid, npoints)

    if realizations and data.ndim != 3:
        data = np.stack([data] * len(realizations))

    time_bounds = [time - timedelta(seconds=period), time]

    return set_up_variable_cube(
        data,
        time=time,
        frt=frt,
        time_bounds=time_bounds,
        spatial_grid=spatial_grid,
        domain_corner=domain_corner,
        x_grid_spacing=grid_spacing,
        y_grid_spacing=grid_spacing,
        realizations=realizations,
    )


def multi_time_cube(
    times: List,
    data: np.ndarray,
    spatial_grid: str = "latlon",
    period: int = 3600,
    realizations: Optional[List] = None,
) -> Cube:
    """Return a multi-time diagnostic cube containing the provided data.

    Args:
        times:
            A list of datetime objects that gives the validity times for
            the cube.
        data:
            The data to be contained in the cube. If the cube is 3-D the
            leading dimension should be the same size as the list of times
            and will be sliced to associate each slice with each time.
        spatial_grid:
            Whether this is a lat-lon or equal areas projection.
        bounds:
            If True return time coordinates with time bounds.
        realizations:
            An optional list of realizations identifiers. The length of this
            list will determine how many realizations are created.
    Returns:
        A diagnostic cube for use in testing.
    """
    cubes = CubeList()
    if data.ndim == 2:
        data = np.stack([data] * len(times))

    frt = sorted(times)[0] - (times[1] - times[0])  # Such that guess bounds are +ve
    for time, data_slice in zip(times, data):
        cubes.append(
            diagnostic_cube(
                data_slice,
                time=time,
                frt=frt,
                period=period,
                spatial_grid=spatial_grid,
                realizations=realizations,
            )
        )
    return cubes.merge_cube()


def fidelity_cube(data: np.ndarray, period: int, fidelity_period: int) -> Cube:
    """Define a cube with an equally spaced leading time coordinate that
    splits the period defined into shorter periods equal in length to the
    fidelity period. Divide the data amongst these equally."""

    time = dt(2024, 6, 15, 21)
    intervals = period // fidelity_period
    times = [
        time - (i * timedelta(seconds=fidelity_period)) for i in range(0, intervals)
    ][::-1]
    fidelity_data = data / intervals
    return multi_time_cube(
        times, fidelity_data.astype(np.float32), period=fidelity_period
    )


@pytest.fixture
def basic_cube(period: int) -> Cube:
    """Define a cube with default values except for the period."""

    data = np.ones((5, 5), dtype=np.float32)
    return diagnostic_cube(data, period=period)


@pytest.fixture
def data_cube(data: np.ndarray, time: dt, period: int, realizations: List[int]) -> Cube:
    """Define a cube with specific period, data, and validity time."""

    return diagnostic_cube(
        data.astype(np.float32), time=time, period=period, realizations=realizations
    )


@pytest.fixture
def renormalisation_cubes(times: List[dt], data: np.ndarray) -> Tuple[Cube, Cube]:
    """Define a cube with an equally spaced leading time coordinate and
    specific data, which is a equal subdivision of the input data which
    is also returned in a single period cube."""
    period = (times[1] - times[0]).total_seconds() * len(times)
    fidelity_data = data / len(times)
    fidelity_period = period / len(times)
    fidelity_cube = multi_time_cube(
        times, fidelity_data.astype(np.float32), period=fidelity_period
    )
    input_cube = diagnostic_cube(data.astype(np.float32), time=times[-1], period=period)
    return input_cube, fidelity_cube


@pytest.mark.parametrize(
    "kwargs,mask_value,exception",
    [
        (
            {
                "target_period": 60,
                "fidelity": 10,
                "night_mask": False,
                "day_mask": False,
            },
            None,  # Expected mask value
            None,  # Expected exception
        ),  # Check plugin initialised correctly without a mask.
        (
            {
                "target_period": 360,
                "fidelity": 1,
                "night_mask": True,
                "day_mask": False,
            },
            0,  # Expected mask value
            None,  # Expected exception
        ),  # Check plugin initialised correctly with night mask option.
        (
            {"target_period": 10, "fidelity": 2, "night_mask": False, "day_mask": True},
            1,  # Expected mask value
            None,  # Expected exception
        ),  # Check plugin initialised correctly with day mask option.
        (
            {"target_period": 360, "fidelity": 1, "night_mask": True, "day_mask": True},
            None,  # Expected mask value
            "Only one or neither of night_mask and day_mask may be set to True",  # Expected exception
        ),  # Check plugin raises exception if day and night mask options both set True.
        (
            {
                "target_period": 0,
                "fidelity": 10,
                "night_mask": False,
                "day_mask": False,
            },
            None,  # Expected mask value
            (
                "Target period and fidelity must be a positive integer numbers of seconds. "
                "Currently set to target_period: 0, fidelity: 10"
            ),  # Expected exception
        ),  # Check plugin raises exception if target period is 0.
        (
            {
                "target_period": -100,
                "fidelity": 10,
                "night_mask": False,
                "day_mask": False,
            },
            None,  # Expected mask value
            (
                "Target period and fidelity must be a positive integer numbers of seconds. "
                "Currently set to target_period: -100, fidelity: 10"
            ),  # Expected exception
        ),  # Check plugin raises exception if target period is < 0.
        (
            {
                "target_period": 100,
                "fidelity": 0,
                "night_mask": False,
                "day_mask": False,
            },
            None,  # Expected mask value
            (
                "Target period and fidelity must be a positive integer numbers of seconds. "
                "Currently set to target_period: 100, fidelity: 0"
            ),  # Expected exception
        ),  # Check plugin raises exception if fidelity is 0.
        (
            {
                "target_period": 100,
                "fidelity": -10,
                "night_mask": False,
                "day_mask": False,
            },
            None,  # Expected mask value
            (
                "Target period and fidelity must be a positive integer numbers of seconds. "
                "Currently set to target_period: 100, fidelity: -10"
            ),  # Expected exception
        ),  # Check plugin raises exception if fidelity is < 0.
    ],
)
def test__init__(kwargs, mask_value, exception):
    """Test plugin configured as expected and exceptions raised by the
    __init__ method."""

    if exception is not None:
        with pytest.raises(ValueError, match=exception):
            DurationSubdivision(**kwargs)
    else:
        plugin = DurationSubdivision(**kwargs)
        for key in [key for key in kwargs.keys() if "mask" not in key]:
            assert getattr(plugin, key) == kwargs[key]
        assert plugin.mask_value == mask_value


@pytest.mark.parametrize(
    "period",
    [60, 120, 3600],  # Diagnostic periods in seconds.
)
def test_cube_period(basic_cube, period):
    """Test that this method returns the cube period in seconds."""

    plugin = DurationSubdivision(60, 1)
    result = plugin.cube_period(basic_cube)
    assert result == period


@pytest.mark.parametrize("realizations", [None, (1, 2)])
@pytest.mark.parametrize(
    "kwargs,data,time,period",
    [
        (
            {
                "target_period": 3600,
                "fidelity": 1800,
                "night_mask": False,
                "day_mask": False,
            },
            np.full((10, 10), 10800),  # Data in the input cube.
            dt(2024, 6, 15, 12),  # Validity time
            10800,  # Input period
        ),
        (
            {
                "target_period": 3600,
                "fidelity": 1800,
                "night_mask": True,
                "day_mask": False,
            },
            np.full((10, 10), 7200),  # Data in the input cube.
            dt(2024, 6, 15, 21),  # Validity time
            3600,  # Input period
        ),
        (
            {
                "target_period": 3600,
                "fidelity": 900,
                "night_mask": False,
                "day_mask": True,
            },
            np.full((10, 10), 10800),  # Data in the input cube.
            dt(2024, 6, 15, 21),  # Validity time
            10800,  # Input period
        ),
    ],
)
def test_allocate_data_for_target_period(
    data_cube, kwargs, data, time, period, realizations
):
    """Test that allocate_data_for_target_period allocates data to the fidelity
    periods within a single target period correctly, and that the metadata
    associated with these shorter periods is correct.

    Unlike the original allocate_data method, this method only constructs
    fidelity cubes for one target period at a time. We therefore test it
    for just the first target period.
    """
    plugin = DurationSubdivision(**kwargs)
    intervals_per_target_period = kwargs["target_period"] // kwargs["fidelity"]
    start_time = data_cube.coord("time").bounds.flatten()[0]

    result = plugin.allocate_data_for_target_period(data_cube, period, int(start_time))

    # Check the correct number of fidelity cubes are returned for a single target period.
    assert len(result) == intervals_per_target_period

    # Check fidelity cubes have the correct time metadata.
    for i, fidelity_cube in enumerate(result):
        expected_lb = start_time + i * kwargs["fidelity"]
        expected_ub = start_time + (i + 1) * kwargs["fidelity"]
        assert fidelity_cube.coord("time").bounds[0][0] == expected_lb
        assert fidelity_cube.coord("time").bounds[0][1] == expected_ub
        assert fidelity_cube.coord("time").points[0] == expected_ub
        (bounds,) = np.unique(np.diff(fidelity_cube.coord("time").bounds, axis=1))
        assert bounds == kwargs["fidelity"]

    # If realizations is not None, look at each fidelity period in turn.
    for fidelity_cube in result:
        if realizations is not None:
            rslices = list(fidelity_cube.slices_over("realization"))
        else:
            rslices = [fidelity_cube]

        for rslice in rslices:
            total_intervals = period // kwargs["fidelity"]
            if not any([kwargs[key] for key in kwargs.keys() if "mask" in key]):
                # Without masking each fidelity period should be an equal fraction
                # of the original data.
                np.testing.assert_array_almost_equal(
                    rslice.data, data / total_intervals
                )
            else:
                # With masking applied some points should be zeroed.
                assert (rslice.data <= data / total_intervals).all()


@pytest.mark.parametrize(
    "masked", [False, True]
)  # Make the input cube data into a masked numpy array if True.
@pytest.mark.parametrize(
    "data,time,period,fidelity",
    [
        (
            np.full((3, 3), 10800),  # Data in the input cube.
            dt(2024, 6, 15, 13),  # Validity time - sets fidelity cubes at 10, 11, 12Z
            10800,  # Input period (3 hours)
            3600,  # Fidelity period (1 hour)
        ),
        (
            np.full((3, 3), 3600),  # Data in the input cube.
            dt(2024, 6, 15, 12),  # Validity time - sets fidelity cubes at 10:30, 12Z
            7200,  # Input period (2 hours)
            3600,  # Fidelity period (1 hour)
        ),
    ],
)
def test__compute_renormalisation_factor(
    data,
    time,
    period,
    fidelity,
    masked,
):
    """Test the _compute_renormalisation_factor method returns the array of
    renormalisation factors.

    Since _compute_renormalisation_factor only applies masking when
    mask_value is not None, and we instantiate the plugin with
    night_mask=False and day_mask=False (mask_value=None), every fidelity
    period retains its full allocation. The retotal therefore equals
    cube.data and the factor should be 1 everywhere, regardless of the
    number of fidelity periods or whether the data array is masked.
    """
    plugin = DurationSubdivision(
        target_period=fidelity, fidelity=fidelity, night_mask=False, day_mask=False
    )  # mask_value=None so no masking is applied.

    cube = diagnostic_cube(data.astype(np.float32), time=time, period=period)

    if masked:
        cube.data = np.ma.masked_array(cube.data)

    result = plugin._compute_renormalisation_factor(cube, period)

    # Without masking (mask_value=None), every fidelity period retains its
    # allocation, so retotal == cube.data and factor == 1 everywhere.
    assert_array_equal(result, np.ones(cube.shape[-2:]))


@pytest.mark.parametrize(
    "kwargs,data,time,period,target_start_offset,target_end_offset,expected_data",
    [
        (
            {
                "target_period": 3600,
                "fidelity": 3600,
                "night_mask": False,
                "day_mask": False,
            },
            np.full((3, 3), 10800),  # Data in the input cube.
            dt(2024, 6, 15, 12),  # Validity time
            10800,  # Input period (3 hours)
            0,  # target_start is the start of the full period
            3600,  # target_end is 1 hour later
            np.full((3, 3), 3600),  # Expected: 10800 / 3 target periods
        ),  # fidelity == target_period: simple subdivision, no intermediate processing.
        (
            {
                "target_period": 3600,
                "fidelity": 1800,
                "night_mask": False,
                "day_mask": False,
            },
            np.full((3, 3), 10800),  # Data in the input cube.
            dt(2024, 6, 15, 12),  # Validity time
            10800,  # Input period (3 hours)
            0,  # target_start is the start of the full period
            3600,  # target_end is 1 hour later
            np.full((3, 3), 3600),  # Expected: 10800 / 3 target periods
        ),  # fidelity < target_period: fidelity cubes are constructed, masked,
        # renormalised, and collapsed. Without masking the result should be
        # identical to simple subdivision.
        (
            {
                "target_period": 3600,
                "fidelity": 1800,
                "night_mask": True,
                "day_mask": False,
            },
            np.full((3, 3), 7200),  # Data in the input cube.
            dt(2024, 6, 15, 21),  # Validity time
            7200,  # Input period (2 hours)
            0,  # target_start is the start of the full period
            3600,  # target_end is 1 hour later
            np.array(
                [[3600, 1800, 0], [3600, 3600, 3600], [3600, 3600, 3600]],
                dtype=np.float32,
            ),  # Expected: night masking applied, some points zeroed or reduced.
        ),  # Night masking is applied: the result for the first target period should
        # match the first time slice of the full process() night-mask test.
    ],
)
def test__process_target_period(
    kwargs, data, time, period, target_start_offset, target_end_offset, expected_data
):
    """Test the _process_target_period method constructs, masks, renormalises,
    and collapses fidelity cubes into a single target period cube correctly.

    This tests the method in isolation for a single target period, verifying
    both the data values and the time coordinate metadata on the returned cube.
    """
    plugin = DurationSubdivision(**kwargs)
    cube = diagnostic_cube(data.astype(np.float32), time=time, period=period)

    # Clip input data as process() does before calling this method.
    cube.data = np.clip(cube.data, 0, period, dtype=cube.data.dtype)

    start_time, _ = cube.coord("time").bounds.flatten()
    target_start = int(start_time) + target_start_offset
    target_end = target_start + target_end_offset
    n_target_periods = period // kwargs["target_period"]

    # Compute factor as process() does.
    if plugin.mask_value is not None:
        factor = plugin._compute_renormalisation_factor(cube, period)
    else:
        factor = np.ones_like(cube.data, dtype=np.float64)

    result = plugin._process_target_period(
        cube, period, n_target_periods, target_start, target_end, factor
    )

    # Check a single cube is returned.
    assert isinstance(result, Cube)

    # Check the time bounds and point are correct.
    assert result.coord("time").bounds[0][0] == target_start
    assert result.coord("time").bounds[0][1] == target_end
    assert result.coord("time").points[0] == target_end

    # Check the period of the returned cube is the target period.
    (bounds,) = np.unique(np.diff(result.coord("time").bounds, axis=1))
    assert bounds == kwargs["target_period"]

    # Check the data values are as expected.
    np.testing.assert_array_almost_equal(result.data, expected_data)


@pytest.mark.parametrize(
    "kwargs,data,time,period,expected,realizations,exception",
    [
        (
            {
                "target_period": 3600,
                "fidelity": None,  # Demonstrate the fidelity argument set to None.
                "night_mask": False,
                "day_mask": False,
            },
            np.full((3, 3), 10800),  # Data in the input cube.
            dt(2024, 6, 15, 12),  # Validity time
            10800,  # Input period
            np.full((3, 3, 3), 3600),  # Expected data in the output cube.
            None,  # List of realization numbers if any
            None,  # Expected exception
        ),
        (
            {
                "target_period": 1100,
                "fidelity": 550,
                "night_mask": False,
                "day_mask": False,
            },
            np.full((3, 3), 10800),  # Data in the input cube.
            dt(2024, 6, 15, 12),  # Validity time
            10800,  # Input period
            None,  # Expected data in the output cube (not used here).
            None,  # List of realization numbers if any
            "The target period must be a factor of the original period",  # Expected exception
        ),  # Raise a ValueError as the target period is not a factor of the input period.
        (
            {
                "target_period": 3600,
                "fidelity": 7200,
                "night_mask": False,
                "day_mask": False,
            },
            np.full((3, 3), 10800),  # Data in the input cube.
            dt(2024, 6, 15, 12),  # Validity time
            10800,  # Input period
            None,  # Expected data in the output cube (not used here).
            None,  # List of realization numbers if any
            "The fidelity period must be less than or equal to the target period.",  # Expected exception
        ),  # Raise a ValueError as the fidelity period is longer than the target period.
        (
            {
                "target_period": 7200,
                "fidelity": 1800,
                "night_mask": False,
                "day_mask": False,
            },
            np.full((3, 3), 3600),  # Data in the input cube.
            dt(2024, 6, 15, 12),  # Validity time
            3600,  # Input period
            None,  # Expected data in the output cube (not used here).
            None,  # List of realization numbers if any
            (
                "The target period must be a factor of the original period "
                "of the input cube and the target period must be <= the input "
                "period. Input period: 3600, target period: 7200"
            ),  # Expected exception
        ),  # Raise a ValueError as the target period is longer than the input period.
        (
            {
                "target_period": 3600,
                "fidelity": 1800,
                "night_mask": False,
                "day_mask": False,
            },
            np.full((3, 3), 3600),  # Data in the input cube.
            dt(2024, 6, 15, 12),  # Validity time
            3600,  # Input period
            None,  # Expected data in the output cube (not used here).
            None,  # List of realization numbers if any
            None,  # Expected exception
        ),  # Return the input cube completely unchanged as the target period matches the input period.
        (
            {
                "target_period": 3600,
                "fidelity": 1800,
                "night_mask": False,
                "day_mask": False,
            },
            np.full((3, 3), 10800),  # Data in the input cube.
            dt(2024, 6, 15, 21),  # Validity time
            7200,  # Input period
            np.full((2, 3, 3), 3600),  # Expected data in the output cube.
            None,  # List of realization numbers if any
            None,  # Expected exception
        ),  # Demonstate clipping of input data as input data exceeds the period.
        (
            {
                "target_period": 3600,
                "fidelity": 900,
                "night_mask": True,
                "day_mask": False,
            },
            np.full((2, 2), 1800),  # Data in the input cube.
            dt(2024, 6, 15, 21),  # Validity time
            7200,  # Input period
            np.array(
                [
                    [[1028.5714, 0], [900, 900]],
                    [[771.4286, 0], [900, 900]],
                ],
                dtype=np.float32,
            ),  # Expected data in the output cube.
            None,  # List of realization numbers if any
            None,  # Expected exception
        ),  # A 2-hour period containing just 30 minutes of sunshine duration.
        # The night mask is applied, which affects the the northern most latitudes.
        # The north-east most cell (top-right) is entirely zeroed in both target
        # times, meaning all of the sunshine duration is lost. The north-west most
        # cell (top-left) is masked in a single of the 900 second fidelity periods
        # that is generated. This results in the original 1800 seconds of sunshine
        # duration being renormalised in the day light period fidelity periods
        # for this cell. Instead of 225 seconds (1800 / 8) each fidelity period
        # contains (1800 / 7) seconds of sunshine duration. This is within the 900
        # seconds possible, so no clipping is applied. The total across the two
        # final 1-hour periods generated for this north-west cell is still 1800
        # seconds of sunshine duration, but it is split unevenly to reflect that the
        # later period is partially a night time period.
        (
            {
                "target_period": 3600,
                "fidelity": 1800,
                "night_mask": True,
                "day_mask": False,
            },
            np.full((3, 3), 7200),  # Data in the input cube.
            dt(2024, 6, 15, 21),  # Validity time
            7200,  # Input period
            np.array(
                [
                    [[3600, 1800, 0], [3600, 3600, 3600], [3600, 3600, 3600]],
                    [[3600, 0, 0], [3600, 3600, 0], [3600, 3600, 3600]],
                ],
                dtype=np.float32,
            ),  # Expected data in the output cube.
            None,  # List of realization numbers if any
            None,  # Expected exception
        ),  # Night masking applies time dependent masking, meaning we get different data
        # in each of the returned periods. The total in any period does not exceed the
        # length of that period; this means that the total across periods is lower than
        # in the input.
        (
            {
                "target_period": 3600,
                "fidelity": 1800,
                "night_mask": False,
                "day_mask": True,
            },
            np.full((3, 3), 7200),  # Data in the input cube.
            dt(2024, 6, 15, 21),  # Validity time
            7200,  # Input period
            np.array(
                [
                    [[0, 1800, 3600], [0, 0, 0], [0, 0, 0]],
                    [[0, 3600, 3600], [0, 0, 3600], [0, 0, 0]],
                ],
                dtype=np.float32,
            ),  # Expected data in the output cube.
            None,  # List of realization numbers if any
            None,  # Expected exception
        ),  # Day masking applies time dependent masking. This is the inverse of the test
        # above, and we get a suitably inverted result.
        (
            {
                "target_period": 3600,
                "fidelity": 360,
                "night_mask": True,
                "day_mask": False,
            },
            np.full((3, 3), 7200),  # Data in the input cube.
            dt(2024, 6, 15, 21),  # Validity time
            7200,  # Input period
            np.array(
                [
                    [[3600, 1440, 0], [3600, 3600, 3240], [3600, 3600, 3600]],
                    [[2880, 0, 0], [3600, 3600, 0], [3600, 3600, 3600]],
                ],
                dtype=np.float32,
            ),  # Expected data in the output cube.
            None,  # List of realization numbers if any
            None,  # Expected exception
        ),  # Repeat the night masking test above but with higher fidelity, meaning that
        # the subdivided data is split into shorter periods and the night mask
        # applied more accurately. As a result we get fractions returned which are
        # multiples of this fidelity period.
        (
            {
                "target_period": 3600,
                "fidelity": 1800,
                "night_mask": True,
                "day_mask": False,
            },
            np.array(
                [
                    [
                        [7200, 7200, 7200],
                        [7200, 7200, 7200],
                        [7200, 7200, 7200],
                    ],
                    [
                        [3600, 3600, 3600],
                        [3600, 3600, 3600],
                        [3600, 3600, 3600],
                    ],
                ],
                dtype=np.float32,
            ),  # Data in the input cube.
            dt(2024, 6, 15, 21),  # Validity time
            7200,  # Input period
            np.array(
                [
                    [
                        [[3600, 1800, 0], [3600, 3600, 3600], [3600, 3600, 3600]],
                        [
                            [1800.0, 1800.0, 0.0],
                            [1800.0, 1800.0, 3600.0],
                            [1800.0, 1800.0, 1800.0],
                        ],
                    ],
                    [
                        [[3600, 0, 0], [3600, 3600, 0], [3600, 3600, 3600]],
                        [
                            [1800.0, 0.0, 0.0],
                            [1800.0, 1800.0, 0.0],
                            [1800.0, 1800.0, 1800.0],
                        ],
                    ],
                ],
                dtype=np.float32,
            ),  # Expected data in the output cube.
            [1, 2],  # List of realization numbers if any
            None,  # Expected exception
        ),  # Repeat the night masking test above but with two realizations
        # to demonstrate this is handled sensibly when these contain
        # different data. We recover the same solution for the first
        # realization that we got in the first night mask test; note it
        # is split in the expected data as the order is time, realization.
        # The second realization gives a different result, though the same
        # points have been modified by the application of the mask.
    ],
)
def test_process(kwargs, data_cube, period, expected, exception, realizations):
    """Test the process method returns the expected data or raises the
    expected exceptions. Includes an example for multi-realization data."""

    plugin = DurationSubdivision(**kwargs)
    if exception is not None:
        with pytest.raises(ValueError, match=exception):
            plugin.process(data_cube)
    else:
        result = plugin.process(data_cube)

        if kwargs["target_period"] == period:
            assert result is data_cube
        else:
            # Check data returned is as expected.
            np.testing.assert_array_almost_equal(result.data, expected)
            # Check periods returned are correct.
            (bounds,) = np.unique(np.diff(result.coord("time").bounds, axis=1))
            assert bounds == kwargs["target_period"]
            (bounds,) = np.unique(
                np.diff(result.coord("forecast_period").bounds, axis=1)
            )
            assert bounds == kwargs["target_period"]
