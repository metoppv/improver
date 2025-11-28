# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

from datetime import datetime

import numpy as np
import pytest
from cf_units import Unit
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList

from improver.fire_weather.build_up_index import BuildUpIndex


def make_cube(
    data: np.ndarray,
    name: str,
    units: str,
    add_time_coord: bool = False,
) -> Cube:
    """Create a dummy Iris Cube with specified data, name, units, and optional
    time coordinates.

    All cubes include a forecast_reference_time coordinate by default.

    Args:
        data (np.ndarray): The data array for the cube.
        name (str): The long name for the cube.
        units (str): The units for the cube.
        add_time_coord (bool): Whether to add a time coordinate with bounds.

    Returns:
        Cube: The constructed Iris Cube with the given properties.
    """
    arr = np.array(data, dtype=np.float64)
    cube = Cube(arr, long_name=name)
    cube.units = units

    # Always add forecast_reference_time
    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"

    # Default forecast reference time: 2025-10-20 00:00:00
    frt = datetime(2025, 10, 20, 0, 0)
    frt_coord = AuxCoord(
        np.array([frt.timestamp() / 3600], dtype=np.float64),
        standard_name="forecast_reference_time",
        units=Unit(time_origin, calendar=calendar),
    )
    cube.add_aux_coord(frt_coord)

    # Optionally add time coordinate with bounds
    if add_time_coord:
        # Default valid time: 2025-10-20 12:00:00 with 12-hour bounds
        valid_time = datetime(2025, 10, 20, 12, 0)
        time_bounds = np.array(
            [
                [
                    (valid_time.timestamp() - 43200) / 3600,
                    valid_time.timestamp() / 3600,
                ]
            ],
            dtype=np.float64,
        )
        time_coord = AuxCoord(
            np.array([valid_time.timestamp() / 3600], dtype=np.float64),
            standard_name="time",
            bounds=time_bounds,
            units=Unit(time_origin, calendar=calendar),
        )
        cube.add_aux_coord(time_coord)

    return cube


def input_cubes(
    dmc_val: float = 10.0,
    dc_val: float = 15.0,
    shape: tuple[int, int] = (5, 5),
    dmc_units: str = "1",
    dc_units: str = "1",
) -> list[Cube]:
    """Create a list of dummy input cubes for BUI tests, with configurable units.

    DMC cube has time coordinates; DC cube does not (following the pattern).

    Args:
        dmc_val (float): DMC value for all grid points.
        dc_val (float): DC value for all grid points.
        shape (tuple[int, int]): Shape of the grid for each cube.
        dmc_units (str): Units for DMC cube.
        dc_units (str): Units for DC cube.

    Returns:
        list[Cube]: List of Iris Cubes for DMC and DC.
    """
    # DMC cube needs time coordinates for _make_bui_cube to copy metadata
    dmc = make_cube(
        np.full(shape, dmc_val),
        "duff_moisture_code",
        dmc_units,
        add_time_coord=True,
    )
    dc = make_cube(np.full(shape, dc_val), "drought_code", dc_units)
    return [dmc, dc]


@pytest.mark.parametrize(
    "dmc_val, dc_val",
    [
        # Case 0: Typical mid-range values
        (10.0, 15.0),
        # Case 1: Low values
        (0.0, 0.0),
        # Case 2: High values
        (100.0, 500.0),
        # Case 3: Low DMC, high DC
        (5.0, 400.0),
        # Case 4: High DMC, low DC
        (80.0, 20.0),
    ],
)
def test_load_input_cubes(
    dmc_val: float,
    dc_val: float,
) -> None:
    """Test BuildUpIndex.load_input_cubes with various input conditions.

    Args:
        dmc_val (float): DMC value for all grid points.
        dc_val (float): DC value for all grid points.

    Raises:
        AssertionError: If the loaded cubes do not match expected shapes and types.
    """
    cubes = input_cubes(dmc_val, dc_val)
    plugin = BuildUpIndex()
    plugin.load_input_cubes(CubeList(cubes))

    attributes = [
        plugin.duff_moisture_code,
        plugin.drought_code,
    ]
    input_values = [dmc_val, dc_val]

    for attr, val in zip(attributes, input_values):
        assert isinstance(attr, Cube)
        assert attr.shape == (5, 5)
        assert np.allclose(attr.data, val)


@pytest.mark.parametrize(
    "num_cubes, should_raise, expected_message",
    [
        # Case 0: Correct number of cubes (2)
        (2, False, None),
        # Case 1: Too few cubes (1 instead of 2)
        (1, True, "Expected 2 cubes, found 1"),
        # Case 2: No cubes (0 instead of 2)
        (0, True, "Expected 2 cubes, found 0"),
        # Case 3: Too many cubes (3 instead of 2)
        (3, True, "Expected 2 cubes, found 3"),
    ],
)
def test_load_input_cubes_wrong_number_raises_error(
    num_cubes: int,
    should_raise: bool,
    expected_message: str,
) -> None:
    """Test that load_input_cubes raises ValueError when given wrong number of cubes.

    Args:
        num_cubes (int): Number of cubes to provide to load_input_cubes.
        should_raise (bool): Whether a ValueError should be raised.
        expected_message (str): Expected error message (or None if no error expected).

    Raises:
        AssertionError: If ValueError behavior does not match expectations.
    """
    # Create a list with the specified number of cubes
    cubes = input_cubes()
    if num_cubes < len(cubes):
        cubes = cubes[:num_cubes]
    elif num_cubes > len(cubes):
        # Add dummy cubes
        for _ in range(num_cubes - len(cubes)):
            cubes.append(make_cube(np.full((5, 5), 0.0), "dummy", "1"))

    plugin = BuildUpIndex()

    if should_raise:
        with pytest.raises(ValueError, match=expected_message):
            plugin.load_input_cubes(CubeList(cubes))
    else:
        plugin.load_input_cubes(CubeList(cubes))


@pytest.mark.parametrize(
    "dmc_val, dc_val, expected_bui",
    [
        # Case 0: Both zero (special case)
        (0.0, 0.0, 0.0),
        # Case 1: DC only (DMC=0) -> BUI should be 0
        (0.0, 50.0, 0.0),
        # Case 2: Small DMC, large DC (DMC <= 0.4*DC)
        (5.0, 30.0, 7.06),
        # Case 3: DMC=10, DC=30 (DMC <= 0.4*DC)
        (10.0, 30.0, 10.91),
        # Case 4: DMC > 0.4*DC case
        (20.0, 30.0, 19.76),
        # Case 5: Equal DMC and DC (DMC > 0.4*DC)
        (50.0, 50.0, 49.41),
        # Case 6: High DC (DMC <= 0.4*DC)
        (30.0, 200.0, 43.64),
        # Case 7: High DMC and DC (DMC > 0.4*DC)
        (45.9, 123.9, 47.67),
        # Case 8: High values with DMC > 0.4*DC
        (100.0, 150.0, 99.46),
    ],
)
def test__calculate_bui(
    dmc_val: float,
    dc_val: float,
    expected_bui: float,
) -> None:
    """Test calculation of BUI from DMC and DC.

    Args:
        dmc_val (float): DMC value to test.
        dc_val (float): DC value to test.
        expected_bui (float): Expected BUI value.

    Raises:
        AssertionError: If the calculated BUI does not match expected value.
    """
    cubes = input_cubes(dmc_val=dmc_val, dc_val=dc_val)
    plugin = BuildUpIndex()
    plugin.load_input_cubes(CubeList(cubes))
    bui = plugin._calculate_bui()

    assert np.allclose(bui, expected_bui, rtol=1e-2, atol=0.1)


def test__calculate_bui_no_negative_values() -> None:
    """Test that BUI calculation never produces negative values."""
    # Test a range of DMC and DC values
    dmc_values = np.array([0.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    dc_values = np.array([0.0, 10.0, 30.0, 100.0, 300.0, 500.0])

    for dmc in dmc_values:
        for dc in dc_values:
            cubes = input_cubes(dmc_val=dmc, dc_val=dc)
            plugin = BuildUpIndex()
            plugin.load_input_cubes(CubeList(cubes))
            bui = plugin._calculate_bui()
            assert np.all(bui >= 0.0), f"Negative BUI for DMC={dmc}, DC={dc}"


def test__calculate_bui_spatially_varying() -> None:
    """Test BUI calculation with spatially varying DMC and DC (vectorization check)."""
    dmc_data = np.array([[5.0, 10.0, 20.0], [15.0, 25.0, 35.0], [30.0, 45.0, 60.0]])
    dc_data = np.array([[10.0, 20.0, 40.0], [30.0, 50.0, 70.0], [60.0, 90.0, 120.0]])

    cubes = [
        make_cube(dmc_data, "duff_moisture_code", "1", add_time_coord=True),
        make_cube(dc_data, "drought_code", "1"),
    ]

    plugin = BuildUpIndex()
    plugin.load_input_cubes(CubeList(cubes))
    bui = plugin._calculate_bui()

    # Verify shape and all values are non-negative
    assert bui.shape == (3, 3)
    assert np.all(bui >= 0.0)

    # Verify unique values (no broadcast errors)
    assert len(np.unique(bui)) > 1

    # Check specific positions using the equations
    # Position [0,0]: DMC=5, DC=10, use eq 27b
    assert np.allclose(bui[0, 0], 4.90, rtol=0.02)

    # Position [2,2]: DMC=60, DC=120, use eq 27b
    assert np.allclose(bui[2, 2], 59.84, rtol=0.02)


@pytest.mark.parametrize(
    "bui_value, shape",
    [
        # Case 0: Typical BUI value with standard grid
        (15.3, (5, 5)),
        # Case 1: Low BUI value with different grid size
        (2.5, (3, 4)),
        # Case 2: High BUI value with larger grid
        (80.0, (10, 10)),
        # Case 3: Zero BUI with small grid
        (0.0, (2, 2)),
        # Case 4: Another typical BUI value
        (27.0, (5, 5)),
    ],
)
def test__make_bui_cube(
    bui_value: float,
    shape: tuple[int, int],
) -> None:
    """Test creation of BUI cube from BUI data.

    Args:
        bui_value (float): BUI value to use.
        shape (tuple[int, int]): Shape of the grid.

    Raises:
        AssertionError: If the created cube does not have expected properties.
    """
    cubes = input_cubes(dmc_val=10.0, dc_val=15.0, shape=shape)
    plugin = BuildUpIndex()
    plugin.load_input_cubes(CubeList(cubes))

    bui_data = np.full(shape, bui_value)
    bui_cube = plugin._make_bui_cube(bui_data)

    assert isinstance(bui_cube, Cube)
    assert bui_cube.shape == shape
    assert bui_cube.long_name == "build_up_index"
    assert bui_cube.units == "1"
    assert np.allclose(bui_cube.data, bui_value)
    assert bui_cube.dtype == np.float32
    assert bui_cube.coord("forecast_reference_time")
    assert bui_cube.coord("time")


@pytest.mark.parametrize(
    "dmc_val, dc_val, expected_bui",
    [
        # Case 0: Typical mid-range values (DMC <= 0.4*DC, use eq 27a)
        (10.0, 30.0, 10.91),
        # Case 1: DMC > 0.4*DC case (use eq 27b)
        (20.0, 30.0, 19.76),
        # Case 2: High DC (DMC <= 0.4*DC, use eq 27a)
        (30.0, 200.0, 43.64),
        # Case 3: High values with DMC > 0.4*DC (use eq 27b)
        (100.0, 150.0, 99.46),
    ],
)
def test_process(
    dmc_val: float,
    dc_val: float,
    expected_bui: float,
) -> None:
    """Integration test for process method with typical input conditions.

    Tests both equation branches with representative values.
    Edge cases (DMC=0, DC=0, both zero) are covered by dedicated tests.

    Args:
        dmc_val (float): DMC value to test.
        dc_val (float): DC value to test.
        expected_bui (float): Expected BUI output value.

    Raises:
        AssertionError: If the calculated BUI does not match expected value.
    """
    cubes = input_cubes(dmc_val=dmc_val, dc_val=dc_val)
    result = BuildUpIndex().process(CubeList(cubes))

    assert isinstance(result, Cube)
    assert result.shape == (5, 5)
    assert result.long_name == "build_up_index"
    assert result.units == "1"
    assert np.allclose(result.data, expected_bui, rtol=1e-2, atol=0.1)
    assert result.dtype == np.float32


def test_process_spatially_varying() -> None:
    """Integration test with spatially varying data (vectorization check)."""
    dmc_data = np.array([[5.0, 15.0, 30.0], [10.0, 25.0, 45.0], [20.0, 35.0, 60.0]])
    dc_data = np.array([[10.0, 30.0, 60.0], [20.0, 50.0, 90.0], [40.0, 70.0, 120.0]])

    cubes = [
        make_cube(dmc_data, "duff_moisture_code", "1", add_time_coord=True),
        make_cube(dc_data, "drought_code", "1"),
    ]

    result = BuildUpIndex().process(CubeList(cubes))

    # Verify shape, type, and all values are non-negative
    assert result.data.shape == (3, 3)
    assert result.data.dtype == np.float32
    assert np.all(result.data >= 0.0)

    # Verify increasing DMC and DC both increase BUI (generally)
    # Compare bottom-right vs top-left
    assert result.data[2, 2] > result.data[0, 0]

    # Verify unique values (no broadcast errors)
    assert len(np.unique(result.data)) > 1

    # Check that different environmental conditions produce different outputs
    assert not np.allclose(result.data[0, 0], result.data[2, 2], atol=1.0)


def test_process_dmc_only() -> None:
    """Test that when DC=0, equation 27a gives BUI close to DMC."""
    dmc_values = np.array([[10.0, 20.0, 30.0], [15.0, 25.0, 35.0], [5.0, 40.0, 50.0]])
    dc_values = np.zeros((3, 3))

    # Calculate expected BUI for each DMC value when DC=0
    expected_bui = dmc_values - (0.92 + (0.0114 * dmc_values) ** 1.7)

    cubes = [
        make_cube(dmc_values, "duff_moisture_code", "1", add_time_coord=True),
        make_cube(dc_values, "drought_code", "1"),
    ]

    result = BuildUpIndex().process(CubeList(cubes))

    # When DC=0, BUI = DMC - [0.92 + (0.0114*DMC)^1.7]
    assert np.allclose(result.data, expected_bui, rtol=1e-3)


def test_process_dc_only() -> None:
    """Test that when DMC=0, BUI equals 0."""
    dmc_values = np.zeros((3, 3))
    dc_values = np.array(
        [[10.0, 50.0, 100.0], [20.0, 75.0, 150.0], [30.0, 90.0, 200.0]]
    )

    cubes = [
        make_cube(dmc_values, "duff_moisture_code", "1", add_time_coord=True),
        make_cube(dc_values, "drought_code", "1"),
    ]

    result = BuildUpIndex().process(CubeList(cubes))

    # When DMC=0, BUI should be 0 (no fuel moisture contribution)
    assert np.allclose(result.data, 0.0, atol=1e-6)


def test_process_both_zero() -> None:
    """Test that when both DMC and DC are zero, BUI equals 0."""
    cubes = input_cubes(dmc_val=0.0, dc_val=0.0)
    result = BuildUpIndex().process(CubeList(cubes))

    assert isinstance(result, Cube)
    assert result.long_name == "build_up_index"
    assert result.units == "1"
    assert np.allclose(result.data, 0.0, atol=1e-6)
    assert result.dtype == np.float32
