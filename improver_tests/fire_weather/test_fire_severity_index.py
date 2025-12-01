# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the FireSeverityIndex plugin."""

from datetime import datetime

import numpy as np
import pytest
from cf_units import Unit
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList

from improver.fire_weather.fire_severity_index import FireSeverityIndex


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
    fwi_val: float = 25.0,
    shape: tuple[int, int] = (5, 5),
    fwi_units: str = "1",
) -> list[Cube]:
    """Create a list of dummy input cubes for DSR tests, with configurable units.

    FWI cube has time coordinates.

    Args:
        fwi_val (float): FWI value for all grid points.
        shape (tuple[int, int]): Shape of the grid for each cube.
        fwi_units (str): Units for FWI cube.

    Returns:
        list[Cube]: List containing FWI Cube.
    """
    # FWI cube needs time coordinates for _make_dsr_cube to copy metadata
    fwi = make_cube(
        np.full(shape, fwi_val),
        "canadian_forest_fire_weather_index",
        fwi_units,
        add_time_coord=True,
    )
    return [fwi]


@pytest.mark.parametrize(
    "fwi_val",
    [
        # Case 0: Typical mid-range value
        25.0,
        # Case 1: Zero
        0.0,
        # Case 2: Low value
        5.0,
        # Case 3: High value
        100.0,
        # Case 4: Very high value
        200.0,
    ],
)
def test_load_input_cubes(fwi_val: float) -> None:
    """Test FireSeverityIndex.load_input_cubes with various input conditions.

    Args:
        fwi_val (float): FWI value for all grid points.

    Raises:
        AssertionError: If the loaded cube does not match expected shape and type.
    """
    cubes = input_cubes(fwi_val)
    plugin = FireSeverityIndex()
    plugin.load_input_cubes(CubeList(cubes))

    assert isinstance(plugin.fire_weather_index, Cube)
    assert plugin.fire_weather_index.shape == (5, 5)
    assert np.allclose(plugin.fire_weather_index.data, fwi_val)


@pytest.mark.parametrize(
    "num_cubes, should_raise, expected_message",
    [
        # Case 0: Correct number of cubes (1)
        (1, False, None),
        # Case 1: No cubes (0 instead of 1)
        (0, True, "Expected 1 cubes, found 0"),
        # Case 2: Too many cubes (2 instead of 1)
        (2, True, "Expected 1 cubes, found 2"),
        # Case 3: Too many cubes (3 instead of 1)
        (3, True, "Expected 1 cubes, found 3"),
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

    plugin = FireSeverityIndex()

    if should_raise:
        with pytest.raises(ValueError, match=expected_message):
            plugin.load_input_cubes(CubeList(cubes))
    else:
        plugin.load_input_cubes(CubeList(cubes))


@pytest.mark.parametrize(
    "fwi_val, expected_dsr",
    [
        # Case 0: Zero FWI
        (0.0, 0.0),
        # Case 1: Low FWI
        (5.0, 0.470),
        # Case 2: Mid-range FWI
        (10.0, 1.602),
        # Case 3: Higher FWI
        (25.0, 8.108),
        # Case 4: High FWI
        (50.0, 27.653),
        # Case 5: Very high FWI
        (100.0, 94.312),
    ],
)
def test__calculate_dsr(
    fwi_val: float,
    expected_dsr: float,
) -> None:
    """Test calculation of DSR from FWI.

    Args:
        fwi_val (float): FWI value to test.
        expected_dsr (float): Expected DSR value.

    Raises:
        AssertionError: If the calculated DSR does not match expected value.
    """
    cubes = input_cubes(fwi_val=fwi_val)
    plugin = FireSeverityIndex()
    plugin.load_input_cubes(CubeList(cubes))
    dsr = plugin._calculate_dsr()

    assert np.allclose(dsr, expected_dsr, rtol=1e-2, atol=0.1)


def test__calculate_dsr_no_negative_values() -> None:
    """Test that DSR calculation never produces negative values."""
    # Test a range of FWI values
    fwi_values = np.array([0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0])

    for fwi in fwi_values:
        cubes = input_cubes(fwi_val=fwi)
        plugin = FireSeverityIndex()
        plugin.load_input_cubes(CubeList(cubes))
        dsr = plugin._calculate_dsr()
        assert np.all(dsr >= 0.0), f"Negative DSR for FWI={fwi}"


def test__calculate_dsr_spatially_varying() -> None:
    """Test DSR calculation with spatially varying FWI (vectorization check)."""
    fwi_data = np.array([[5.0, 10.0, 20.0], [8.0, 15.0, 30.0], [12.0, 25.0, 50.0]])

    cubes = [
        make_cube(
            fwi_data, "canadian_forest_fire_weather_index", "1", add_time_coord=True
        ),
    ]

    plugin = FireSeverityIndex()
    plugin.load_input_cubes(CubeList(cubes))
    dsr = plugin._calculate_dsr()

    # Verify shape and all values are non-negative
    assert dsr.shape == (3, 3)
    assert np.all(dsr >= 0.0)

    # Verify unique values (no broadcast errors)
    assert len(np.unique(dsr)) > 1

    # Check specific position - low FWI should give low DSR
    assert np.allclose(dsr[0, 0], 0.470, rtol=0.02)

    # Check specific position - high FWI should give high DSR
    assert np.allclose(dsr[2, 2], 27.653, rtol=0.02)


@pytest.mark.parametrize(
    "dsr_value, shape",
    [
        # Case 0: Typical DSR value with standard grid
        (14.6, (5, 5)),
        # Case 1: Low DSR value with different grid size
        (1.63, (3, 4)),
        # Case 2: High DSR value with larger grid
        (61.7, (10, 10)),
        # Case 3: Zero DSR with small grid
        (0.0, (2, 2)),
        # Case 4: Another typical DSR value
        (25.0, (5, 5)),
    ],
)
def test__make_dsr_cube(
    dsr_value: float,
    shape: tuple[int, int],
) -> None:
    """Test creation of DSR cube from DSR data.

    Args:
        dsr_value (float): DSR value to use.
        shape (tuple[int, int]): Shape of the grid.

    Raises:
        AssertionError: If the created cube does not have expected properties.
    """
    cubes = input_cubes(fwi_val=25.0, shape=shape)
    plugin = FireSeverityIndex()
    plugin.load_input_cubes(CubeList(cubes))

    dsr_data = np.full(shape, dsr_value)
    dsr_cube = plugin._make_dsr_cube(dsr_data)

    assert isinstance(dsr_cube, Cube)
    assert dsr_cube.shape == shape
    assert dsr_cube.long_name == "fire_severity_index"
    assert dsr_cube.units == "1"
    assert np.allclose(dsr_cube.data, dsr_value)
    assert dsr_cube.dtype == np.float32
    assert dsr_cube.coord("forecast_reference_time")
    assert dsr_cube.coord("time")


@pytest.mark.parametrize(
    "fwi_val, expected_dsr",
    [
        # Case 0: Zero
        (0.0, 0.0),
        # Case 1: Low value
        (10.0, 1.602),
        # Case 2: Mid value
        (25.0, 8.108),
        # Case 3: High value
        (50.0, 27.653),
    ],
)
def test_process(
    fwi_val: float,
    expected_dsr: float,
) -> None:
    """Integration test for process method with various input conditions.

    Args:
        fwi_val (float): FWI value to test.
        expected_dsr (float): Expected DSR output value.

    Raises:
        AssertionError: If the calculated DSR does not match expected value.
    """
    cubes = input_cubes(fwi_val=fwi_val)
    result = FireSeverityIndex().process(CubeList(cubes))

    assert isinstance(result, Cube)
    assert result.shape == (5, 5)
    assert result.long_name == "fire_severity_index"
    assert result.units == "1"
    assert np.allclose(result.data, expected_dsr, rtol=1e-2, atol=0.1)
    assert result.dtype == np.float32


def test_process_spatially_varying() -> None:
    """Integration test with spatially varying data (vectorization check)."""
    fwi_data = np.array([[5.0, 10.0, 20.0], [8.0, 15.0, 30.0], [12.0, 25.0, 50.0]])

    cubes = [
        make_cube(
            fwi_data, "canadian_forest_fire_weather_index", "1", add_time_coord=True
        ),
    ]

    result = FireSeverityIndex().process(CubeList(cubes))

    # Verify shape, type, and all values are non-negative
    assert result.data.shape == (3, 3)
    assert result.data.dtype == np.float32
    assert np.all(result.data >= 0.0)

    # Verify increasing FWI increases DSR
    # Compare bottom-right vs top-left
    assert result.data[2, 2] > result.data[0, 0]

    # Verify unique values (no broadcast errors)
    assert len(np.unique(result.data)) > 1

    # Check that different FWI values produce different DSR outputs
    assert not np.allclose(result.data[0, 0], result.data[2, 2], atol=0.1)


def test_process_zero_fwi() -> None:
    """Test that when FWI=0, DSR equals 0."""
    cubes = input_cubes(fwi_val=0.0)
    result = FireSeverityIndex().process(CubeList(cubes))

    assert isinstance(result, Cube)
    assert result.long_name == "fire_severity_index"
    assert result.units == "1"
    assert np.allclose(result.data, 0.0, atol=1e-6)
    assert result.dtype == np.float32
