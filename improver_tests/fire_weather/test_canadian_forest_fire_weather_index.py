# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the CanadianForestFireWeatherIndex plugin."""

from datetime import datetime

import numpy as np
import pytest
from cf_units import Unit
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList

from improver.fire_weather.canadian_forest_fire_weather_index import (
    CanadianForestFireWeatherIndex,
)


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
    isi_val: float = 10.0,
    bui_val: float = 50.0,
    shape: tuple[int, int] = (5, 5),
    isi_units: str = "1",
    bui_units: str = "1",
) -> list[Cube]:
    """Create a list of dummy input cubes for FWI tests, with configurable units.

    ISI cube has time coordinates; BUI cube does not (following the pattern).

    Args:
        isi_val (float): ISI value for all grid points.
        bui_val (float): BUI value for all grid points.
        shape (tuple[int, int]): Shape of the grid for each cube.
        isi_units (str): Units for ISI cube.
        bui_units (str): Units for BUI cube.

    Returns:
        list[Cube]: List of Iris Cubes for ISI and BUI.
    """
    # ISI cube needs time coordinates for _make_fwi_cube to copy metadata
    isi = make_cube(
        np.full(shape, isi_val),
        "initial_spread_index",
        isi_units,
        add_time_coord=True,
    )
    bui = make_cube(np.full(shape, bui_val), "build_up_index", bui_units)
    return [isi, bui]


@pytest.mark.parametrize(
    "isi_val, bui_val",
    [
        # Case 0: Typical mid-range values
        (10.0, 50.0),
        # Case 1: Low values
        (0.0, 0.0),
        # Case 2: High values
        (50.0, 200.0),
        # Case 3: Low ISI, high BUI
        (5.0, 150.0),
        # Case 4: High ISI, low BUI
        (40.0, 20.0),
    ],
)
def test_load_input_cubes(
    isi_val: float,
    bui_val: float,
) -> None:
    """Test CanadianForestFireWeatherIndex.load_input_cubes with various input conditions.

    Args:
        isi_val (float): ISI value for all grid points.
        bui_val (float): BUI value for all grid points.

    Raises:
        AssertionError: If the loaded cubes do not match expected shapes and types.
    """
    cubes = input_cubes(isi_val, bui_val)
    plugin = CanadianForestFireWeatherIndex()
    plugin.load_input_cubes(CubeList(cubes))

    attributes = [
        plugin.initial_spread_index,
        plugin.build_up_index,
    ]
    input_values = [isi_val, bui_val]

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

    plugin = CanadianForestFireWeatherIndex()

    if should_raise:
        with pytest.raises(ValueError, match=expected_message):
            plugin.load_input_cubes(CubeList(cubes))
    else:
        plugin.load_input_cubes(CubeList(cubes))


@pytest.mark.parametrize(
    "bui_val, expected_dmf",
    [
        # Case 0: BUI = 0
        (0.0, 2.0),
        # Case 1: BUI = 20 (equation 28a)
        (20.0, 9.065),
        # Case 2: BUI = 50 (equation 28a)
        (50.0, 16.827),
        # Case 3: BUI = 80 boundary (equation 28a)
        (80.0, 23.686),
        # Case 4: BUI = 100 (equation 28b)
        (100.0, 27.861),
        # Case 5: BUI = 200 (equation 28b)
        (200.0, 38.326),
    ],
)
def test__calculate_extrapolated_duff_moisture_function(
    bui_val: float,
    expected_dmf: float,
) -> None:
    """Test calculation of extrapolated DMF from BUI.

    Args:
        bui_val (float): BUI value to test.
        expected_dmf (float): Expected extrapolated DMF value.

    Raises:
        AssertionError: If the calculated DMF does not match expected value.
    """
    cubes = input_cubes(isi_val=10.0, bui_val=bui_val)
    plugin = CanadianForestFireWeatherIndex()
    plugin.load_input_cubes(CubeList(cubes))
    dmf = plugin._calculate_extrapolated_duff_moisture_function()

    assert np.allclose(dmf, expected_dmf, rtol=1e-2, atol=0.1)


def test__calculate_extrapolated_duff_moisture_function_no_negative() -> None:
    """Test that extrapolated DMF calculation never produces negative values."""
    bui_values = np.array([0.0, 10.0, 50.0, 80.0, 100.0, 150.0, 250.0])

    for bui in bui_values:
        cubes = input_cubes(isi_val=10.0, bui_val=bui)
        plugin = CanadianForestFireWeatherIndex()
        plugin.load_input_cubes(CubeList(cubes))
        dmf = plugin._calculate_extrapolated_duff_moisture_function()
        assert np.all(dmf >= 0.0), f"Negative DMF for BUI={bui}"


@pytest.mark.parametrize(
    "isi_val, bui_val, expected_fwi",
    [
        # Case 0: Both zero
        (0.0, 0.0, 0.0),
        # Case 1: BUI = 0 only
        (10.0, 0.0, 3.492),
        # Case 2: ISI = 0 only
        (0.0, 50.0, 0.0),
        # Case 3: Low values, BUI <= 80 (equation 28a), B <= 1 (equation 30b)
        (1.0, 20.0, 0.906),
        # Case 4: Mid values, BUI <= 80 (equation 28a), B > 1 (equation 30a)
        (10.0, 50.0, 22.241),
        # Case 5: BUI = 80 boundary (equation 28a)
        (15.0, 80.0, 37.003),
        # Case 6: High BUI > 80 (equation 28b)
        (20.0, 100.0, 49.368),
        # Case 7: Very high values
        (50.0, 200.0, 103.265),
    ],
)
def test__calculate_fwi(
    isi_val: float,
    bui_val: float,
    expected_fwi: float,
) -> None:
    """Test calculation of FWI from ISI and BUI.

    Args:
        isi_val (float): ISI value to test.
        bui_val (float): BUI value to test.
        expected_fwi (float): Expected FWI value.

    Raises:
        AssertionError: If the calculated FWI does not match expected value.
    """
    cubes = input_cubes(isi_val=isi_val, bui_val=bui_val)
    plugin = CanadianForestFireWeatherIndex()
    plugin.load_input_cubes(CubeList(cubes))
    extrapolated_DMF = plugin._calculate_extrapolated_duff_moisture_function()
    fwi = plugin._calculate_fwi(extrapolated_DMF)

    assert np.allclose(fwi, expected_fwi, rtol=1e-2, atol=0.2)


def test__calculate_fwi_no_negative_values() -> None:
    """Test that FWI calculation never produces negative values."""
    # Test a range of ISI and BUI values
    isi_values = np.array([0.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    bui_values = np.array([0.0, 10.0, 50.0, 80.0, 150.0, 250.0])

    for isi in isi_values:
        for bui in bui_values:
            cubes = input_cubes(isi_val=isi, bui_val=bui)
            plugin = CanadianForestFireWeatherIndex()
            plugin.load_input_cubes(CubeList(cubes))
            extrapolated_DMF = plugin._calculate_extrapolated_duff_moisture_function()
            fwi = plugin._calculate_fwi(extrapolated_DMF)
            assert np.all(fwi >= 0.0), f"Negative FWI for ISI={isi}, BUI={bui}"


def test__calculate_fwi_spatially_varying() -> None:
    """Test FWI calculation with spatially varying ISI and BUI (vectorization check)."""
    isi_data = np.array([[5.0, 10.0, 20.0], [8.0, 15.0, 30.0], [12.0, 25.0, 50.0]])
    bui_data = np.array([[20.0, 40.0, 60.0], [30.0, 70.0, 90.0], [50.0, 100.0, 150.0]])

    cubes = [
        make_cube(isi_data, "initial_spread_index", "1", add_time_coord=True),
        make_cube(bui_data, "build_up_index", "1"),
    ]

    plugin = CanadianForestFireWeatherIndex()
    plugin.load_input_cubes(CubeList(cubes))
    extrapolated_DMF = plugin._calculate_extrapolated_duff_moisture_function()
    fwi = plugin._calculate_fwi(extrapolated_DMF)

    # Verify shape and all values are non-negative
    assert fwi.shape == (3, 3)
    assert np.all(fwi >= 0.0)

    # Verify unique values (no broadcast errors)
    assert len(np.unique(fwi)) > 1

    # Check specific position - low ISI and BUI should give low FWI
    assert fwi[0, 0] < 10.0

    # Check specific position - high ISI and BUI should give high FWI
    assert fwi[2, 2] > 50.0


@pytest.mark.parametrize(
    "fwi_value, shape",
    [
        # Case 0: Typical FWI value with standard grid
        (25.0, (5, 5)),
        # Case 1: Low FWI value with different grid size
        (5.0, (3, 4)),
        # Case 2: High FWI value with larger grid
        (80.0, (10, 10)),
        # Case 3: Zero FWI with small grid
        (0.0, (2, 2)),
        # Case 4: Another typical FWI value
        (45.0, (5, 5)),
    ],
)
def test__make_fwi_cube(
    fwi_value: float,
    shape: tuple[int, int],
) -> None:
    """Test creation of FWI cube from FWI data.

    Args:
        fwi_value (float): FWI value to use.
        shape (tuple[int, int]): Shape of the grid.

    Raises:
        AssertionError: If the created cube does not have expected properties.
    """
    cubes = input_cubes(isi_val=10.0, bui_val=50.0, shape=shape)
    plugin = CanadianForestFireWeatherIndex()
    plugin.load_input_cubes(CubeList(cubes))

    fwi_data = np.full(shape, fwi_value)
    fwi_cube = plugin._make_fwi_cube(fwi_data)

    assert isinstance(fwi_cube, Cube)
    assert fwi_cube.shape == shape
    assert fwi_cube.long_name == "canadian_forest_fire_weather_index"
    assert fwi_cube.units == "1"
    assert np.allclose(fwi_cube.data, fwi_value)
    assert fwi_cube.dtype == np.float32
    assert fwi_cube.coord("forecast_reference_time")
    assert fwi_cube.coord("time")


@pytest.mark.parametrize(
    "isi_val, bui_val, expected_fwi",
    [
        # Case 0: Both zero
        (0.0, 0.0, 0.0),
        # Case 1: Low values
        (1.0, 20.0, 0.906),
        # Case 2: Mid values
        (10.0, 50.0, 22.241),
        # Case 3: High values
        (20.0, 100.0, 49.368),
    ],
)
def test_process(
    isi_val: float,
    bui_val: float,
    expected_fwi: float,
) -> None:
    """Integration test for process method with various input conditions.

    Args:
        isi_val (float): ISI value to test.
        bui_val (float): BUI value to test.
        expected_fwi (float): Expected FWI output value.

    Raises:
        AssertionError: If the calculated FWI does not match expected value.
    """
    cubes = input_cubes(isi_val=isi_val, bui_val=bui_val)
    result = CanadianForestFireWeatherIndex().process(CubeList(cubes))

    assert isinstance(result, Cube)
    assert result.shape == (5, 5)
    assert result.long_name == "canadian_forest_fire_weather_index"
    assert result.units == "1"
    assert np.allclose(result.data, expected_fwi, rtol=1e-2, atol=0.2)
    assert result.dtype == np.float32


def test_process_spatially_varying() -> None:
    """Integration test with spatially varying data (vectorization check)."""
    isi_data = np.array([[5.0, 10.0, 20.0], [8.0, 15.0, 30.0], [12.0, 25.0, 50.0]])
    bui_data = np.array([[20.0, 40.0, 60.0], [30.0, 70.0, 90.0], [50.0, 100.0, 150.0]])

    cubes = [
        make_cube(isi_data, "initial_spread_index", "1", add_time_coord=True),
        make_cube(bui_data, "build_up_index", "1"),
    ]

    result = CanadianForestFireWeatherIndex().process(CubeList(cubes))

    # Verify shape, type, and all values are non-negative
    assert result.data.shape == (3, 3)
    assert result.data.dtype == np.float32
    assert np.all(result.data >= 0.0)

    # Verify increasing ISI and BUI both increase FWI (generally)
    # Compare bottom-right vs top-left
    assert result.data[2, 2] > result.data[0, 0]

    # Verify unique values (no broadcast errors)
    assert len(np.unique(result.data)) > 1

    # Check that different environmental conditions produce different outputs
    assert not np.allclose(result.data[0, 0], result.data[2, 2], atol=1.0)


def test_process_isi_zero() -> None:
    """Test that when ISI=0, FWI equals 0."""
    isi_values = np.zeros((3, 3))
    bui_values = np.array(
        [[10.0, 50.0, 100.0], [20.0, 75.0, 150.0], [30.0, 90.0, 200.0]]
    )

    cubes = [
        make_cube(isi_values, "initial_spread_index", "1", add_time_coord=True),
        make_cube(bui_values, "build_up_index", "1"),
    ]

    result = CanadianForestFireWeatherIndex().process(CubeList(cubes))

    # When ISI=0, FWI should be 0
    assert np.allclose(result.data, 0.0, atol=1e-6)


def test_process_bui_zero() -> None:
    """Test that when BUI=0, FWI is calculated with minimum extrapolated DMF (2.0)."""
    isi_values = np.array([[5.0, 10.0, 20.0], [8.0, 15.0, 30.0], [12.0, 25.0, 50.0]])
    bui_values = np.zeros((3, 3))

    cubes = [
        make_cube(isi_values, "initial_spread_index", "1", add_time_coord=True),
        make_cube(bui_values, "build_up_index", "1"),
    ]

    result = CanadianForestFireWeatherIndex().process(CubeList(cubes))

    # All values should be positive and vary with ISI
    assert np.all(result.data > 0.0)
    assert len(np.unique(result.data)) > 1  # Different ISI values give different FWI
    assert np.isclose(result.data[0, 1], 3.492, rtol=1e-2)


def test_process_both_zero() -> None:
    """Test that when both ISI and BUI are zero, FWI equals 0."""
    cubes = input_cubes(isi_val=0.0, bui_val=0.0)
    result = CanadianForestFireWeatherIndex().process(CubeList(cubes))

    assert isinstance(result, Cube)
    assert result.long_name == "canadian_forest_fire_weather_index"
    assert result.units == "1"
    assert np.allclose(result.data, 0.0, atol=1e-6)
    assert result.dtype == np.float32
