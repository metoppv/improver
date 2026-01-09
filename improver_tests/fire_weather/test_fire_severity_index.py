# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the FireSeverityIndex plugin."""

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.fire_weather.fire_severity_index import FireSeverityIndex
from improver_tests.fire_weather import make_cube, make_input_cubes


def input_cubes(
    fwi_val: float = 25.0,
    shape: tuple[int, int] = (5, 5),
    fwi_units: str = "1",
) -> list[Cube]:
    """Create a list of dummy input cubes for DSR tests, with configurable units.

    FWI cube has time coordinates.

    Args:
        fwi_val:
            FWI value for all grid points.
        shape:
            Shape of the grid for each cube.
        fwi_units:
            Units for FWI cube.

    Returns:
        List containing FWI Cube.
    """
    return make_input_cubes(
        [("canadian_forest_fire_weather_index", fwi_val, fwi_units, True)],
        shape=shape,
    )


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
def test__calculate(
    fwi_val: float,
    expected_dsr: float,
) -> None:
    """Test calculation of DSR from FWI.

    Verifies DSR calculation from FWI values.

    Args:
        fwi_val:
            FWI value to test.
        expected_dsr:
            Expected DSR value.
    """
    cubes = input_cubes(fwi_val=fwi_val)
    plugin = FireSeverityIndex()
    plugin.load_input_cubes(CubeList(cubes))
    dsr = plugin._calculate()

    assert np.allclose(dsr, expected_dsr, rtol=1e-2, atol=0.1)


@pytest.mark.parametrize(
    "fwi_val",
    [0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0],
)
def test__calculate_no_negative_values(fwi_val: float) -> None:
    """Test that DSR calculation never produces negative values.

    Args:
        fwi_val:
            Fire Weather Index value to test.
    """
    cubes = input_cubes(fwi_val=fwi_val)
    plugin = FireSeverityIndex()
    plugin.load_input_cubes(CubeList(cubes))
    dsr = plugin._calculate()
    assert np.all(dsr >= 0.0), f"Negative DSR for FWI={fwi_val}"


def test__calculate_spatially_varying() -> None:
    """Test DSR calculation with spatially varying FWI.

    Verifies vectorized DSR calculation with varying values across the grid.
    """
    fwi_data = np.array([[5.0, 10.0, 20.0], [8.0, 15.0, 30.0], [12.0, 25.0, 50.0]])

    cubes = [
        make_cube(
            fwi_data, "canadian_forest_fire_weather_index", "1", add_time_coord=True
        ),
    ]

    plugin = FireSeverityIndex()
    plugin.load_input_cubes(CubeList(cubes))
    dsr = plugin._calculate()

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

    Verifies cube creation with proper metadata for DSR.

    Args:
        dsr_value:
            DSR value to use.
        shape:
            Shape of the grid.
    """
    cubes = input_cubes(fwi_val=25.0, shape=shape)
    plugin = FireSeverityIndex()
    plugin.load_input_cubes(CubeList(cubes))

    dsr_data = np.full(shape, dsr_value)
    dsr_cube = plugin._make_output_cube(dsr_data)

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
    """Integration test for the complete DSR calculation process.

    Verifies end-to-end DSR calculation with various FWI input values.

    Args:
        fwi_val:
            FWI value to test.
        expected_dsr:
            Expected DSR output value.
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
    """Integration test with spatially varying FWI data.

    Verifies vectorized DSR calculation with varying values across the grid.
    """
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
