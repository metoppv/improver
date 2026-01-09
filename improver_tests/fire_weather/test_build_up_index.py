# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.fire_weather.build_up_index import BuildUpIndex
from improver_tests.fire_weather import make_input_cubes


def input_cubes(
    dmc_val: float | np.ndarray,
    dc_val: float | np.ndarray,
    shape: tuple[int, ...] = (5, 5),
    dmc_units: str = "1",
    dc_units: str = "1",
) -> CubeList:
    """Create a list of dummy input cubes for BUI tests, with configurable units.

    DMC cube has time coordinates; DC cube does not.

    Args:
        dmc_val:
            DMC value for all grid points.
        dc_val:
            DC value for all grid points.
        shape:
            Shape of the grid for each cube.
        dmc_units:
            Units for DMC cube.
        dc_units:
            Units for DC cube.

    Returns:
        A CubeList of Iris Cubes for DMC and DC.
    """
    return CubeList(
        make_input_cubes(
            [
                ("duff_moisture_code", dmc_val, dmc_units, True),
                ("drought_code", dc_val, dc_units, False),
            ],
            shape=shape,
        )
    )


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
        # Case 9; DC=0, BUI should be close to DMC
        (60.0, 0.0, 58.56),
    ],
)
def test__calculate(
    dmc_val: float,
    dc_val: float,
    expected_bui: float,
):
    """Test calculation of BUI from DMC and DC.

    Verifies BUI calculation from DMC and DC values.

    Args:
        dmc_val:
            DMC value to test.
        dc_val:
            DC value to test.
        expected_bui:
            Expected BUI value.
    """
    cubes = input_cubes(dmc_val=dmc_val, dc_val=dc_val)
    plugin = BuildUpIndex()
    plugin.load_input_cubes(cubes)
    bui = plugin._calculate()

    assert np.allclose(bui, expected_bui, rtol=0.01, atol=0.01)


def test__calculate_spatially_varying() -> None:
    """Test BUI calculation with spatially varying DMC and DC.

    Verifies vectorized BUI calculation with varying values across the grid.
    """
    dmc_data = np.array([[5.0, 10.0, 20.0], [15.0, 25.0, 35.0], [30.0, 45.0, 60.0]])
    dc_data = np.array([[10.0, 20.0, 40.0], [30.0, 50.0, 70.0], [60.0, 90.0, 120.0]])

    cubes = input_cubes(dmc_val=dmc_data, dc_val=dc_data, shape=dmc_data.shape)

    plugin = BuildUpIndex()
    plugin.load_input_cubes(cubes)
    bui = plugin._calculate()

    # Verify shape and all values are non-negative
    assert bui.shape == (3, 3)
    assert np.all(bui >= 0.0)

    # Verify unique values (no broadcast errors)
    assert len(np.unique(bui)) > 1

    # Check specific positions using the equations
    # Position [0,0]: DMC=5, DC=10, use eq 27b
    assert np.allclose(bui[0, 0], 4.90, rtol=0.01)

    # Position [2,2]: DMC=60, DC=120, use eq 27b
    assert np.allclose(bui[2, 2], 59.84, rtol=0.01)


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
    """Integration test for the complete BUI calculation process.

    Verifies end-to-end BUI calculation with various input conditions. Tests both
    equation branches. Edge cases (DMC=0, DC=0, both zero) are covered by dedicated tests.

    Args:
        dmc_val:
            DMC value to test.
        dc_val:
            DC value to test.
        expected_bui:
            Expected BUI output value.
    """
    cubes = input_cubes(dmc_val=dmc_val, dc_val=dc_val)
    result = BuildUpIndex().process(cubes)

    assert isinstance(result, Cube)
    assert result.shape == (5, 5)
    assert result.long_name == "build_up_index"
    assert result.units == "1"
    assert np.allclose(result.data, expected_bui, rtol=0.01, atol=0.01)
    assert result.dtype == np.float32


def test_process_spatially_varying() -> None:
    """Integration test with spatially varying input data.

    Verifies vectorized BUI calculation with varying values across the grid.
    """
    dmc_data = np.array([[5.0, 15.0, 30.0], [10.0, 25.0, 45.0], [20.0, 35.0, 60.0]])
    dc_data = np.array([[10.0, 30.0, 60.0], [20.0, 50.0, 90.0], [40.0, 70.0, 120.0]])

    cubes = input_cubes(dmc_val=dmc_data, dc_val=dc_data, shape=dmc_data.shape)

    result = BuildUpIndex().process(cubes)

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
    assert not np.allclose(result.data[0, 0], result.data[2, 2], atol=0.01)
