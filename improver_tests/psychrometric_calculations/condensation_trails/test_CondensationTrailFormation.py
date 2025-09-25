# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the CondensationTrailFormation plugin"""

from typing import List, Optional, Tuple

import numpy as np
import pytest

from improver.psychrometric_calculations.condensation_trails import (
    CondensationTrailFormation,
)
from improver.psychrometric_calculations.psychrometric_calculations import (
    calculate_svp_in_air,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

LOCAL_MANDATORY_ATTRIBUTES = {
    "title": "unit test data",
    "source": "unit test",
    "institution": "somewhere",
}


@pytest.mark.parametrize(
    "provided_engine_contrail_factors, expected_factors",
    [
        (None, np.array([3e-5, 3.4e-5, 3.9e-5], dtype=np.float32)),  # Test default
        ([1e-5, 1.4e-5, 1.9e-5], np.array([1e-5, 1.4e-5, 1.9e-5], dtype=np.float32)),
        ([2e-5, 2.4e-5, 2.9e-5], np.array([2e-5, 2.4e-5, 2.9e-5], dtype=np.float32)),
        ([3e-5, 3.4e-5, 3.9e-5], np.array([3e-5, 3.4e-5, 3.9e-5], dtype=np.float32)),
    ],
)
def test_initialisation_with_arguments_and_defaults(
    provided_engine_contrail_factors: Optional[List[float]],
    expected_factors: np.ndarray,
) -> None:
    """
    Test that the CondensationTrailFormation plugin can be initialised
    with custom or default engine contrail factors.

    This test checks that when a custom list of engine contrail factors
    is provided to the plugin, or when no list is provided (using
    defaults), the internal _engine_contrail_factors attribute is
    correctly set as a numpy array with the expected values.

    Args:
        provided_engine_contrail_factors (Optional[List[float]]): List
            of engine contrail factors to initialise the plugin with,
            or None for defaults.
        expected_factors (np.ndarray): The expected numpy array of
            engine contrail factors.
    """
    if provided_engine_contrail_factors is not None:
        plugin = CondensationTrailFormation(
            engine_contrail_factors=provided_engine_contrail_factors
        )
    else:
        plugin = CondensationTrailFormation()
    assert isinstance(plugin._engine_contrail_factors, np.ndarray)
    np.testing.assert_array_equal(plugin._engine_contrail_factors, expected_factors)


@pytest.mark.parametrize("with_iris", [True, False])
@pytest.mark.parametrize(
    "pressure_levels",
    [
        (np.array([100000], dtype=np.float32)),
        (np.array([100000, 90000], dtype=np.float32)),
        (np.array([100000, 90000, 80000], dtype=np.float32)),
    ],
)
def test_pressure_levels_stored(
    pressure_levels: np.ndarray,
    with_iris: bool,
) -> None:
    """
    Test that the CondensationTrailFormation plugin stores the correct
    pressure_levels to the relevant method during processing.

    Args:
        pressure_levels (np.ndarray): Array of pressure levels to use
            for the cubes.
        with_iris (bool): Whether to use iris cubes or numpy arrays.
    """
    shape = (len(pressure_levels), 3, 2)
    temperature_data = np.full(shape, 250, dtype=np.float32)
    humidity_data = np.full(shape, 50, dtype=np.float32)

    plugin = CondensationTrailFormation()
    if with_iris:
        temperature_cube = set_up_variable_cube(
            temperature_data,
            name="air_temperature",
            units="K",
            attributes=LOCAL_MANDATORY_ATTRIBUTES,
            vertical_levels=pressure_levels,
            pressure=True,
        )
        humidity_cube = set_up_variable_cube(
            humidity_data,
            name="relative_humidity",
            units="%",
            attributes=LOCAL_MANDATORY_ATTRIBUTES,
            vertical_levels=pressure_levels,
            pressure=True,
        )
        plugin.process(temperature_cube, humidity_cube)
    else:
        plugin.process_from_arrays(temperature_data, humidity_data, pressure_levels)

    # Check that pressure_levels attribute is set correctly
    np.testing.assert_array_equal(plugin.pressure_levels, pressure_levels)


@pytest.mark.parametrize(
    "pressure_levels, expected_shape, expected_mixing_ratios",
    [
        (
            np.array([100000], dtype=np.float32),
            (3, 1),
            np.array([[4.823306], [5.466414], [6.2702975]], dtype=np.float32),
        ),
        (
            np.array([100000, 90000], dtype=np.float32),
            (3, 2),
            np.array(
                [[4.823306, 4.3409758], [5.466414, 4.919772], [6.2702975, 5.643268]],
                dtype=np.float32,
            ),
        ),
        (
            np.array([100000, 90000, 80000], dtype=np.float32),
            (3, 3),
            np.array(
                [
                    [4.823306, 4.3409758, 3.8586447],
                    [5.466414, 4.919772, 4.373131],
                    [6.2702975, 5.643268, 5.016238],
                ],
                dtype=np.float32,
            ),
        ),
    ],
)
def test_engine_mixing_ratio(
    pressure_levels: np.ndarray,
    expected_shape: Tuple,
    expected_mixing_ratios: np.ndarray,
) -> None:
    """
    Test that the engine mixing ratios are calculated correctly for
    various pressure level arrays.

    This test runs the CondensationTrailFormation._calculate_engine_mixing_ratios method
    and checks that the calculated engine mixing ratios match the
    expected values and shapes.

    Args:
        pressure_levels (np.ndarray): Array of pressure levels to use
            for the cubes.
        expected_shape (tuple): Expected shape of the mixing ratios
            output.
        expected_mixing_ratios (np.ndarray): Expected mixing ratios for
            the given input.
    """
    plugin = CondensationTrailFormation()

    # Check that _calculate_engine_mixing_ratios works after process
    mixing_ratios = plugin._calculate_engine_mixing_ratios(pressure_levels)
    np.testing.assert_array_equal(mixing_ratios, expected_mixing_ratios)


@pytest.mark.parametrize(
    "temperature, relative_humidity, pressure_levels, expected_vapour_pressure",
    [
        # Test with multiple pressure levels
        (
            np.array([250.0, 260.0, 270.0], dtype=np.float32),
            np.array([0.5, 0.6, 0.7], dtype=np.float32),
            np.array([100000, 90000, 80000], dtype=np.float32),
            calculate_svp_in_air(
                np.array([250.0, 260.0, 270.0], dtype=np.float32),
                np.array([100000, 90000, 80000], dtype=np.float32),
            )
            * np.array([0.5, 0.6, 0.7], dtype=np.float32),
        ),
        # Test with single pressure level
        (
            np.array([280.0], dtype=np.float32),
            np.array([0.8], dtype=np.float32),
            np.array([95000], dtype=np.float32),
            calculate_svp_in_air(
                np.array([280.0], dtype=np.float32),
                np.array([95000], dtype=np.float32),
            )
            * np.array([0.8], dtype=np.float32),
        ),
    ],
)
def test_find_local_vapour_pressure(
    temperature: np.ndarray,
    relative_humidity: np.ndarray,
    pressure_levels: np.ndarray,
    expected_vapour_pressure: np.ndarray,
) -> None:
    """
    Test that _find_local_vapour_pressure returns the expected local vapour pressure values.

    This test sets the temperature and relative_humidity attributes on the plugin,
    calls _find_local_vapour_pressure, and checks the output against expected values.

    Args:
        temperature (np.ndarray): Array of temperature values (K).
        relative_humidity (np.ndarray): Array of relative humidity values (fraction).
        pressure_levels (np.ndarray): Array of pressure levels (Pa).
        expected_vapour_pressure (np.ndarray): Expected vapour pressure output (Pa).
    """
    plugin = CondensationTrailFormation()
    plugin.temperature = temperature
    plugin.relative_humidity = relative_humidity
    result = plugin._find_local_vapour_pressure(pressure_levels)
    np.testing.assert_allclose(result, expected_vapour_pressure)


@pytest.mark.parametrize(
    "local_vapour_pressure, engine_mixing_ratio, temperature, critical_intercept, critical_temperature, svp_ice, forms_contrail, is_persistent",
    [
        (1e3, 0, 2e2, 0, 1e3, 0, True, True),
        (1e3, 0, 3e2, 0, 1e3, 0, True, False),
        (1e3, 0, 3e2, 0, 1e3, 1e5, True, False),
        (1e3, 0, 2e2, 0, 1e3, 1e5, True, False),
        (1e3, 0, 3e2, 0, 0, 0, False, False),
        (0, 1, 2e2, 0, 1000, -1, False, False),
        (0, 1, 2e2, 0, 0, -1, False, False),
        (0, 1, 2e2, 0, 0, 1e5, False, False),
        (0, 1, 2e2, 0, 0, 1e5, False, False),
        (0, 1, 3e2, 0, 0, -1, False, False),
        (1000, 0, 3e2, 0, 0, 0, False, False),
        (1000, 0, 2e2, 0, 0, 1e5, False, False),
        (0, 1, 3e2, 0, 1000, -1, False, False),
        (0, 1, 3e2, 0, 1000, 1e5, False, False),
        (0, 1, 2e2, 0, 0, -1, False, False),
        (1000, 0, 2e2, 0, 0, 0, False, False),
    ],
)
def test_calculate_contrail_persistency(
    local_vapour_pressure,
    engine_mixing_ratio,
    temperature,
    critical_intercept,
    critical_temperature,
    svp_ice,
    forms_contrail,
    is_persistent,
) -> None:
    """
    Test that the contrail persistency calculation returns the expected pair of boolean arrays.

    There are 16 sets of (mostly unphysical) input parameters to this test, which correspond to
    the 2^4=16 unique combinations of the four conditions required for contrail formation.

    In the context of this test, the elements within a given output array are identical, i.e.
    all 'True' or all 'False'. However, the two arrays may differ.

    Args:
        local_vapour_pressure (float): Local vapour pressure, calculated with respect to water (Pa).
        engine_mixing_ratio (float): Engine mixing ratio (Pa/K).
        temperature (float): Ambient air temperature (K).
        critical_intercept (float): Critical intercept threshold (Pa).
        critical_temperature (float): Critical temperature threshold (K).
        svp_ice (float): Saturated vapour pressure, calcualted with respect to ice (Pa).
        forms_contrails (bool): True if any contrail will form.
        is_persistent (bool): True only if a persistent contrail will form.
    """
    # plugin output arrays will have expected shape: (3, 2, 1, 4)
    contrail_factors = np.array([3e-5, 3.4e-9, 3.9e-9])
    pressure_levels = np.array([1e4, 1e3])
    temperature = np.full((1, 4), temperature)
    temperature_on_pressure_levels = np.broadcast_to(
        temperature, pressure_levels.shape + temperature.shape
    )
    plugin = CondensationTrailFormation(contrail_factors)

    # broadcast test inputs to plugin arrays
    plugin.local_vapour_pressure = np.broadcast_to(
        local_vapour_pressure, temperature_on_pressure_levels.shape
    )
    plugin.engine_mixing_ratios = np.broadcast_to(
        engine_mixing_ratio, contrail_factors.shape + pressure_levels.shape
    )
    plugin.temperature = temperature_on_pressure_levels
    plugin.critical_intercepts = np.broadcast_to(
        critical_intercept, contrail_factors.shape + pressure_levels.shape
    )
    plugin.critical_temperatures = np.broadcast_to(
        critical_temperature,
        contrail_factors.shape + temperature_on_pressure_levels.shape,
    )
    svp_ice_on_pressure_levels = np.broadcast_to(
        svp_ice, temperature_on_pressure_levels.shape
    )

    persistent_result, nonpersistent_result = plugin._calculate_contrail_persistency(
        svp_ice_on_pressure_levels
    )

    nonpersistent_expected = np.full(
        plugin.critical_temperatures.shape,
        forms_contrail and not is_persistent,
        dtype=bool,
    )
    persistent_expected = np.full(
        plugin.critical_temperatures.shape, forms_contrail and is_persistent, dtype=bool
    )

    np.testing.assert_array_equal(
        nonpersistent_result, nonpersistent_expected, strict=True
    )
    np.testing.assert_array_equal(persistent_result, persistent_expected, strict=True)
