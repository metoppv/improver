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
    "engine_contrail_factors, pressure_levels, relative_humidity, expected_critical_temperatures, expected_critical_intercepts",
    [
        # one contrail factor, one pressure level, 1D relative humidity
        (
            [3e-5],
            np.array([1e4], dtype=np.float32),
            np.array([0, 0.5, 1], dtype=np.float32),
            np.array([[[210.87348485, 212.74835668, 219.25]]], dtype=np.float32),
            np.array([[-101.36159382]], dtype=np.float32),
        ),
        # one contrail factor, multiple pressure levels, 1D relative humidity
        (
            [3e-5],
            np.array([1e5, 1e4], dtype=np.float32),
            np.array([0, 0.5, 1], dtype=np.float32),
            np.array(
                [
                    [
                        [232.65, 235.00909317, 243.25],
                        [210.87348485, 212.74835668, 219.25],
                    ]
                ],
                dtype=np.float32,
            ),
            np.array([[-1126.49826644, -101.36159382]], dtype=np.float32),
        ),
        # one contrail factor, multiple pressure levels, 2D relative humidity
        (
            [3e-5],
            np.array([1e5, 1e4], dtype=np.float32),
            np.array([[0, 0.2, 0.4], [0.6, 0.8, 1]], dtype=np.float32),
            np.array(
                [
                    [
                        [
                            [232.65, 233.45120317, 234.42736744],
                            [235.68303131, 237.51159792, 243.25],
                        ],
                        [
                            [210.87348485, 211.51295453, 212.28699057],
                            [213.28191226, 214.72817613, 219.25],
                        ],
                    ]
                ],
                dtype=np.float32,
            ),
            np.array([[-1126.49826644, -101.36159382]], dtype=np.float32),
        ),
        # multiple contrail factors, multiple pressure levels, 2D relative humidity
        (
            [3e-5, 3.4e-5, 3.9e-5],
            np.array([1e5, 1e4], dtype=np.float32),
            np.array([[0, 0.2, 0.4], [0.6, 0.8, 1]], dtype=np.float32),
            np.array(
                [
                    [
                        [
                            [232.65, 233.45120317, 234.42736744],
                            [235.68303131, 237.51159792, 243.25],
                        ],
                        [
                            [210.87348485, 211.51295453, 212.28699057],
                            [213.28191226, 214.72817613, 219.25],
                        ],
                    ],
                    [
                        [
                            [233.95, 234.72353454, 235.71222294],
                            [236.98378723, 238.83600501, 244.65],
                        ],
                        [
                            [211.96734685, 212.61486738, 213.39860275],
                            [214.40593674, 215.87057442, 220.45],
                        ],
                    ],
                    [
                        [
                            [235.45, 236.26764963, 237.27141769],
                            [238.56273465, 240.44372252, 246.35],
                        ],
                        [
                            [213.15171974, 213.80810569, 214.60240506],
                            [215.62327924, 217.10765399, 221.75],
                        ],
                    ],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [-1126.49826644, -101.36159382],
                    [-1274.79858365, -116.00541492],
                    [-1478.1242258, -134.00868055],
                ],
                dtype=np.float32,
            ),
        ),
    ],
)
def test_calculate_critical_temperatures_and_intercepts_values(
    engine_contrail_factors: List[float],
    pressure_levels: np.ndarray,
    relative_humidity: np.ndarray,
    expected_critical_temperatures: np.ndarray,
    expected_critical_intercepts: np.ndarray,
):
    """
    Test that _calculate_critical_temperatures_and_intercepts() returns two arrays with
    the expected shapes and values.

    Args:
        engine_contrail_factors (List[float]) List of contrail factors used to initialise the contrails class (kg/kg/K).
        pressure_levels (np.ndarray): Array of pressure levels (Pa).
        relative_humidity (np.ndarray): Array of relative humidity values (kg/kg).
        expected_critical_temperatures (np.ndarray): Expected critical temperature output on pressure levels. Array axes are [contrail factors, pressure levels, relative humidity] (K).
        expected_critical_intercepts (np.ndarray): Expected critical intercept output on pressure levels. Array axes are [contrail factors, pressure levels] (Pa).
    """
    plugin = CondensationTrailFormation(engine_contrail_factors)
    plugin.engine_mixing_ratios = plugin._calculate_engine_mixing_ratios(
        pressure_levels
    )
    plugin.relative_humidity = np.broadcast_to(
        relative_humidity, pressure_levels.shape + relative_humidity.shape
    )

    critical_temperatures, critical_intercepts = (
        plugin._calculate_critical_temperatures_and_intercepts()
    )

    np.testing.assert_allclose(
        critical_temperatures,
        expected_critical_temperatures,
        rtol=1e-3,
        strict=True,
        verbose=True,
    )
    np.testing.assert_allclose(
        critical_intercepts,
        expected_critical_intercepts,
        rtol=1e-3,
        strict=True,
        verbose=True,
    )


@pytest.mark.parametrize(
    "engine_mixing_ratios, relative_humidity, expected_exception",
    [
        ([1, 1], np.ones((2, 3)), TypeError),
        (np.ones((1, 2)), [1, 1], TypeError),
        (np.ones(1), np.ones((2, 3)), ValueError),
        (np.ones((1, 2)), np.ones(2), ValueError),
        (np.ones((1, 2)), np.ones((3, 2)), ValueError),
    ],
)
def test_calculate_critical_temperatures_and_intercepts_raises_on_bad_input(
    engine_mixing_ratios: np.ndarray | List,
    relative_humidity: np.ndarray | List,
    expected_exception: Exception,
):
    """
    Test that _calculate_critical_temperatures_and_intercepts() raises the appropriate
    exception when called with incorrect input parameters.

    Args:
        engine_mixing_ratios (np.ndarray | List): Engine mixing ratios on pressure levels. Engine contrail factors are the leading axis (Pa/K).
        relative_humidity (np.ndarray | List): Relative humidity values on pressure levels. Pressure levels are the leading axis (kg/kg).
        expected_exception (exception | None): The exception that should be raised by the function.
    """
    plugin = CondensationTrailFormation([3e-5])
    plugin.engine_mixing_ratios = engine_mixing_ratios
    plugin.relative_humidity = relative_humidity
    with pytest.raises(expected_exception):
        plugin._calculate_critical_temperatures_and_intercepts()
