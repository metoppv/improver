# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the CondensationTrailFormation plugin"""

from typing import List, Optional, Tuple

import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import Cube

from improver.psychrometric_calculations.condensation_trails import (
    CondensationTrailFormation,
)
from improver.psychrometric_calculations.psychrometric_calculations import (
    calculate_svp_in_air,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import get_dim_coord_names

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
                phase="water",
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
                phase="water",
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
            np.array([[[210.8735, 212.7484, 219.25]]], dtype=np.float32),
            np.array([[-101.3616]], dtype=np.float32),
        ),
        # one contrail factor, multiple pressure levels, 1D relative humidity
        (
            [3e-5],
            np.array([1e5, 1e4], dtype=np.float32),
            np.array([0, 0.5, 1], dtype=np.float32),
            np.array(
                [
                    [
                        [232.65, 235.0091, 243.25],
                        [210.8735, 212.7484, 219.25],
                    ]
                ],
                dtype=np.float32,
            ),
            np.array([[-1126.4983, -101.3616]], dtype=np.float32),
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
                            [232.65, 233.4512, 234.4274],
                            [235.6830, 237.5116, 243.25],
                        ],
                        [
                            [210.8735, 211.5130, 212.2870],
                            [213.2819, 214.7282, 219.25],
                        ],
                    ]
                ],
                dtype=np.float32,
            ),
            np.array([[-1126.498, -101.3616]], dtype=np.float32),
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
                            [232.65, 233.4512, 234.4274],
                            [235.6830, 237.5116, 243.25],
                        ],
                        [
                            [210.8735, 211.5130, 212.2870],
                            [213.2819, 214.7282, 219.25],
                        ],
                    ],
                    [
                        [
                            [233.95, 234.7235, 235.7122],
                            [236.9838, 238.8360, 244.65],
                        ],
                        [
                            [211.9674, 212.6149, 213.3986],
                            [214.4059, 215.8706, 220.45],
                        ],
                    ],
                    [
                        [
                            [235.45, 236.2676, 237.2714],
                            [238.5627, 240.4437, 246.35],
                        ],
                        [
                            [213.1517, 213.8081, 214.6024],
                            [215.6233, 217.1077, 221.75],
                        ],
                    ],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [-1126.498, -101.3616],
                    [-1274.799, -116.0054],
                    [-1478.124, -134.0087],
                ],
                dtype=np.float32,
            ),
        ),
    ],
)
def test_calculate_critical_temperatures_and_intercepts(
    engine_contrail_factors: List[float],
    pressure_levels: np.ndarray,
    relative_humidity: np.ndarray,
    expected_critical_temperatures: np.ndarray,
    expected_critical_intercepts: np.ndarray,
):
    """
    Test that _calculate_critical_temperatures_and_intercepts() returns two arrays with the expected shapes and values.

    Args:
        engine_contrail_factors (List[float]) List of contrail factors used to initialise the contrails class (kg/kg/K).
        pressure_levels (np.ndarray): Array of pressure levels (Pa).
        relative_humidity (np.ndarray): Array of relative humidity values (kg/kg).
        expected_critical_temperatures (np.ndarray): Expected critical temperature output on pressure levels. Array axes
            are [contrail factors, pressure levels, relative humidity] (K).
        expected_critical_intercepts (np.ndarray): Expected critical intercept output on pressure levels. Array axes
            are [contrail factors, pressure levels] (Pa).
    """
    plugin = CondensationTrailFormation(engine_contrail_factors)
    plugin.engine_mixing_ratios = plugin._calculate_engine_mixing_ratios(
        pressure_levels
    )
    plugin.relative_humidity = np.broadcast_to(
        relative_humidity, pressure_levels.shape + relative_humidity.shape
    )
    plugin._calculate_critical_temperatures_and_intercepts()

    np.testing.assert_allclose(
        plugin.critical_temperatures,
        expected_critical_temperatures,
        rtol=1e-6,
        strict=True,
        verbose=True,
    )
    np.testing.assert_allclose(
        plugin.critical_intercepts,
        expected_critical_intercepts,
        rtol=1e-6,
        strict=True,
        verbose=True,
    )


@pytest.mark.parametrize(
    "engine_mixing_ratio, critical_intercept, temperature, critical_temperature, local_vapour_pressure, forms_contrail, is_persistent",
    [
        # air temperature = 212 K, increase relative humidity 1 -> 20 -> 40 -> 60 %
        (0.482, -101.36, 212, 210.90, 0.00924, False, False),
        (0.482, -101.36, 212, 211.51, 0.531, False, False),
        (0.482, -101.36, 212, 212.29, 10.9, True, True),
        (0.482, -101.36, 212, 213.28, 117, True, True),
        # increase air temperature from 208 K to 214 K, relative humidity = 10 %,
        (0.482, -101.36, 208, 211.18, 0.0529, True, False),
        (0.482, -101.36, 210, 211.18, 0.0701, True, False),
        (0.482, -101.36, 212, 211.18, 0.0924, False, False),
        (0.482, -101.36, 214, 211.18, 0.121, False, False),
        # arbitrary, unphysical values
        (0, 0, 280, 300, 0.01, True, False),
        (0, 0, 280, 300, 10, True, False),
        (0, 0, 200, 0, 1, False, False),
        (1, 0, 200, 0, 0, False, False),
        (0, 0, 300, 0, 1, False, False),
        (1, 0, 300, 0, 0, False, False),
    ],
)
def test_calculate_contrail_persistency_combinations(
    engine_mixing_ratio: float,
    critical_intercept: float,
    temperature: float,
    critical_temperature: float,
    local_vapour_pressure: float,
    forms_contrail: bool,
    is_persistent: bool,
) -> None:
    """
    Test that the contrail persistency calculation returns the expected pair of boolean arrays
    for each combination of four formation conditions. The first 8 sets of input parameters are physical
    values that could exist in a real system, whereas the remaining sets are arbitrary and unphysical.

    In the first 4 sets, the relative humidity is increased from 1 % to 60 %, whilst the air temperature
    is held constant at 212 K. This causes the local vapour pressure and critical temperature to increase.

    In the next 4 sets, the air temperature is increased from 208 K to 214 K, whilst the relative
    humidity is held constant at 10 %. This causes the local and saturated vapour pressures to increase.

    The remaining unphysical sets check for other combinations of the conditions for contrail formation.

    In the context of this test, the elements within a given output array are identical, i.e.
    all 'True' or all 'False'. However, the two arrays may differ.

    Args:
        engine_mixing_ratio (float): Engine mixing ratio (Pa/K).
        critical_intercept (float): Critical intercept threshold (Pa).
        temperature (float): Ambient air temperature (K).
        critical_temperature (float): Critical temperature threshold (K).
        local_vapour_pressure (float): Local vapour pressure, calculated with respect to water (Pa).
        svp_ice (float): Saturated vapour pressure, calculated with respect to ice (Pa).
        forms_contrails (bool): True if any contrail will form.
        is_persistent (bool): True only if a persistent contrail will form.
    """
    # plugin output arrays will have expected shape: (1, 1, 5, 4)
    contrail_factors = np.array([3e-5])
    pressure_levels = np.array([1e4])
    temperature = np.full((5, 4), temperature)
    temperature_on_pressure_levels = np.broadcast_to(
        temperature, pressure_levels.shape + temperature.shape
    )

    plugin = CondensationTrailFormation(contrail_factors)
    plugin.temperature = temperature_on_pressure_levels

    # construct remaining input arrays, each filled with a specific value
    plugin.local_vapour_pressure = np.full(
        temperature_on_pressure_levels.shape, local_vapour_pressure
    )
    plugin.engine_mixing_ratios = np.full(
        contrail_factors.shape + pressure_levels.shape, engine_mixing_ratio
    )
    plugin.critical_intercepts = np.full(
        contrail_factors.shape + pressure_levels.shape, critical_intercept
    )
    plugin.critical_temperatures = np.full(
        contrail_factors.shape + temperature_on_pressure_levels.shape,
        critical_temperature,
    )
    plugin._calculate_contrail_persistency()

    nonpersistent_expected = np.full(
        plugin.critical_temperatures.shape,
        forms_contrail and not is_persistent,
        dtype=bool,
    )
    persistent_expected = np.full(
        plugin.critical_temperatures.shape, forms_contrail and is_persistent, dtype=bool
    )

    np.testing.assert_array_equal(
        plugin.nonpersistent_contrails, nonpersistent_expected, strict=True
    )
    np.testing.assert_array_equal(
        plugin.persistent_contrails, persistent_expected, strict=True
    )


@pytest.mark.parametrize(
    "latitude_dimension_size, longitude_dimension_size",
    [
        (0, 0),
        (1, 0),
        (0, 1),
        (5, 4),
    ],
)
def test_calculate_contrail_persistency_shapes(
    latitude_dimension_size: int,
    longitude_dimension_size: int,
) -> None:
    """
    Test that the contrail persistency calculation returns the expected pair of boolean arrays when given input arrays
    with differing shapes.

    The output arrays of the persistency calculation always have leading axes of [contrail factors, pressure levels].
    However, the temperature and humidity arrays of the contrails class may contain latitude and longitude axes. If
    these are present, the persistency outputs will have axes [contrail factors, pressure levels, latitude, longitude].

    Args:
        latitude_dimension_size (int): Number of elements in the latitude axis of the air temperature array.
        latitude_dimension_size (int): Number of elements in the longitude axis of the air temperature array.

    """
    contrail_factors = np.array([3e-5, 3.4e-9, 3.9e-9])
    pressure_levels = np.array([1e4, 1e3])

    # (contrail factors, pressure levels)
    cf_p_shape = contrail_factors.shape + pressure_levels.shape

    # (pressure levels, latitude, longitude)
    p_lat_long_shape = pressure_levels.shape
    if latitude_dimension_size > 0:
        p_lat_long_shape += (latitude_dimension_size,)
    if longitude_dimension_size > 0:
        p_lat_long_shape += (longitude_dimension_size,)

    # (contrail factors, pressure levels, latitude, longitude)
    cf_p_lat_long_shape = contrail_factors.shape + p_lat_long_shape

    # values for input arrays that result in persistent contrails
    engine_mixing_ratio = 0.482
    critical_intercept = -101.36
    temperature = 212
    critical_temperature = 213.28
    local_vapour_pressure = 117

    plugin = CondensationTrailFormation(contrail_factors)

    # construct input arrays, each filled with a specific value
    plugin.temperature = np.full(p_lat_long_shape, temperature)
    plugin.local_vapour_pressure = np.full(p_lat_long_shape, local_vapour_pressure)
    plugin.engine_mixing_ratios = np.full(cf_p_shape, engine_mixing_ratio)
    plugin.critical_intercepts = np.full(cf_p_shape, critical_intercept)
    plugin.critical_temperatures = np.full(cf_p_lat_long_shape, critical_temperature)

    plugin._calculate_contrail_persistency()

    nonpersistent_expected = np.full(cf_p_lat_long_shape, False, dtype=bool)
    persistent_expected = np.full(cf_p_lat_long_shape, True, dtype=bool)

    np.testing.assert_array_equal(
        plugin.nonpersistent_contrails, nonpersistent_expected, strict=True
    )
    np.testing.assert_array_equal(
        plugin.persistent_contrails, persistent_expected, strict=True
    )


@pytest.mark.parametrize(
    "nonpersistent_contrails, persistent_contrails, categorical_expected",
    [
        (
            np.array([False, True, False], dtype=bool),
            np.array([False, False, True], dtype=bool),
            np.array([0, 1, 2]),
        ),
        (
            np.array([[False, True], [True, False]], dtype=bool),
            np.array([[False, False], [False, True]], dtype=bool),
            np.array([[0, 1], [1, 2]]),
        ),
        (
            np.array([[[False]], [[True]], [[False]]], dtype=bool),
            np.array([[[False]], [[False]], [[True]]], dtype=bool),
            np.array([[[0]], [[1]], [[2]]]),
        ),
        (
            np.array(
                [
                    [[[False, False, True]], [[True, True, False]]],
                    [[[False, False, True]], [[True, True, False]]],
                ],
                dtype=bool,
            ),
            np.array(
                [
                    [[[False, False, False]], [[False, False, True]]],
                    [[[True, True, False]], [[False, False, False]]],
                ],
                dtype=bool,
            ),
            np.array([[[[0, 0, 1]], [[1, 1, 2]]], [[[2, 2, 1]], [[1, 1, 0]]]]),
        ),
    ],
)
def test_categorical_array_output(
    nonpersistent_contrails: np.ndarray,
    persistent_contrails: np.ndarray,
    categorical_expected: np.ndarray,
) -> None:
    """
    Given two boolean arrays of non-persistent and persistent contrails, check that the expected categorical (integer)
    array can be constructed.

    Args:
        nonpersistent_contrails (np.ndarray): Boolean data of non-persistent contrails.
        persistent_contrails (np.ndarray): Boolean data of persistent contrails.
        categorical_expected (np.ndarray): Categorical (integer) array, where 0 = no contrails, 1 = non-persistent
            contrails and 2 = persistent contrails.
    """

    plugin = CondensationTrailFormation()
    plugin.nonpersistent_contrails = nonpersistent_contrails
    plugin.persistent_contrails = persistent_contrails
    categorical_result = plugin._boolean_to_categorical()
    assert isinstance(categorical_result, np.ndarray)
    np.testing.assert_array_equal(categorical_result, categorical_expected)


@pytest.mark.parametrize(
    "pressure_levels, y_points, x_points",
    [
        (np.array([1e4, 1e3]), np.linspace(0, 1e5, 6), np.linspace(0, 1e5, 8)),
        (np.array([1e4, 1e3]), None, np.linspace(0, 1e5, 8)),
        (np.array([1e4, 1e3]), np.linspace(0, 1e5, 6), None),
        (np.array([1e4, 1e3]), None, None),
    ],
)
def test_categorical_cube_output(
    pressure_levels: np.ndarray,
    y_points: Optional[np.ndarray],
    x_points: Optional[np.ndarray],
) -> None:
    """
    Given pressure levels and optional geospatial points, check that the expected categorical cube of contrail
    formation can be constructed. Data values are integers 0, 1 and 2, where 0 = no contrails, 1 = non-persistent
    contrails and 2 = persistent contrails.

    Args:
        pressure_levels (np.ndarray): Float array of pressure levels (Pa).
        y_points (Optional[np.ndarray]): Float array of latitude grid points (m).
        x_points (Optional[np.ndarray]): Float array of longitude grid points (m).
    """
    # construct template temperature cube
    template_shape = pressure_levels.shape
    temperature_dim_coords = [DimCoord(pressure_levels, var_name="pressure")]
    if y_points is not None:
        template_shape += y_points.shape
        temperature_dim_coords.append(DimCoord(y_points, var_name="latitude"))
    if x_points is not None:
        template_shape += x_points.shape
        temperature_dim_coords.append(DimCoord(x_points, var_name="longitude"))

    temperature_dim_coords_and_dims = [
        (coord, i) for i, coord in enumerate(temperature_dim_coords)
    ]
    temperature_cube = Cube(
        var_name="temperature",
        dim_coords_and_dims=temperature_dim_coords_and_dims,
        data=np.full(template_shape, 200),
    )

    # generate input categorical data
    plugin = CondensationTrailFormation()
    output_shape = plugin._engine_contrail_factors.shape + temperature_cube.shape
    categorical_data = np.zeros(output_shape, dtype=np.int32)
    categorical_data[np.random.randint(0, 3, output_shape)] = 1
    categorical_data[np.random.randint(0, 3, output_shape)] = 2

    # construct output categorical cube
    categorical_cube = plugin._create_contrail_formation_cube(
        categorical_data, temperature_cube
    )

    # type and data check
    assert isinstance(categorical_cube, Cube)
    assert categorical_cube.data.max() <= 2 and categorical_cube.data.min() >= 0
    np.testing.assert_array_equal(
        categorical_data, categorical_cube.data, strict=True, verbose=True
    )

    # coordinate check
    output_coord_names = get_dim_coord_names(categorical_cube)
    assert output_coord_names[0] == "engine_contrail_factor"
    assert output_coord_names[1] == "pressure"
    if y_points is not None:
        assert "latitude" in output_coord_names
    if x_points is not None:
        assert "longitude" in output_coord_names

    # category check
    categories_expected = {"None": 0, "Non-persistent": 1, "Persistent": 2}
    categories_result = dict(
        zip(
            categorical_cube.attributes["contrail_type_meaning"].split(" "),
            categorical_cube.attributes["contrail_type"],
        )
    )
    assert categories_result == categories_expected


@pytest.mark.parametrize(
    "temperature, relative_humidity, pressure_levels, expected_exception",
    [
        # correct inputs
        (np.zeros(2), np.zeros(2), np.zeros(2), None),
        (np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(2), None),
        (np.zeros((2, 3, 4)), np.zeros((2, 3, 4)), np.zeros(2), None),
        # invalid types
        (2, np.zeros(2), np.zeros(2), AttributeError),
        (np.zeros(2), 2, np.zeros(2), AttributeError),
        (np.zeros(2), np.zeros(2), 2, AttributeError),
        # invalid leading axes
        (np.zeros((1, 3, 4)), np.zeros((2, 3, 4)), np.zeros(2), ValueError),
        (np.zeros((2, 3, 4)), np.zeros((1, 3, 4)), np.zeros(2), ValueError),
        (np.zeros((2, 3, 4)), np.zeros((2, 3, 4)), np.zeros(1), ValueError),
        # invalid shapes
        (np.zeros((2, 1)), np.zeros((2, 3)), np.zeros(2), ValueError),
        (np.zeros((2, 3, 1)), np.zeros((2, 3, 4)), np.zeros(2), ValueError),
        (np.zeros((2, 1, 4)), np.zeros((2, 3, 4)), np.zeros(2), ValueError),
        (np.zeros(2), np.zeros(2), np.zeros((2, 3)), ValueError),
        (np.zeros(2), np.zeros(2), np.zeros(()), ValueError),
    ],
)
def test_process_from_arrays_raises(
    temperature: np.ndarray,
    relative_humidity: np.ndarray,
    pressure_levels: np.ndarray,
    expected_exception: Optional[Exception],
) -> None:
    """
    Check that 'process_from_arrays' returns the expected exceptions when given invalid inputs.

    Args:
        temperature (np.ndarray): Float array of ambient air temperature array (K).
        relative_humidity (np.ndarray): Float array of relative humidity (kg/kg).
        pressure_levels (np.ndarray): Float array of pressure levels (Pa).
        expected_exception (Optional[Exception]): The exception returned by 'process_from_arrays'.
    """
    plugin = CondensationTrailFormation()
    if expected_exception:
        with pytest.raises(expected_exception):
            plugin.process_from_arrays(temperature, relative_humidity, pressure_levels)
    else:
        result = plugin.process_from_arrays(
            temperature, relative_humidity, pressure_levels
        )
        assert isinstance(result, np.ndarray)


@pytest.mark.parametrize(
    "cube_names, cube_shapes, coord_names, expected_exception, expected_message",
    [
        # correct inputs
        (
            ("air_temperature", "relative_humidity"),
            ((2, 2), (2, 2)),
            (("pressure", "latitude"), ("pressure", "latitude")),
            None,
            None,
        ),
        # wrong variable name
        (
            ("air_temperature", "surface_air_pressure"),
            ((2, 2), (2, 2)),
            (("pressure", "latitude"), ("pressure", "latitude")),
            ValueError,
            "could not extract",
        ),
        # single cube
        (
            ("air_temperature",),
            ((2,), (2,)),
            (("pressure", "latitude"),),
            ValueError,
            "cubes, got",
        ),
        # unmatching coordinates
        (
            ("air_temperature", "relative_humidity"),
            ((2, 3), (2, 2)),
            (("pressure", "latitude"), ("pressure", "latitude")),
            ValueError,
            "coordinates must match",
        ),
        (
            ("air_temperature", "relative_humidity"),
            ((2, 2), (2, 2)),
            (("pressure", "latitude"), ("pressure", "longitude")),
            ValueError,
            "coordinates must match",
        ),
        # pressure not leading axis
        (
            ("air_temperature", "relative_humidity"),
            ((2, 2), (2, 2)),
            (("latitude", "pressure"), ("latitude", "pressure")),
            ValueError,
            "must be the leading axis",
        ),
    ],
)
def test_process_raises(
    cube_names: Tuple[str],
    cube_shapes: Tuple[int],
    coord_names: Tuple[Tuple[str]],
    expected_exception: Optional[Exception],
    expected_message: Optional[str],
) -> None:
    """
    Check that 'process' returns the expected exceptions when given invalid inputs.

    Args:
        cube_names (Tuple[str]): Names of input cubes.
        cube_shapes (Tuple[int]): Shapes of input cubes.
        coord_names (Tuple[Tuple[str]]): Names of dimensional coordinates for each cube.
        expected_exception (Optional[Exception]): The exception returned by 'process'.
        expected_message (Optional[str]): The message associated with the exception.
    """
    units_dict = {
        "air_temperature": "K",
        "relative_humidity": "kg kg-1",
        "pressure": "Pa",
        "latitude": "m",
        "longitude": "m",
        "surface_air_pressure": "Pa",
    }

    input_cubes = []
    for cube_name, cube_shape, coord_names_for_cube in zip(
        cube_names, cube_shapes, coord_names
    ):
        # create dimensional coordinates
        dim_coords = []
        for coord_name, coord_shape in zip(coord_names_for_cube, cube_shape):
            points = np.linspace(0, 1, coord_shape)
            dim_coords.append(
                DimCoord(points, var_name=coord_name, units=units_dict[coord_name])
            )

        dim_coords_and_dims = [(coord, i) for i, coord in enumerate(dim_coords)]

        # create input cube
        input_cubes.append(
            Cube(
                var_name=cube_name,
                dim_coords_and_dims=dim_coords_and_dims,
                data=np.ones(cube_shape),
                units=units_dict[cube_name],
            )
        )

    # check exceptions are raised
    plugin = CondensationTrailFormation()
    if expected_exception:
        with pytest.raises(expected_exception) as e:
            plugin.process(input_cubes)
        assert expected_message.casefold() in str(e.value).casefold()
    else:
        result = plugin.process(input_cubes)
        assert isinstance(result, Cube)


@pytest.mark.parametrize(
    "engine_contrail_factor, pressure, temperature, relative_humidity, expected_contrail_type",
    [
        # always form
        (3e-5, 5e3, 198, 0, 1),  # -75 C
        (3e-5, 1e4, 208, 0, 1),  # -65 C
        (3e-5, 4e4, 218, 0, 1),  # -55 C
        # might form (increasing relative humidity)
        # -75 C
        (3e-5, 5e3, 198, 0.5, 1),
        (3e-5, 5e3, 198, 0.55, 2),
        (3e-5, 5e3, 198, 0.6, 2),
        # -65 C
        (3e-5, 5e3, 208, 0.65, 0),
        (3e-5, 5e3, 208, 0.7, 2),
        (3e-5, 5e3, 208, 0.75, 2),
        # -45 C
        (3e-5, 4e4, 228, 0.9, 0),
        (3e-5, 4e4, 228, 0.95, 2),
        (3e-5, 4e4, 228, 1, 2),
        # never form
        (3e-5, 5e3, 218, 1, 0),  # -55 C
        (3e-5, 2e4, 228, 1, 0),  # -45 C
        (3e-5, 5e3, 238, 1, 0),  # -35 C
    ],
)
def test_process_values(
    engine_contrail_factor: float,
    pressure: float,
    temperature: float,
    relative_humidity: float,
    expected_contrail_type: int,
) -> None:
    """
    Check that 'process' returns the expected contrail types from temperature
    and relative humidity cube inputs.

    This test covers the entire contrail formation class, from the user entry
    point to the output categorical cube.

    An Appleman diagram has three regions of interest for this test. Contrails
    'always' form (below 0% line of constant relative humidity), 'never' form
    (above 100% line), or 'might' form (between 0% and 100% lines).

    Args:
        engine_contrail_factor (float): Engine contrail factor (kg/kg/K).
        pressure (float): Ambient air pressure (Pa).
        temperature (float): Ambient air temperature (K).
        relative_humidity (float): Relative humidity (kg/kg).
        expected_contrail_type (int): Denotes the type of contrail that may form,
            where 0 = no contrails, 1 = non-persistent contrails and 2 = persistent
            contrails.
    """

    pressure_levels = np.array([pressure])
    temperatures = np.array([temperature])
    relative_humidities = np.array([relative_humidity])

    dim_coords_and_dims = [
        (DimCoord(pressure_levels, var_name="pressure", units="Pa"), 0),
    ]
    temperature_cube = Cube(
        var_name="air_temperature",
        dim_coords_and_dims=dim_coords_and_dims,
        data=temperatures,
        units="K",
    )
    humidity_cube = Cube(
        var_name="relative_humidity",
        dim_coords_and_dims=dim_coords_and_dims,
        data=relative_humidities,
        units="kg kg-1",
    )

    plugin = CondensationTrailFormation([engine_contrail_factor])
    result = plugin.process([temperature_cube, humidity_cube])

    assert isinstance(result, Cube)
    np.testing.assert_array_equal(
        result.data,
        np.array([[expected_contrail_type]], dtype=np.int32),
        strict=True,
        verbose=True,
    )
