# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the ExtractLevel plugin"""
from collections.abc import Iterable

import numpy as np
import pytest
from iris.cube import Cube

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_extraction import ExtractLevel

LOCAL_MANDATORY_ATTRIBUTES = {
    "title": "unit test data",
    "source": "unit test",
    "institution": "somewhere",
}
pytest.importorskip("stratify")


@pytest.fixture
def temperature_on_pressure_levels() -> Cube:
    """Set up a r, p, y, x cube of temperature of atmosphere on pressure levels"""
    temperatures = np.array([300, 286, 280, 274, 267, 262, 257, 245], dtype=np.float32)
    data = np.broadcast_to(
        temperatures.reshape((1, len(temperatures), 1, 1)), (2, len(temperatures), 3, 2)
    )
    t_cube = set_up_variable_cube(
        data,
        pressure=True,
        height_levels=np.arange(100000, 29999, -10000),
        name="temperature_on_pressure_levels",
        units="K",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return t_cube


@pytest.fixture
def temperature_on_height_levels() -> Cube:
    """Set up a r, h, y, x cube of temperature of atmosphere on height levels"""
    temperatures = np.array([300, 286, 280, 274, 267, 262, 257, 245], dtype=np.float32)
    data = np.broadcast_to(
        temperatures.reshape((1, len(temperatures), 1, 1)), (2, len(temperatures), 3, 2)
    )
    t_cube = set_up_variable_cube(
        data,
        height_levels=np.arange(0, 8000, 1000),
        name="temperature_on_height_levels",
        units="K",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return t_cube


def metadata_check(cube_slice: Cube, value: float, units: str, coordinate: str):
    """Checks the cube produced by plugin has the expected metadata."""
    if coordinate == "pressure":
        assert cube_slice.long_name == f"pressure_of_atmosphere_at_{value}{units}"
        assert cube_slice.units == "Pa"
    else:
        assert cube_slice.long_name == f"height_at_{value}{units}"
        assert cube_slice.units == "m"
    assert cube_slice.attributes == {
        "title": "unit test data",
        "source": "unit test",
        "institution": "somewhere",
    }


def cube_shape_check_with_realizations(cube_slice):
    """Checks cube coordinates and dimensions when two realizations are present"""
    coord_names = [coord.name() for coord in cube_slice.coords()]
    assert coord_names == [
        "realization",
        "latitude",
        "longitude",
        "forecast_period",
        "forecast_reference_time",
        "time",
    ]
    assert cube_slice.shape == (2, 3, 2)


def cube_shape_check_without_realizations(cube_slice):
    """Checks cube coordinates and dimensions when realization is a scalar coord"""
    coord_names = [coord.name() for coord in cube_slice.coords()]
    assert coord_names == [
        "latitude",
        "longitude",
        "forecast_period",
        "forecast_reference_time",
        "realization",
        "time",
    ]
    assert cube_slice.shape == (3, 2)


@pytest.mark.parametrize("coordinate", ("pressure", "height"))
@pytest.mark.parametrize("least_significant_digit", (0, None))
@pytest.mark.parametrize("reverse_coordinate", (False, True))
@pytest.mark.parametrize(
    "special_value", (None, np.nan, True, np.inf, (np.nan, np.nan))
)
@pytest.mark.parametrize("with_realization", (True, False))
@pytest.mark.parametrize(
    "temperature,expected_index",
    (
        (280, 2),  # Exactly matches a value
        (277, 2.5),  # Half way between values
        (301, 0),  # Temperature above max snaps to at max
        (244, 7),  # Temperature below min snaps to at min
    ),
)
def test_basic(
    temperature,
    request,
    expected_index,
    with_realization,
    special_value,
    reverse_coordinate,
    least_significant_digit,
    coordinate,
):
    """Tests the ExtractLevel plugin with values for temperature and
    temperature on levels to check for expected result.
    Tests behaviour when temperature and/or pressure/height increase or decrease along
    the pressure/height axis.
    Tests behaviour with different special values in the temperature data.
    Tests behaviour with and without a realization coordinate.
    Tests behaviour when extracting a pressure level or height level.
    Also checks the metadata of the output cube"""
    if coordinate == "height":
        temperature_on_levels = request.getfixturevalue("temperature_on_height_levels")
    else:
        temperature_on_levels = request.getfixturevalue(
            "temperature_on_pressure_levels"
        )
    special_value_index = 0
    positive_correlation = True
    if reverse_coordinate:
        # Flip the pressure coordinate for this test. We also swap which end the
        # special value goes, so we can test _one_way_fill in both modes.
        temperature_on_levels.coord(coordinate).points = temperature_on_levels.coord(
            coordinate
        ).points[::-1]
        special_value_index = -1
        positive_correlation = False

    if coordinate == "height":
        # height has the opposite correlation to pressure
        positive_correlation = not positive_correlation
    expected = np.interp(
        expected_index,
        range(len(temperature_on_levels.coord(coordinate).points)),
        temperature_on_levels.coord(coordinate).points,
    )

    expected_data = np.full_like(temperature_on_levels.data[:, 0, ...], expected)

    if special_value is True:
        # This is a proxy for setting a mask=True entry
        temperature_on_levels.data = np.ma.MaskedArray(
            temperature_on_levels.data, mask=False
        )
        temperature_on_levels.data.mask[0, special_value_index, 0, 0] = special_value
    elif special_value is None:
        pass
    else:
        temperature_on_levels.data = temperature_on_levels.data.copy()
        if isinstance(special_value, Iterable):
            # This catches the test case where two consecutive special values are to be used
            if special_value_index < 0:
                temperature_on_levels.data[0, -2:, 0, 0] = special_value
            else:
                temperature_on_levels.data[0, 0:2, 0, 0] = special_value
        else:
            temperature_on_levels.data[0, special_value_index, 0, 0] = special_value

    if not with_realization:
        temperature_on_levels = temperature_on_levels[0]
        expected_data = expected_data[0]

    if least_significant_digit:
        temperature_on_levels.attributes[
            "least_significant_digit"
        ] = least_significant_digit

    result = ExtractLevel(
        value_of_level=temperature, positive_correlation=positive_correlation
    )(temperature_on_levels)

    assert not np.ma.is_masked(result.data)
    np.testing.assert_array_almost_equal(result.data, expected_data)
    metadata_check(result, temperature, temperature_on_levels.units, coordinate)
    if with_realization:
        cube_shape_check_with_realizations(result)
    else:
        cube_shape_check_without_realizations(result)


@pytest.mark.parametrize(
    "index, expected",
    (
        (0, 30000),
        (1, 30000),
        (2, 80000),
        (3, 100000),
        (4, 100000),
        (5, 100000),
        (6, 100000),
    ),
)
@pytest.mark.parametrize("special_value", (np.nan, True, np.inf))
def test_only_one_point(
    temperature_on_pressure_levels, index, expected, special_value,
):
    """Tests the ExtractLevel plugin with the unusual case that only one layer has
    a valid value.
    """
    temperature_on_pressure_levels = temperature_on_pressure_levels[0]

    if special_value is True:
        # This is a proxy for setting a mask=True entry
        temperature_on_pressure_levels.data = np.ma.MaskedArray(
            temperature_on_pressure_levels.data, mask=False
        )
        temperature_on_pressure_levels.data.mask[:index, 0, 0] = special_value
        temperature_on_pressure_levels.data.mask[index + 1 :, 0, 0] = special_value
    else:
        temperature_on_pressure_levels.data = temperature_on_pressure_levels.data.copy()
        temperature_on_pressure_levels.data[:index, 0, 0] = special_value
        temperature_on_pressure_levels.data[index + 1 :, 0, 0] = special_value

    expected_data = np.full_like(temperature_on_pressure_levels.data[0, ...], 80000)
    expected_data[0, 0] = expected

    result = ExtractLevel(value_of_level=280, positive_correlation=True)(
        temperature_on_pressure_levels
    )
    assert not np.ma.is_masked(result.data)
    np.testing.assert_array_almost_equal(result.data, expected_data)


def test_both_pressure_and_height_error(temperature_on_height_levels):
    """Tests an error is raised if both pressure and height coordinates are present
    on the input cube"""
    temperature_on_height_levels.coord("realization").rename("pressure")
    with pytest.raises(
        NotImplementedError,
        match="Input Cube has both a pressure and height coordinate.",
    ):
        ExtractLevel(value_of_level=277, positive_correlation=True)(
            temperature_on_height_levels
        )
