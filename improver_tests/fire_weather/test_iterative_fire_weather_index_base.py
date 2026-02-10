# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

import warnings
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pytest
from iris.cube import Cube

from improver.fire_weather import IterativeFireWeatherIndexBase
from improver_tests.fire_weather import make_input_cubes


class ConcreteIterativeFireWeatherIndex(IterativeFireWeatherIndexBase):
    """Concrete implementation of IterativeFireWeatherIndexBase for testing."""

    INPUT_CUBE_NAMES = ["air_temperature", "relative_humidity", "iterative_cube"]
    REFERENCE_CUBE_NAME = "air_temperature"
    OUTPUT_CUBE_NAME = "iterative_cube"

    _REQUIRED_UNITS = IterativeFireWeatherIndexBase._REQUIRED_UNITS | {
        OUTPUT_CUBE_NAME: 1
    }

    STARTING_VALUE = 25
    LAG_TIME = 10

    # Type hints for dynamically created attributes
    temperature: Cube
    relative_humidity: Cube

    def _calculate(self) -> np.ndarray:
        """Simple test calculation: sum temperature and relative humidity."""
        return self.temperature.data + self.relative_humidity.data


plugin = ConcreteIterativeFireWeatherIndex()


@pytest.fixture
def input_cubes_no_start_date() -> tuple[Cube]:
    """Input cubes with start_date missing from attributes on output cube."""
    return make_input_cubes(
        [
            ("air_temperature", 20.0, "Celsius", False, {}),
            ("relative_humidity", 50.0, "1", False, {}),
            ("iterative_cube", 50.0, "1", False, {}),
        ],
        shape=(5, 5),
    )


@pytest.fixture
def initialisation_input_cubes() -> tuple[Cube]:
    """Fixture of input_cubes with no output_cube."""
    return make_input_cubes(
        [
            ("air_temperature", 20.0, "Celsius", False, {}),
            ("relative_humidity", 50.0, "1", False, {}),
        ],
        shape=(5, 5),
    )


def test_raise_value_error_if_state_no_start_date_attribute(
    input_cubes_no_start_date: tuple[Cube],
) -> None:
    """Confirm ValueError raised if no start_date attribute in output cube."""
    with pytest.raises(ValueError, match=r"no start_date attribute"):
        plugin.process(input_cubes_no_start_date)


def test_warning_for_start_dates_inside_lag_time() -> None:
    """When start_date + 9 days runtime < LAG_TIME so warning is created."""
    under_lag_time = str(datetime.now() - timedelta(days=9))
    cube_args = [
        ("air_temperature", 20.0, "Celsius", False, {}),
        ("relative_humidity", 50.0, "1", False, {}),
        ("iterative_cube", 50.0, "1", False, {"start_date": under_lag_time}),
    ]
    cubes = make_input_cubes(cube_args, shape=(5, 5))

    msg = r"iterative_cube is 9 days in to it's initialisation"
    with pytest.warns(UserWarning, match=msg):
        plugin.process(cubes)


def test_no_warning_for_start_dates_outside_lag_time(
    recwarn: list[warnings.WarningMessage],
) -> None:
    """When start_date + 11 days runtime > LAG_TIME so no warning is created."""
    over_lag_time = str(datetime.now() - timedelta(days=11))
    cube_args = [
        ("air_temperature", 20.0, "Celsius", False, {}),
        ("relative_humidity", 50.0, "1", False, {}),
        ("iterative_cube", 50.0, "1", False, {"start_date": over_lag_time}),
    ]
    cubes = make_input_cubes(cube_args, shape=(5, 5))
    plugin.process(cubes)

    assert len(recwarn) == 0


def test_initialise_true_leads_to_user_warning(
    initialisation_input_cubes: tuple[Cube],
) -> None:
    """When initialise=True start_date=now, so runtime < LAG_TIME and warning created."""
    msg = r"iterative_cube is 0 days in to it's initialisation"
    with pytest.warns(UserWarning, match=msg):
        plugin.process(initialisation_input_cubes, initialise=True)


def test_raise_value_error_if_output_cube_present_for_initialisation(
    input_cubes_no_start_date: tuple[Cube],
) -> None:
    """Confirm ValueError raised if output cube given during initialisation."""
    with pytest.raises(
        ValueError, match=r"Unexpected output cube .* when attempting init"
    ):
        plugin.process(input_cubes_no_start_date, initialise=True)


def test_reference_cube_not_found(initialisation_input_cubes):
    """Raise ValueErrir if reference cube not found during initialisation."""
    patch_args = (
        ConcreteIterativeFireWeatherIndex,
        "REFERENCE_CUBE_NAME",
        "test_missing",
    )
    with patch.object(*patch_args):
        msg = r"Reference cube 'test_missing' not found during init"
        with pytest.raises(ValueError, match=msg):
            plugin.process(initialisation_input_cubes, initialise=True)


def test_reference_cube_has_start_date_attribute() -> None:
    """Test ValueError raised if start_date set on reference_cube."""
    dt_attributes = {"start_date": str(datetime(2025, 1, 10, 13, 14, 31))}
    cube_args = [
        ("air_temperature", 20.0, "Celsius", False, dt_attributes),
        ("relative_humidity", 50.0, "1", False, {}),
    ]
    cubes = make_input_cubes(cube_args, shape=(5, 5))

    msg = r"Unexpected start_date in reference_cube attributes"
    with pytest.raises(ValueError, match=msg):
        plugin.process(cubes, initialise=True)


# Test utils
# Write tests for make baseline cube and move to utils or sim


# Test child classes
# Write test with initialisation set for each of the 3 states
# Write test for initialisation values in some sense
