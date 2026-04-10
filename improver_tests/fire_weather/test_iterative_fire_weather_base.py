# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

import warnings
from datetime import timedelta
from unittest.mock import patch

import numpy as np
import pytest
from dateutil.parser import parse
from iris.cube import Cube

from improver.fire_weather import IterativeFireWeatherBase
from improver_tests.fire_weather import (
    DEFAULT_START_DATE,
    DEFAULT_TIME,
    INPUT_ATTRIBUTES,
    make_cube,
    make_input_cubes,
)

LAG_TIME = 10


class ConcreteIterativeFireWeather(IterativeFireWeatherBase):
    """Concrete implementation of IterativeFireWeatherBase for testing."""

    METADATA_SOURCE_CUBE = "iterative_cube"
    INPUT_CUBE_NAMES = [
        "air_temperature",
        "lwe_thickness_of_precipitation_amount",
        "iterative_cube",
    ]
    REFERENCE_CUBE_NAME = "air_temperature"
    OUTPUT_CUBE_NAME = "iterative_cube"

    _REQUIRED_UNITS = IterativeFireWeatherBase._REQUIRED_UNITS | {OUTPUT_CUBE_NAME: 1}

    STARTING_VALUE = 25
    LAG_TIME = LAG_TIME

    # Type hints for dynamically created attributes
    temperature: Cube
    precipitation: Cube

    def _calculate(self) -> np.ndarray:
        """Simple test calculation: sum temperature and relative humidity."""
        return self.temperature.data + self.precipitation.data


plugin = ConcreteIterativeFireWeather()


@pytest.fixture
def input_cubes_no_metadata() -> tuple[Cube]:
    """Input cubes with metadata missing from output cube."""
    return make_input_cubes(
        [
            ("air_temperature", 20.0, "Celsius", False, {}),
            ("lwe_thickness_of_precipitation_amount", 50.0, "mm", False, {}),
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
            ("lwe_thickness_of_precipitation_amount", 50.0, "mm", False, {}),
        ],
        shape=(5, 5),
    )


@pytest.mark.parametrize(
    "attributes",
    (
        {"iteration_start_date": DEFAULT_START_DATE, "iteration_count": 55},
        {"iteration_start_date": DEFAULT_START_DATE, "analysis_ready": True},
        {"iteration_count": 55, "analysis_ready": True},
        {},
    ),
)
def test_raise_value_error_if_metadata_incomplete_on_output_cube(attributes) -> None:
    """Confirm ValueError raised if missing metadata in output cube."""
    with pytest.raises(ValueError, match=r"missing metadata attributes"):
        plugin._record_lag_time_state(
            make_cube(np.full((5, 5), 50.0), "iterative_cube", "1", False, attributes),
        )


def test_warning_for_metadata_inside_lag_time() -> None:
    """When iteration_count is 9 runtime < LAG_TIME so a warning is created."""
    iteration_count = 9
    under_lag_time = str(DEFAULT_TIME - timedelta(days=iteration_count))
    attributes = {
        "iteration_start_date": under_lag_time,
        "analysis_ready": 0,
        "iteration_count": iteration_count,
    }
    cube_args = [
        ("air_temperature", 20.0, "Celsius", False, {}),
        ("lwe_thickness_of_precipitation_amount", 50.0, "mm", False, {}),
        ("iterative_cube", 50.0, "1", False, attributes),
    ]
    cubes = make_input_cubes(cube_args, shape=(5, 5))

    msg = r"iterative_cube is 9 iterations in to its spin-up period"
    with pytest.warns(UserWarning, match=msg):
        plugin.process(cubes)

    cube = plugin.process(cubes)

    assert cube.attributes["iteration_start_date"] == under_lag_time
    assert cube.attributes["iteration_count"] == iteration_count + 1
    assert cube.attributes["analysis_ready"] == "False"


def test_no_warning_for_metadata_outside_lag_time(
    recwarn: list[warnings.WarningMessage],
) -> None:
    """When iteration_count is 11 runtime > LAG_TIME so no warning created."""
    iteration_count = 11
    over_lag_time = str(DEFAULT_TIME - timedelta(days=iteration_count))
    attributes = {
        "iteration_start_date": over_lag_time,
        "analysis_ready": 1,
        "iteration_count": iteration_count,
    }
    cube_args = [
        ("air_temperature", 20.0, "Celsius", False, {}),
        ("lwe_thickness_of_precipitation_amount", 50.0, "mm", False, {}),
        ("iterative_cube", 50.0, "1", False, attributes),
    ]
    cubes = make_input_cubes(cube_args, shape=(5, 5))
    cube = plugin.process(cubes)

    assert len(recwarn) == 0
    assert cube.attributes["iteration_start_date"] == over_lag_time
    assert cube.attributes["iteration_count"] == iteration_count + 1
    assert cube.attributes["analysis_ready"] == "True"


def test_anaylsis_ready_marked_true_when_iteration_count_passes_lag_time() -> None:
    """When iteration_count equals LAG_TIME analysis_ready changes from False to True."""
    iteration_count = LAG_TIME
    equal_to_lag_time = str(DEFAULT_TIME - timedelta(days=iteration_count))
    attributes = {
        "iteration_start_date": equal_to_lag_time,
        "analysis_ready": 0,
        "iteration_count": iteration_count,
    }
    cube_args = [
        ("air_temperature", 20.0, "Celsius", False, {}),
        ("lwe_thickness_of_precipitation_amount", 50.0, "mm", False, {}),
        ("iterative_cube", 50.0, "1", False, attributes),
    ]
    cubes = make_input_cubes(cube_args, shape=(5, 5))
    cube = plugin.process(cubes)

    assert cube.attributes["iteration_start_date"] == equal_to_lag_time
    assert cube.attributes["iteration_count"] == iteration_count + 1
    assert cube.attributes["analysis_ready"] == "True"


def test_initialise_true_leads_to_user_warning(
    initialisation_input_cubes: tuple[Cube],
) -> None:
    """When initialise=True then iteration_count < LAG_TIME so a warning is created."""
    msg = r"iterative_cube is 0 iterations in to its spin-up period"
    with pytest.warns(UserWarning, match=msg):
        cube = plugin.process(initialisation_input_cubes, initialise=True)

    iteration_start_date = parse(cube.attributes["iteration_start_date"])
    diff = abs(iteration_start_date - DEFAULT_TIME).total_seconds()
    assert diff < 0.1, f"Difference is {diff}s"


def test_raise_value_error_if_output_cube_present_for_initialisation(
    input_cubes_no_metadata: tuple[Cube],
) -> None:
    """Confirm ValueError raised if output cube given during initialisation."""
    with pytest.raises(
        ValueError, match=r"Unexpected output cube .* when attempting init"
    ):
        plugin.process(input_cubes_no_metadata, initialise=True)


def test_reference_cube_not_found(initialisation_input_cubes):
    """Raise ValueError if reference cube not found during initialisation."""
    patch_args = (
        ConcreteIterativeFireWeather,
        "REFERENCE_CUBE_NAME",
        "test_missing",
    )
    with patch.object(*patch_args):
        msg = r"Reference cube 'test_missing' not found during init"
        with pytest.raises(ValueError, match=msg):
            plugin.process(initialisation_input_cubes, initialise=True)


def test_reference_cube_has_metadata_attribute() -> None:
    """Test ValueError raised if metadata set on reference_cube."""
    cube_args = [
        ("air_temperature", 20.0, "Celsius", False, INPUT_ATTRIBUTES),
        ("lwe_thickness_of_precipitation_amount", 50.0, "mm", False, {}),
    ]
    cubes = make_input_cubes(cube_args, shape=(5, 5))

    msg = r"Unexpected metadata on reference_cube"
    with pytest.raises(ValueError, match=msg):
        plugin.process(cubes, initialise=True)


def test_process_unpacked_cubes() -> None:
    """
    Verify that the plugin runs successfully when given an unpacked list of
    cubes in its arguments.
    """
    cube_args = [
        ("air_temperature", 20.0, "Celsius", 0, INPUT_ATTRIBUTES),
        ("lwe_thickness_of_precipitation_amount", 50.0, "mm", 0, INPUT_ATTRIBUTES),
        ("iterative_cube", 50.0, "1", 0, INPUT_ATTRIBUTES),
    ]
    cubes = make_input_cubes(cube_args, shape=(5, 5))
    result = plugin.process(*cubes)
    assert isinstance(result, Cube)


def test_process_unpacked_cubes_and_kwargs() -> None:
    """
    Verify that the plugin runs successfully when given an unpacked list of
    cubes in its arguments, in addition to keyword arguments.
    """
    cube_args = [
        ("air_temperature", 20.0, "Celsius", 0, INPUT_ATTRIBUTES),
        ("lwe_thickness_of_precipitation_amount", 50.0, "mm", 0, INPUT_ATTRIBUTES),
        ("iterative_cube", 50.0, "1", 0, INPUT_ATTRIBUTES),
    ]
    cubes = make_input_cubes(cube_args, shape=(5, 5))
    result = plugin.process(*cubes, month=1, initialise=False)
    assert isinstance(result, Cube)
