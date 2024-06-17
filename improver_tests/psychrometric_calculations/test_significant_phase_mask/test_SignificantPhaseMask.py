# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for psychrometric_calculations SignificantPhaseMask plugin."""

import iris
import numpy as np
import pytest

from improver.psychrometric_calculations.significant_phase_mask import (
    SignificantPhaseMask,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


def test_basic():
    """Test that the __init__ method configures the plugin as expected."""
    plugin = SignificantPhaseMask()
    assert np.isclose(plugin.lower_threshold, 0.01)
    assert np.isclose(plugin.upper_threshold, 0.99)
    assert isinstance(plugin.phase_operator, dict)
    assert all(callable(v) for v in plugin.phase_operator.values())


@pytest.mark.parametrize(
    ("snow_fraction", "phase", "expected"),
    (
        (0, "snow", 0),
        (0.5, "snow", 0),
        (1, "snow", 1),
        (0, "sleet", 0),
        (0.5, "sleet", 1),
        (1, "sleet", 0),
        (0, "rain", 1),
        (0.5, "rain", 0),
        (1, "rain", 0),
        (SignificantPhaseMask().lower_threshold, "rain", 1),
        (SignificantPhaseMask().lower_threshold, "sleet", 0),
        (SignificantPhaseMask().upper_threshold, "snow", 1),
        (SignificantPhaseMask().upper_threshold, "sleet", 0),
    ),
)
def test_values(snow_fraction, phase, expected):
    """Test specific values give expected results, including all meta-data"""
    input_cube = set_up_variable_cube(
        np.full((2, 2), fill_value=snow_fraction, dtype=np.float32),
        name="snow_fraction",
        units="1",
        standard_grid_metadata="uk_ens",
    )
    result = SignificantPhaseMask()(input_cube, phase)
    assert isinstance(result, iris.cube.Cube)
    assert result.name() == f"{phase}_mask"
    assert str(result.units) == "1"
    assert result.dtype == np.int8
    assert (result.data == expected).all()


@pytest.mark.parametrize("model_id_attr", (None, "mosg__model_configuration"))
def test_model_id_attr(model_id_attr):
    """Test attribute handling"""
    input_cube = set_up_variable_cube(
        np.full((2, 2), fill_value=0.1, dtype=np.float32),
        name="snow_fraction",
        units="1",
        standard_grid_metadata="uk_ens",
    )
    expected_attributes = {
        "source": "Unit test",
        "institution": "Met Office",
        "title": "Post-Processed IMPROVER unit test",
    }
    input_cube.attributes.update(expected_attributes)
    if model_id_attr:
        expected_attributes[model_id_attr] = input_cube.attributes[model_id_attr]
    result = SignificantPhaseMask(model_id_attr=model_id_attr)(input_cube, "snow")
    assert result.attributes == expected_attributes


def test_masked_values():
    """Test specific values give expected results"""
    data = np.zeros((2, 2), dtype=np.float32)
    data = np.ma.masked_array(data, [[True, False], [False, False]])
    input_cube = set_up_variable_cube(
        data, name="snow_fraction", units="1", standard_grid_metadata="uk_ens",
    )
    with pytest.raises(
        NotImplementedError, match="SignificantPhaseMask cannot handle masked data"
    ):
        SignificantPhaseMask()(input_cube, "snow")


def test_name_error():
    """Tests behaviour when input cube has wrong name"""
    input_cube = set_up_variable_cube(
        np.ones((2, 2), dtype=np.float32),
        name="snow_leopards",
        units="1",
        standard_grid_metadata="uk_ens",
    )
    msg = "Expected cube named 'snow_fraction', not snow_leopards"
    with pytest.raises(ValueError, match=msg):
        SignificantPhaseMask()(input_cube, "snow")


def test_units_error():
    """Tests behaviour when input cube has wrong units"""
    input_cube = set_up_variable_cube(
        np.ones((2, 2), dtype=np.float32),
        name="snow_fraction",
        units="m",
        standard_grid_metadata="uk_ens",
    )
    msg = "Expected cube with units '1', not m"
    with pytest.raises(ValueError, match=msg):
        SignificantPhaseMask()(input_cube, "snow")


def test_phase_error():
    """Tests behaviour when requested phase is invalid"""
    input_cube = set_up_variable_cube(
        np.ones((2, 2), dtype=np.float32),
        name="snow_fraction",
        units="1",
        standard_grid_metadata="uk_ens",
    )
    msg = r"Requested phase mask 'kittens' not in \['rain', 'sleet', 'snow'\]"
    with pytest.raises(KeyError, match=msg):
        SignificantPhaseMask()(input_cube, "kittens")


@pytest.mark.parametrize("snow_fraction", (-1.0, 4.0))
def test_data_error(snow_fraction):
    """Tests behaviour when input cube has invalid data"""
    input_cube = set_up_variable_cube(
        np.full((2, 2), fill_value=snow_fraction, dtype=np.float32),
        name="snow_fraction",
        units="1",
        standard_grid_metadata="uk_ens",
    )
    msg = "Expected cube data to be in range 0 <= x <= 1. Found max={0}; min={0}".format(
        snow_fraction
    )
    with pytest.raises(ValueError, match=msg):
        SignificantPhaseMask()(input_cube, "snow")
