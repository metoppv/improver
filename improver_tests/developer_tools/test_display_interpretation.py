# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the display_interpretation function"""

import pytest

from improver.developer_tools.metadata_interpreter import display_interpretation


def test_unhandled(emos_coefficient_cube, interpreter):
    """Test unhandled intermediate cube"""
    interpreter.run(emos_coefficient_cube)
    result = display_interpretation(interpreter)
    assert result == "emos_coefficient_alpha is not handled by this interpreter\n"


def test_realizations(ensemble_cube, interpreter):
    """Test interpretation of temperature realizations from MOGREPS-UK"""
    expected_result = (
        "This is a gridded file containing one or more realizations\n"
        "It contains realizations of air temperature\n"
        "It has undergone no significant post-processing\n"
        "It contains data from MOGREPS-UK\n"
    )
    interpreter.run(ensemble_cube)
    result = display_interpretation(interpreter)
    assert result == expected_result


def test_static_ancillary(landmask_cube, interpreter):
    """Test interpretation of static ancillary"""
    expected_result = (
        "This is a gridded ancillary file\n"
        "This is a static ancillary with no time information\n"
    )
    interpreter.run(landmask_cube)
    result = display_interpretation(interpreter)
    assert result == expected_result


def test_percentiles(wind_gust_percentile_cube, interpreter):
    """Test interpretation of wind gust percentiles from MOGREPS-UK"""
    expected_result = (
        "This is a gridded percentiles file\n"
        "It contains percentiles of wind gust\n"
        "It has undergone no significant post-processing\n"
        "It contains data from MOGREPS-UK\n"
    )
    interpreter.run(wind_gust_percentile_cube)
    result = display_interpretation(interpreter)
    assert result == expected_result


def test_warnings_displayed(wind_gust_percentile_cube, interpreter):
    """Test any warnings are appended to the end of the displayed output"""
    expected_result = (
        "This is a gridded percentiles file\n"
        "It contains percentiles of wind gust\n"
        "It has undergone no significant post-processing\n"
        "It contains data from MOGREPS-UK\n"
        "WARNINGS:\n"
        "dict_keys(['source', 'title', 'institution', 'mosg__model_configuration', "
        "'wind_gust_diagnostic', 'enigma']) include unexpected attributes ['enigma']. "
        "Please check the standard to ensure this is valid.\n"
    )
    wind_gust_percentile_cube.attributes["enigma"] = "intriguing and mysterious details"
    interpreter.run(wind_gust_percentile_cube)
    result = display_interpretation(interpreter)
    assert result == expected_result


def test_probabilities_above(probability_above_cube, interpreter):
    """Test interpretation of probability of temperature above threshold
    from UKV"""
    expected_result = (
        "This is a gridded probabilities file\n"
        "It contains probabilities of air temperature greater than thresholds\n"
        "It has undergone some significant post-processing\n"
        "It contains data from UKV\n"
    )
    interpreter.run(probability_above_cube)
    result = display_interpretation(interpreter)
    assert result == expected_result


def test_verbose_probability(probability_above_cube, interpreter):
    """Test verbose output for a simple probability diagnostic"""
    expected_result = (
        "This is a gridded probabilities file\n"
        "    Source: name, coordinates\n"
        "It contains probabilities of air temperature greater than thresholds\n"
        "    Source: name, threshold coordinate (probabilities only)\n"
        "It has undergone some significant post-processing\n"
        "    Source: title attribute\n"
        "It contains data from UKV\n"
        "    Source: model ID attribute\n"
    )
    interpreter.run(probability_above_cube)
    result = display_interpretation(interpreter, verbose=True)
    assert result == expected_result


def test_probabilities_below(blended_probability_below_cube, interpreter):
    """Test interpretation of blended probability of max temperature in hour
    below threshold with model cycle information available"""
    expected_result = (
        "This is a gridded probabilities file\n"
        "It contains probabilities of air temperature less than thresholds\n"
        "These probabilities are of air temperature maximum over time\n"
        "It has undergone some significant post-processing\n"
        "It contains blended data from models: UKV (cycle: 20171109T2300Z), "
        "MOGREPS-UK (cycle: 20171109T2100Z)\n"
    )
    interpreter.run(blended_probability_below_cube)
    result = display_interpretation(interpreter)
    assert result == expected_result


def test_probabilities_below_verbose(blended_probability_below_cube, interpreter):
    """Test interpretation of blended probability of max temperature in hour
    below threshold with model cycle information available"""
    expected_result = (
        "This is a gridded probabilities file\n"
        "    Source: name, coordinates\n"
        "It contains probabilities of air temperature less than thresholds\n"
        "    Source: name, threshold coordinate (probabilities only)\n"
        "These probabilities are of air temperature maximum over time\n"
        "    Source: cell methods\n"
        "It has undergone some significant post-processing\n"
        "    Source: title attribute\n"
        "It contains blended data from models: UKV (cycle: 20171109T2300Z), "
        "MOGREPS-UK (cycle: 20171109T2100Z)\n"
        "    Source: title attribute, model ID attribute, model run attribute\n"
    )
    interpreter.run(blended_probability_below_cube)
    result = display_interpretation(interpreter, verbose=True)
    assert result == expected_result


def test_verbose_snow_level(snow_level_cube, interpreter):
    """Test interpretation of a diagnostic cube with "probability" in the name,
    which is not designed for blending with other models"""
    expected_result = (
        "This is a gridded file containing one or more realizations\n"
        "    Source: name, coordinates\n"
        "It contains realizations of probability of snow falling level below ground level\n"
        "    Source: name, threshold coordinate (probabilities only)\n"
        "It has undergone some significant post-processing\n"
        "    Source: title attribute\n"
        "It has no source model information and cannot be blended\n"
        "    Source: model ID attribute (missing)\n"
    )
    interpreter.run(snow_level_cube)
    result = display_interpretation(interpreter, verbose=True)
    assert result == expected_result


def test_spot_median(blended_spot_median_cube, interpreter):
    """Test interpretation of spot median"""
    expected_result = (
        "This is a spot percentiles file\n"
        "It contains percentiles of air temperature\n"
        "It has undergone some significant post-processing\n"
        "It contains blended data from models: UKV (cycle: 20210203T0900Z), "
        "MOGREPS-UK (cycle: 20210203T0700Z)\n"
    )
    interpreter.run(blended_spot_median_cube)
    result = display_interpretation(interpreter)
    assert result == expected_result


def test_wind_direction(wind_direction_cube, interpreter):
    """Test interpretation of wind direction field with mean over realizations
    cell method"""
    expected_result = (
        "This is a gridded wind from direction file\n"
        "It contains data from MOGREPS-UK\n"
    )
    interpreter.run(wind_direction_cube)
    result = display_interpretation(interpreter)
    assert result == expected_result


def test_weather_code(wxcode_cube, interpreter):
    """Test interpretation of weather code field"""
    expected_result = (
        "This is a gridded weather code file\n"
        "It contains blended data from models: UKV (cycle: 20171109T2300Z), "
        "MOGREPS-UK (cycle: 20171109T2100Z)\n"
    )
    interpreter.run(wxcode_cube)
    result = display_interpretation(interpreter)
    assert result == expected_result


@pytest.mark.parametrize("period", [1, 3])
def test_weather_mode_code(wxcode_mode_cube, period, interpreter):
    """Test interpretation of weather code field"""
    expected_result = (
        "This is a gridded weather code file\n"
        f"These weather code are mode of {period} hour weather code over time\n"
        "It contains blended data from models: UKV (cycle: 20171109T2300Z), "
        "MOGREPS-UK (cycle: 20171109T2100Z)\n"
    )
    interpreter.run(wxcode_mode_cube)
    result = display_interpretation(interpreter)
    assert result == expected_result
