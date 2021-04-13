# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Unit tests for the MOMetadataInterpreter plugin"""

import numpy as np
import pytest
from iris.coords import CellMethod

## Test successful outputs (input cubes in alphabetical order by fixture)


def test_realizations(ensemble_cube, interpreter):
    """Test interpretation of temperature realizations from MOGREPS-UK"""
    interpreter.run(ensemble_cube)
    assert interpreter.prod_type == "gridded"
    assert interpreter.field_type == "realizations"
    assert interpreter.diagnostic == "air_temperature"
    assert interpreter.relative_to_threshold is None
    assert not interpreter.methods
    assert not interpreter.post_processed
    assert interpreter.model == "MOGREPS-UK"
    assert not interpreter.blended
    assert not interpreter.warnings


def test_improver_in_title(ensemble_cube, interpreter):
    """Test that an unblended title including 'IMPROVER' rather than a model name
    does not cause an error, and that the source model is still identified via the
    appropriate attribute"""
    ensemble_cube.attributes["title"].replace("MOGREPS-UK", "IMPROVER")
    interpreter.run(ensemble_cube)
    assert interpreter.model == "MOGREPS-UK"


def test_static_ancillary(landmask_cube, interpreter):
    """Test interpretation of static ancillary"""
    interpreter.run(landmask_cube)
    assert interpreter.prod_type == "gridded"
    assert interpreter.field_type == "ancillary"
    assert interpreter.diagnostic == "land_binary_mask"
    assert interpreter.relative_to_threshold is None
    assert not interpreter.methods
    assert interpreter.post_processed is None
    assert interpreter.model is None
    assert not interpreter.blended
    assert not interpreter.warnings


def test_percentiles(wind_gust_percentile_cube, interpreter):
    """Test interpretation of wind gust percentiles from MOGREPS-UK"""
    interpreter.run(wind_gust_percentile_cube)
    assert interpreter.prod_type == "gridded"
    assert interpreter.field_type == "percentiles"
    assert interpreter.diagnostic == "wind_gust"
    assert interpreter.relative_to_threshold is None
    assert not interpreter.methods
    assert not interpreter.post_processed
    assert interpreter.model == "MOGREPS-UK"
    assert not interpreter.blended
    assert not interpreter.warnings


def test_precip_accum(precip_accum_cube, interpreter):
    """Test interpretation of nowcast precipitation accumulations"""
    interpreter.run(precip_accum_cube)
    assert interpreter.prod_type == "gridded"
    assert interpreter.field_type == "realizations"
    assert interpreter.diagnostic == "lwe_thickness_of_precipitation_amount"
    assert interpreter.relative_to_threshold is None
    assert interpreter.methods == " sum over time"
    assert not interpreter.post_processed
    assert interpreter.model == "Nowcast"
    assert not interpreter.blended
    assert not interpreter.warnings


def test_probabilities_above(probability_above_cube, interpreter):
    """Test interpretation of probability of temperature above threshold
    from UKV"""
    interpreter.run(probability_above_cube)
    assert interpreter.prod_type == "gridded"
    assert interpreter.field_type == "probabilities"
    assert interpreter.diagnostic == "air_temperature"
    assert interpreter.relative_to_threshold == "greater_than"
    assert not interpreter.methods
    assert interpreter.post_processed
    assert interpreter.model == "UKV"
    assert not interpreter.blended
    assert not interpreter.warnings


def test_handles_duplicate_model_string(probability_above_cube, interpreter):
    """Test the interpreter can distinguish between the Global Model and the
    Global Grid in cube titles"""
    probability_above_cube.attributes[
        "title"
    ] = "UKV Model Forecast on Global 10 km Standard Grid"
    interpreter.run(probability_above_cube)
    assert interpreter.model == "UKV"


def test_probabilities_below(blended_probability_below_cube, interpreter):
    """Test interpretation of blended probability of max temperature in hour
    below threshold"""
    interpreter.run(blended_probability_below_cube)
    assert interpreter.prod_type == "gridded"
    assert interpreter.field_type == "probabilities"
    assert interpreter.diagnostic == "air_temperature"
    assert interpreter.relative_to_threshold == "less_than"
    assert interpreter.methods == " maximum over time"
    assert interpreter.post_processed
    assert interpreter.model == "UKV, MOGREPS-UK"
    assert interpreter.blended
    assert not interpreter.warnings


def test_snow_level(snow_level_cube, interpreter):
    """Test interpretation of a diagnostic cube with "probability" in the name,
    which is not designed for blending with other models"""
    interpreter.run(snow_level_cube)
    assert interpreter.prod_type == "gridded"
    assert interpreter.field_type == "realizations"
    assert (
        interpreter.diagnostic == "probability_of_snow_falling_level_below_ground_level"
    )
    assert interpreter.relative_to_threshold is None
    assert not interpreter.methods
    assert interpreter.post_processed
    assert interpreter.model is None
    assert not interpreter.blended
    assert not interpreter.warnings


def test_spot_median(blended_spot_median_cube, interpreter):
    """Test interpretation of spot median"""
    interpreter.run(blended_spot_median_cube)
    assert interpreter.prod_type == "spot"
    assert interpreter.field_type == "percentiles"
    assert interpreter.diagnostic == "air_temperature"
    assert interpreter.relative_to_threshold is None
    assert not interpreter.methods
    assert interpreter.post_processed
    assert interpreter.model == "UKV, MOGREPS-UK"
    assert interpreter.blended
    assert not interpreter.warnings


def test_wind_direction(wind_direction_cube, interpreter):
    """Test interpretation of wind direction field with mean over realizations
    cell method"""
    interpreter.run(wind_direction_cube)
    assert interpreter.diagnostic == "wind_from_direction"
    assert interpreter.model == "MOGREPS-UK"
    assert not interpreter.blended


def test_weather_code(wxcode_cube, interpreter):
    """Test interpretation of weather code field"""
    interpreter.run(wxcode_cube)
    assert interpreter.diagnostic == "weather_code"
    assert interpreter.model == "UKV, MOGREPS-UK"
    assert interpreter.blended


## Test errors and warnings (input cubes in alphabetical order by fixture)


def test_error_inconsistent_model_attributes(ensemble_cube, interpreter):
    """Test error raised when the model ID and title attributes are inconsistent"""
    ensemble_cube.attributes["mosg__model_configuration"] = "uk_det"
    with pytest.raises(ValueError, match="Title.*is inconsistent with model ID"):
        interpreter.run(ensemble_cube)


def test_error_ancillary_cell_method(landmask_cube, interpreter):
    """Test error raised when there's a cell method on a static ancillary"""
    landmask_cube.add_cell_method(CellMethod(method="maximum", coords="time"))
    with pytest.raises(ValueError, match="Unexpected cell methods"):
        interpreter.run(landmask_cube)


def test_error_wrong_percentile_name_units(wind_gust_percentile_cube, interpreter):
    """Test incorrect percentile coordinate name and units"""
    wind_gust_percentile_cube.coord("percentile").units = "1"
    wind_gust_percentile_cube.coord("percentile").rename("percentile_over_realization")
    msg = (
        ".*should have name percentile, has percentile_over_realization\n"
        ".*should have units of %, has 1"
    )
    with pytest.raises(ValueError, match=msg):
        interpreter.run(wind_gust_percentile_cube)


# Test attribute errors


def test_error_forbidden_attributes(wind_gust_percentile_cube, interpreter):
    """Test error raised when a forbidden attribute is present"""
    wind_gust_percentile_cube.attributes["mosg__forecast_run_duration"] = "wrong"
    with pytest.raises(ValueError, match="Attributes.*include.*forbidden values"):
        interpreter.run(wind_gust_percentile_cube)


def test_error_missing_required_attribute(wind_gust_percentile_cube, interpreter):
    """Test error raised when a mandatory attribute is missing"""
    wind_gust_percentile_cube.attributes.pop("source")
    with pytest.raises(ValueError, match="missing.*mandatory values"):
        interpreter.run(wind_gust_percentile_cube)


def test_error_missing_wind_gust_attribute(wind_gust_percentile_cube, interpreter):
    """Test error when a wind gust percentile cube is missing a required attribute"""
    wind_gust_percentile_cube.attributes.pop("wind_gust_diagnostic")
    with pytest.raises(ValueError, match="missing .* required values"):
        interpreter.run(wind_gust_percentile_cube)


def test_warning_wind_gust_attribute_wrong_diagnostic(
    wind_gust_percentile_cube, interpreter
):
    """Test a warning is raised if a known diagnostic-specific attribute is included
    on an unexpected diagnostic"""
    wind_gust_percentile_cube.rename("wind_speed")
    interpreter.run(wind_gust_percentile_cube)
    assert interpreter.warnings == [
        "dict_keys(['source', 'title', 'institution', 'mosg__model_configuration', "
        "'wind_gust_diagnostic']) include unexpected attributes ['wind_gust_diagnostic']. "
        "Please check the standard to ensure this is valid."
    ]


# Test cell method errors


def test_warning_unexpected_cell_method(wind_gust_percentile_cube, interpreter):
    """Test a warning is raised if an unexpected, but not forbidden, cell method is
    present"""
    wind_gust_percentile_cube.add_cell_method(
        CellMethod(method="variance", coords="time")
    )
    interpreter.run(wind_gust_percentile_cube)
    assert interpreter.warnings == [
        "Unexpected cell method variance: time. Please check the standard to "
        "ensure this is valid"
    ]


def test_error_missing_accum_cell_method(precip_accum_cube, interpreter):
    """Test error when precip accumulation cube has no time cell method"""
    precip_accum_cube.cell_methods = []
    with pytest.raises(ValueError, match="Expected sum over time"):
        interpreter.run(precip_accum_cube)


def test_error_wrong_accum_cell_method(precip_accum_cube, interpreter):
    """Test error when precipitation accumulation cube has the wrong cell method"""
    precip_accum_cube.cell_methods = []
    precip_accum_cube.add_cell_method(CellMethod(method="mean", coords="time"))
    with pytest.raises(ValueError, match="Expected sum over time"):
        interpreter.run(precip_accum_cube)


# Test probabilistic metadata errors


def test_error_invalid_probability_name(probability_above_cube, interpreter):
    """Test error raised if probability cube name is invalid"""
    probability_above_cube.rename("probability_air_temperature_is_above_threshold")
    with pytest.raises(ValueError, match="is not a valid probability cube name"):
        interpreter.run(probability_above_cube)


def test_error_invalid_probability_name_no_threshold(
    probability_above_cube, interpreter
):
    """Test error raised if probability cube name is invalid"""
    probability_above_cube.rename("probability_of_air_temperature")
    with pytest.raises(
        ValueError, match="is not consistent with spp__relative_to_threshold"
    ):
        interpreter.run(probability_above_cube)


def test_error_invalid_probability_units(probability_above_cube, interpreter):
    """Test error raised if probability cube does not have units of 1"""
    probability_above_cube.units = "no_unit"
    with pytest.raises(ValueError, match="Expected units of 1 on probability data"):
        interpreter.run(probability_above_cube)


def test_error_no_threshold_coordinate(probability_above_cube, interpreter):
    """Test error raised if probability cube has no threshold coordinate"""
    cube = next(probability_above_cube.slices_over("air_temperature"))
    cube.remove_coord("air_temperature")
    with pytest.raises(ValueError, match="no coord with var_name='threshold' found"):
        interpreter.run(cube)


def test_error_invalid_threshold_name(probability_above_cube, interpreter):
    """Test error raised if threshold coordinate name does not match cube name"""
    probability_above_cube.coord("air_temperature").rename("screen_temperature")
    probability_above_cube.coord("screen_temperature").var_name = "threshold"
    with pytest.raises(ValueError, match="expected threshold coord.*incorrect name"):
        interpreter.run(probability_above_cube)


def test_error_no_threshold_var_name(probability_above_cube, interpreter):
    """Test error raised if threshold coordinate does not have var_name='threshold'"""
    probability_above_cube.coord("air_temperature").var_name = None
    with pytest.raises(ValueError, match="does not have var_name='threshold'"):
        interpreter.run(probability_above_cube)


def test_error_missing_relative_to_threshold(probability_above_cube, interpreter):
    """Test error raised if threshold coordinate has no spp__relative_to_threshold
    attribute"""
    probability_above_cube.coord("air_temperature").attributes = {}
    with pytest.raises(ValueError, match="has no spp__relative_to_threshold attribute"):
        interpreter.run(probability_above_cube)


def test_error_invalid_relative_to_threshold(probability_above_cube, interpreter):
    """Test error raised if the spp__relative_to_threshold attribute is not one of the
    permitted values"""
    probability_above_cube.coord("air_temperature").attributes[
        "spp__relative_to_threshold"
    ] = "above"
    with pytest.raises(
        ValueError, match="attribute 'above' is not in permitted value set"
    ):
        interpreter.run(probability_above_cube)


def test_error_inconsistent_relative_to_threshold(probability_above_cube, interpreter):
    """Test error raised if the spp__relative_to_threshold attribute is inconsistent
    with the cube name"""
    probability_above_cube.coord("air_temperature").attributes[
        "spp__relative_to_threshold"
    ] = "less_than"
    with pytest.raises(
        ValueError, match="name.*above.*is not consistent with.*less_than"
    ):
        interpreter.run(probability_above_cube)


# Test errors related to time coordinates


def test_error_missing_time_coords(probability_above_cube, interpreter):
    """Test error raised if an unblended cube doesn't have the expected time
    coordinates"""
    probability_above_cube.remove_coord("forecast_period")
    with pytest.raises(ValueError, match="Missing one or more coordinates"):
        interpreter.run(probability_above_cube)


def test_error_time_coord_units(probability_above_cube, interpreter):
    """Test error raised if time coordinate units do not match the IMPROVER standard"""
    probability_above_cube.coord("forecast_period").convert_units("hours")
    with pytest.raises(ValueError, match="does not have required units"):
        interpreter.run(probability_above_cube)


# Test the interpreter can return multiple errors.


def test_multiple_error_concatenation(probability_above_cube, interpreter):
    """Test multiple errors are concatenated and returned correctly in a readable
    format"""
    probability_above_cube.coord("air_temperature").attributes[
        "spp__relative_to_threshold"
    ] = "less_than"
    probability_above_cube.coord("air_temperature").var_name = None
    probability_above_cube.attributes["um_version"] = "irrelevant"
    msg = (
        ".*does not have var_name='threshold'.*\n"
        ".*name.*above.*is not consistent with.*less_than.*\n"
        "Attributes.*include one or more forbidden values.*"
    )
    with pytest.raises(ValueError, match=msg):
        interpreter.run(probability_above_cube)


# Test errors for cell methods that aren't compliant with the standard.


def test_error_forbidden_cell_method(blended_probability_below_cube, interpreter):
    """Test error raised when a forbidden cell method is present"""
    blended_probability_below_cube.add_cell_method(
        CellMethod(method="mean", coords="forecast_reference_time")
    )
    with pytest.raises(ValueError, match="Non-standard cell method"):
        interpreter.run(blended_probability_below_cube)


def test_error_probability_cell_method_no_comment(
    blended_probability_below_cube, interpreter
):
    """Test error raised if a probability cube cell method does not have a
    comment referring to the source diagnostic"""
    cell_method = CellMethod(method="maximum", coords="time")
    blended_probability_below_cube.cell_methods = [cell_method]
    with pytest.raises(
        ValueError, match="Cell method.*on probability data should have comment"
    ):
        interpreter.run(blended_probability_below_cube)


# Test errors caused by time coordinate bounds


def test_error_missing_time_bounds(blended_probability_below_cube, interpreter):
    """Test error raised if a minimum in time cube has no time bounds"""
    blended_probability_below_cube.coord("time").bounds = None
    with pytest.raises(ValueError, match="has no time bounds"):
        interpreter.run(blended_probability_below_cube)


def test_error_incorrect_time_bounds(blended_probability_below_cube, interpreter):
    """Test error raised if time points are not equal to upper bounds"""
    blended_probability_below_cube.coord("time").bounds = np.array(
        blended_probability_below_cube.coord("time").bounds + 3600, dtype=np.int64
    )
    with pytest.raises(ValueError, match="points should be equal to upper bounds"):
        interpreter.run(blended_probability_below_cube)


# Tests for metadata related to blending


def test_error_missing_blended_coords(blended_probability_below_cube, interpreter):
    """Test error raised if a blended cube doesn't have the expected time
    coordinates"""
    blended_probability_below_cube.remove_coord("blend_time")
    with pytest.raises(ValueError, match="Missing one or more coordinates"):
        interpreter.run(blended_probability_below_cube)


def test_error_missing_model_information(blended_probability_below_cube, interpreter):
    """Test error raised if a blended cube doesn't have a model ID attribute"""
    blended_probability_below_cube.attributes.pop("mosg__model_configuration")
    with pytest.raises(ValueError, match="on blended file"):
        interpreter.run(blended_probability_below_cube)


def test_error_unrecognised_model_in_blend(blended_probability_below_cube, interpreter):
    """Test error when a blended model ID attribute has an unknown value"""
    blended_probability_below_cube.attributes[
        "mosg__model_configuration"
    ] = "nc_ens uk_det"
    with pytest.raises(ValueError, match="unrecognised model code"):
        interpreter.run(blended_probability_below_cube)


def test_error_blend_missing_from_title(blended_probability_below_cube, interpreter):
    """Test error raised if a blended cube title doesn't indicate a blend, but the
    model ID attribute contains multiple models"""
    blended_probability_below_cube.attributes[
        "title"
    ] = "IMPROVER Forecast on UK 2 km Standard Grid"
    with pytest.raises(ValueError, match="is not a valid single model"):
        interpreter.run(blended_probability_below_cube)


# Test errors due to incorrect spot metadata


def test_error_missing_spot_coords(blended_spot_median_cube, interpreter):
    """Test error raised if a spot cube doesn't have all the expected metadata"""
    blended_spot_median_cube.remove_coord("altitude")
    with pytest.raises(ValueError, match="Missing one or more coordinates"):
        interpreter.run(blended_spot_median_cube)


def test_error_inconsistent_spot_title(blended_spot_median_cube, interpreter):
    """Test error raised if a spot cube has a non-spot title"""
    blended_spot_median_cube.attributes[
        "title"
    ] = "IMPROVER Post-Processed Multi-Model Blend on UK 2 km Standard Grid"
    with pytest.raises(ValueError, match="not consistent with spot data"):
        interpreter.run(blended_spot_median_cube)


# Test errors related to special cases


def test_error_forbidden_wind_direction_cell_method(wind_direction_cube, interpreter):
    """Test error if special case cubes have a cell method that would usually be
    permitted"""
    wind_direction_cube.add_cell_method(CellMethod(method="maximum", coords="time"))
    with pytest.raises(ValueError, match="Unexpected cell methods"):
        interpreter.run(wind_direction_cube)


def test_error_forbidden_weather_code_cell_method(wxcode_cube, interpreter):
    """Test error if special case cubes have a cell method that would usually be
    permitted"""
    wxcode_cube.add_cell_method(CellMethod(method="maximum", coords="time"))
    with pytest.raises(ValueError, match="Unexpected cell methods"):
        interpreter.run(wxcode_cube)


def test_error_missing_weather_code_attribute(wxcode_cube, interpreter):
    """Test error when weather code required attributes are missing"""
    wxcode_cube.attributes.pop("weather_code")
    with pytest.raises(ValueError, match="missing .* required values"):
        interpreter.run(wxcode_cube)
