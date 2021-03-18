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
"""Set up fixtures for metadata interpreter tests"""

from datetime import datetime

import iris
import numpy as np
import pytest

from improver.developer_tools.metadata_interpreter import MOMetadataInterpreter
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.synthetic_data.set_up_test_cubes import (
    construct_scalar_time_coords,
    set_up_percentile_cube,
    set_up_probability_cube,
    set_up_variable_cube,
)
from improver.wxcode.utilities import weather_code_attributes


@pytest.fixture(name="interpreter")
def interpreter_fixture():
    return MOMetadataInterpreter()


def _update_blended_time_coords(cube):
    """Replace forecast period and forecast reference time with blend time
    on blended cubes (to be updated in setup utilities once standard is
    fully implemented).  Modifies cube in place."""
    blend_time = cube.coord("forecast_reference_time").copy()
    blend_time.rename("blend_time")
    cube.add_aux_coord(blend_time)
    cube.remove_coord("forecast_period")
    cube.remove_coord("forecast_reference_time")


@pytest.fixture(name="probability_above_cube")
def probability_above_fixture():
    """Probability of air temperature above threshold cube from UKV"""
    data = 0.5 * np.ones((3, 3, 3), dtype=np.float32)
    thresholds = np.array([280, 282, 284], dtype=np.float32)
    attributes = {
        "source": "Met Office Unified Model",
        "title": "Post-Processed UKV Model Forecast on 2 km Standard Grid",
        "institution": "Met Office",
        "mosg__model_configuration": "uk_det",
    }
    return set_up_probability_cube(data, thresholds, attributes=attributes)


@pytest.fixture(name="blended_probability_below_cube")
def probability_below_fixture():
    """Probability of minimum screen temperature below threshold blended cube"""
    data = 0.5 * np.ones((3, 3, 3), dtype=np.float32)
    thresholds = np.array([280, 282, 284], dtype=np.float32)
    attributes = {
        "source": "IMPROVER",
        "title": "IMPROVER Post-Processed Multi-Model Blend on 2 km Standard Grid",
        "institution": "Met Office",
        "mosg__model_configuration": "uk_det uk_ens",
    }
    cube = set_up_probability_cube(
        data, thresholds, attributes=attributes, spp__relative_to_threshold="less_than",
    )
    _update_blended_time_coords(cube)
    cube.add_cell_method(
        iris.coords.CellMethod(
            method="maximum", coords="time", comments="of air_temperature"
        )
    )
    cube.coord("time").bounds = [
        cube.coord("time").points[0] - 3600,
        cube.coord("time").points[0],
    ]
    return cube


@pytest.fixture(name="wind_gust_percentile_cube")
def percentile_fixture():
    """Percentiles of wind gust from MOGREPS-UK"""
    data = np.array(
        [[[2, 4], [4, 2]], [[5, 8], [6, 6]], [[12, 16], [16, 15]]], dtype=np.float32
    )
    percentiles = np.array([10, 50, 90], dtype=np.float32)
    attributes = {
        "source": "Met Office Unified Model",
        "title": "MOGREPS-UK Model Forecast on 2 km Standard Grid",
        "institution": "Met Office",
        "mosg__model_configuration": "uk_ens",
        "wind_gust_diagnostic": "Typical gusts",
    }
    return set_up_percentile_cube(
        data, percentiles, name="wind_gust", units="m s-1", attributes=attributes
    )


@pytest.fixture(name="ensemble_cube")
def ensemble_fixture():
    """Raw air temperature ensemble in realization space"""
    data = 285 * np.ones((3, 3, 3), dtype=np.float32)
    attributes = {
        "source": "Met Office Unified Model",
        "title": "MOGREPS-UK Model Forecast on 2 km Standard Grid",
        "institution": "Met Office",
        "mosg__model_configuration": "uk_ens",
    }
    return set_up_variable_cube(data, attributes=attributes)


@pytest.fixture(name="snow_level_cube")
def snow_level_fixture():
    """Probability of snow falling level cube (which is a diagnostic field)"""
    data = np.zeros((3, 3), dtype=np.float32)
    name = "probability_of_snow_falling_level_below_ground_level"
    attributes = {
        "source": "IMPROVER",
        "institution": "Met Office",
        "title": "Post-Processed MOGREPS-UK Model Forecast on 2 km Standard Grid",
    }
    return set_up_variable_cube(data, name=name, units="1", attributes=attributes)


@pytest.fixture(name="blended_spot_median_cube")
def spot_fixture():
    """Spot temperature cube"""
    alts = np.array([15, 82, 0, 4, 15, 269], dtype=np.float32)
    lats = np.array([60.75, 60.13, 58.95, 57.37, 58.22, 57.72], dtype=np.float32)
    lons = np.array([-0.85, -1.18, -2.9, -7.40, -6.32, -4.90], dtype=np.float32)
    wmo_ids = np.array(["3002", "3005", "3017", "3023", "3026", "3031"])
    spot_cube = build_spotdata_cube(
        np.arange(6).astype(np.float32),
        "air_temperature",
        "degC",
        alts,
        lats,
        lons,
        wmo_ids,
    )
    spot_cube.add_aux_coord(
        iris.coords.AuxCoord([50], long_name="percentile", units="%")
    )
    spot_cube.attributes = {
        "source": "IMPROVER",
        "institution": "Met Office",
        "title": "IMPROVER Post-Processed Multi-Model Blend UK Spot Values",
        "mosg__model_configuration": "uk_det uk_ens",
    }
    (time, _), (blend_time, _), (_, _) = construct_scalar_time_coords(
        time=datetime(2021, 2, 3, 14), time_bounds=None, frt=datetime(2021, 2, 3, 10)
    )
    blend_time.rename("blend_time")
    spot_cube.add_aux_coord(time)
    spot_cube.add_aux_coord(blend_time)
    return spot_cube


@pytest.fixture(name="wxcode_cube")
def wxcode_fixture():
    """Weather symbols cube (randomly sampled data in expected range)"""
    data = np.random.randint(0, high=31, size=(3, 3))
    attributes = {
        "source": "IMPROVER",
        "institution": "Met Office",
        "title": "IMPROVER Post-Processed Multi-Model Blend on 2 km Standard Grid",
        "mosg__model_configuration": "uk_det uk_ens",
    }
    attributes.update(weather_code_attributes())
    cube = set_up_variable_cube(
        data, name="weather_code", units="1", attributes=attributes
    )
    _update_blended_time_coords(cube)
    return cube


@pytest.fixture(name="wind_direction_cube")
def wind_direction_fixture():
    """Wind direction cube from MOGREPS-UK"""
    data = np.arange(9).reshape(3, 3).astype(np.float32)
    attributes = {
        "source": "Met Office Unified Model",
        "institution": "Met Office",
        "title": "Post-Processed MOGREPS-UK Model Forecast on 2 km Standard Grid",
        "mosg__model_configuration": "uk_ens",
    }
    cube = set_up_variable_cube(
        data, name="wind_from_direction", units="degrees", attributes=attributes
    )
    cube.add_cell_method(iris.coords.CellMethod("mean", coords="realization"))
    return cube


@pytest.fixture(name="landmask_cube")
def landmask_fixture():
    """Static ancillary cube (no attributes or time coordinates)"""
    data = np.arange(9).reshape(3, 3).astype(np.float32)
    cube = set_up_variable_cube(data, name="land_binary_mask", units="1")
    for coord in ["time", "forecast_reference_time", "forecast_period"]:
        cube.remove_coord(coord)
    return cube
