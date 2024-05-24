# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Set up fixtures for metadata interpreter tests"""

from datetime import datetime

import iris
import numpy as np
import pytest

from improver.categorical.utilities import categorical_attributes
from improver.developer_tools.metadata_interpreter import MOMetadataInterpreter
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.synthetic_data.set_up_test_cubes import (
    construct_scalar_time_coords,
    construct_yx_coords,
    set_up_percentile_cube,
    set_up_probability_cube,
    set_up_variable_cube,
)
from improver_tests.categorical.decision_tree import wxcode_decision_tree


def _update_blended_time_coords(cube):
    """Replace forecast period and forecast reference time with blend time
    on blended cubes (to be updated in setup utilities once standard is
    fully implemented).  Modifies cube in place."""
    blend_time = cube.coord("forecast_reference_time").copy()
    blend_time.rename("blend_time")
    cube.add_aux_coord(blend_time)
    cube.remove_coord("forecast_period")
    cube.remove_coord("forecast_reference_time")


@pytest.fixture(name="emos_coefficient_cube")
def emos_coefficient_fixture():
    """EMOS coefficient cube (unhandled)"""
    y_coord, x_coord = construct_yx_coords(1, 1, "equalarea")
    cube = iris.cube.Cube(
        0,
        long_name="emos_coefficient_alpha",
        units="K",
        dim_coords_and_dims=None,
        aux_coords_and_dims=[(y_coord, None), (x_coord, None)],
    )
    return cube


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
    return set_up_variable_cube(data, attributes=attributes, spatial_grid="equalarea")


@pytest.fixture(name="interpreter")
def interpreter_fixture():
    return MOMetadataInterpreter()


@pytest.fixture(name="landmask_cube")
def landmask_fixture():
    """Static ancillary cube (no attributes or time coordinates)"""
    data = np.arange(9).reshape(3, 3).astype(np.float32)
    cube = set_up_variable_cube(
        data, name="land_binary_mask", units="1", spatial_grid="equalarea"
    )
    for coord in ["time", "forecast_reference_time", "forecast_period"]:
        cube.remove_coord(coord)
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
        data,
        percentiles,
        name="wind_gust",
        units="m s-1",
        attributes=attributes,
        spatial_grid="equalarea",
    )


@pytest.fixture(name="precip_accum_cube")
def precip_accum_fixture():
    """Nowcast 15 minute accumulation cube"""
    data = 0.2 * np.ones((5, 5), dtype=np.float32)
    attributes = {
        "source": "IMPROVER",
        "title": "MONOW Nowcast on UK 2 km Standard Grid",
        "institution": "Met Office",
        "mosg__model_configuration": "nc_det",
    }
    cube = set_up_variable_cube(
        data,
        name="lwe_thickness_of_precipitation_amount",
        units="mm",
        attributes=attributes,
        spatial_grid="equalarea",
    )
    cube.add_cell_method(iris.coords.CellMethod(method="sum", coords="time"))
    for coord in ["time", "forecast_period"]:
        cube.coord(coord).bounds = np.array(
            [cube.coord(coord).points[0] - 900, cube.coord(coord).points[0]],
            dtype=cube.coord(coord).dtype,
        )
    return cube


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
    return set_up_probability_cube(
        data, thresholds, attributes=attributes, spatial_grid="equalarea"
    )


@pytest.fixture(name="probability_over_time_in_vicinity_above_cube")
def probability_over_time_in_vicinity_above_fixture():
    """Probability of precipitation accumulation in 15M in vicinity above threshold cube from UKV"""
    data = 0.5 * np.ones((3, 3, 3), dtype=np.float32)
    thresholds = np.array([280, 282, 284], dtype=np.float32)
    attributes = {
        "source": "Met Office Unified Model",
        "title": "Post-Processed UKV Model Forecast on 2 km Standard Grid",
        "institution": "Met Office",
        "mosg__model_configuration": "uk_det",
    }
    diagnostic_name = "lwe_thickness_of_precipitation_amount"
    cube = set_up_probability_cube(
        data,
        thresholds,
        attributes=attributes,
        spatial_grid="equalarea",
        variable_name=f"{diagnostic_name}_in_vicinity",
    )
    cube.add_cell_method(
        iris.coords.CellMethod(
            method="sum", coords="time", comments=(f"of {diagnostic_name}",),
        )
    )
    for coord in ["time", "forecast_period"]:
        cube.coord(coord).bounds = np.array(
            [cube.coord(coord).points[0] - 900, cube.coord(coord).points[0]],
            dtype=cube.coord(coord).dtype,
        )
    return cube


@pytest.fixture(name="blended_probability_below_cube")
def probability_below_fixture():
    """Probability of maximum screen temperature below threshold blended cube"""
    data = 0.5 * np.ones((3, 3, 3), dtype=np.float32)
    thresholds = np.array([280, 282, 284], dtype=np.float32)
    attributes = {
        "source": "IMPROVER",
        "title": "IMPROVER Post-Processed Multi-Model Blend on 2 km Standard Grid",
        "institution": "Met Office",
        "mosg__model_configuration": "uk_det uk_ens",
        "mosg__model_run": "uk_det:20171109T2300Z:\nuk_ens:20171109T2100Z:",
    }
    cube = set_up_probability_cube(
        data,
        thresholds,
        attributes=attributes,
        spp__relative_to_threshold="less_than",
        spatial_grid="equalarea",
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
    return set_up_variable_cube(
        data, name=name, units="1", attributes=attributes, spatial_grid="equalarea"
    )


@pytest.fixture(name="spot_template")
def spot_fixture():
    alts = np.array([15, 82, 0, 4, 15, 269], dtype=np.float32)
    lats = np.array([60.75, 60.13, 58.95, 57.37, 58.22, 57.72], dtype=np.float32)
    lons = np.array([-0.85, -1.18, -2.9, -7.40, -6.32, -4.90], dtype=np.float32)
    wmo_ids = ["3002", "3005", "3017", "3023", "3026", "3031"]
    cube = build_spotdata_cube(
        np.arange(6).astype(np.float32),
        "air_temperature",
        "degC",
        alts,
        lats,
        lons,
        wmo_ids,
    )
    cube.add_aux_coord(iris.coords.AuxCoord([50], long_name="percentile", units="%"))
    return cube


@pytest.fixture(name="blended_spot_median_cube")
def blended_spot_median_spot_fixture(spot_template):
    """Spot temperature cube from blend"""
    cube = spot_template.copy()
    cube.attributes = {
        "source": "IMPROVER",
        "institution": "Met Office",
        "title": "IMPROVER Post-Processed Multi-Model Blend UK Spot Values",
        "mosg__model_configuration": "uk_det uk_ens",
        "mosg__model_run": "uk_det:20210203T0900Z:\nuk_ens:20210203T0700Z:",
    }
    (time, _), (blend_time, _), (_, _) = construct_scalar_time_coords(
        time=datetime(2021, 2, 3, 14), time_bounds=None, frt=datetime(2021, 2, 3, 10)
    )
    blend_time.rename("blend_time")
    cube.add_aux_coord(time)
    cube.add_aux_coord(blend_time)
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
        data,
        name="wind_from_direction",
        units="degrees",
        attributes=attributes,
        spatial_grid="equalarea",
    )
    cube.add_cell_method(iris.coords.CellMethod("mean", coords="realization"))
    return cube


@pytest.fixture(name="wxcode_cube")
def wxcode_fixture():
    """Weather symbols cube (randomly sampled data in expected range)"""
    data = np.random.randint(0, high=31, size=(3, 3))
    attributes = {
        "source": "IMPROVER",
        "institution": "Met Office",
        "title": "IMPROVER Post-Processed Multi-Model Blend on 2 km Standard Grid",
        "mosg__model_configuration": "uk_det uk_ens",
        "mosg__model_run": "uk_det:20171109T2300Z:\nuk_ens:20171109T2100Z:",
    }
    decision_tree = wxcode_decision_tree()
    attributes.update(
        categorical_attributes(decision_tree, decision_tree["meta"]["name"])
    )
    cube = set_up_variable_cube(
        data,
        name="weather_code",
        units="1",
        attributes=attributes,
        spatial_grid="equalarea",
        time_bounds=(datetime(2017, 11, 10, 3, 0), datetime(2017, 11, 10, 4, 0)),
    )
    _update_blended_time_coords(cube)
    return cube


@pytest.fixture(name="wxcode_mode_cube")
def wxcode_mode_fixture(wxcode_cube, period):
    """Weather symbols cube representing mode over time"""
    cube = wxcode_cube.copy()
    cube.add_cell_method(
        iris.coords.CellMethod("mode", coords="time", intervals=f"{period} hour")
    )
    return cube
