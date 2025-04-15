# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for SnowProbabilityAtSurface plugin"""

import numpy as np
import pytest

from improver.precipitation.snow_probability_at_surface import (
    SnowProbabilityAtSurface,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

ATTRIBUTES = {
    "institution": "Met Office",
    "source": "Met Office Unified Model",
    "title": "UKV Forecast on UK 2 km Standard Grid",
}


@pytest.fixture()
def wet_bulb_temperature():
    """Create a test cube of wet bulb temperature at surface"""
    data = np.full(
        (2, 3), 100, dtype=np.float32
    )  # dummy data that will be overriden in tests
    wet_bulb_temp_at_surf = set_up_variable_cube(
        data,
        name="wet_bulb_temperature_integral",
        units=1,
        attributes=ATTRIBUTES,
    )

    return wet_bulb_temp_at_surf


@pytest.mark.parametrize(
    "wb_temp_data , expected_probability",
    ((-3, 1), (0, 1), (56.25, 0.75), (112.5, 0.5), (168.75, 0.25), (225, 0), (255, 0)),
)
def test_scenarios(wet_bulb_temperature, wb_temp_data, expected_probability):
    """Test the snow probability at surface plugin for a range of input wet bulb integral temperatures."""
    wet_bulb_temperature.data.fill(wb_temp_data)

    result = SnowProbabilityAtSurface()(wet_bulb_temperature)
    assert np.all(result.data == expected_probability)
    assert result.name() == "probability_of_snow_at_surface"
    assert result.units == "1"
    assert result.attributes == ATTRIBUTES
