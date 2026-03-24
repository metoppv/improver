# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the OrogLapseRate plugin."""

import cf_units
import numpy as np
import pytest
from iris.cube import Cube

from improver.orographic_adjustment.lapse_rate import OrogLapseRate
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


@pytest.fixture()
def snow_depth():
    """Sets up an array of snow depths in Liquid-water-equivalent metres."""
    snow_depth = np.array(
        [
            [0.14, 0.33, 0.57, 0.48, 0.43, 0.47, 0.31],
            [0.07, 0.17, 0.40, 0.52, 0.62, 0.59, 0.39],
            [0.03, 0.04, 0.15, 0.37, 0.58, 0.54, 0.37],
            [0.01, 0.01, 0.03, 0.13, 0.32, 0.41, 0.34],
            [0.00, 0.00, 0.00, 0.01, 0.08, 0.17, 0.23],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.03, 0.06],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.02],
        ],
        dtype=np.float32,
    )
    return snow_depth


@pytest.fixture()
def orography():
    """Sets up an array of orography altitudes in metres."""
    orography = np.array(
        [
            [836, 922, 1022, 944, 897, 928, 822],
            [756, 805, 959, 1010, 1040, 1016, 884],
            [686, 642, 771, 955, 1052, 997, 894],
            [604, 557, 609, 779, 924, 957, 902],
            [458, 456, 482, 562, 676, 807, 858],
            [310, 330, 356, 392, 460, 591, 706],
            [248, 275, 318, 383, 454, 485, 536],
        ],
        dtype=np.float32,
    )
    return orography


@pytest.fixture()
def new_orography():
    """Sets up an array of orography altitudes in metres."""
    new_orography = np.array(
        [
            [826, 1106, 860, 895, 997, 881, 727],
            [644, 1058, 1010, 1091, 1044, 889, 861],
            [526, 729, 923, 1106, 922, 935, 827],
            [493, 634, 738, 824, 1062, 814, 796],
            [415, 466, 606, 584, 836, 727, 901],
            [302, 343, 390, 425, 561, 579, 776],
            [276, 317, 358, 443, 478, 489, 657],
        ],
        dtype=np.float32,
    )
    return new_orography


def test_basic(snow_depth, orography, new_orography):
    """Test that the plugin can be run and produces expected results."""
    snow_depth_cube = set_up_variable_cube(
        snow_depth, name="snow_depth", units="m", spatial_grid="equalarea"
    )
    orography_cube = set_up_variable_cube(
        orography, name="orography", units="m", spatial_grid="equalarea"
    )
    new_orography_cube = set_up_variable_cube(
        new_orography, name="orography", units="m", spatial_grid="equalarea"
    )
    # There are examples of reaching the upper and lower local limits.
    expected_data = [
        [0.1739, 0.6200, 0.2570, 0.3518, 0.5113, 0.3536, 0.1300],
        [0.0296, 0.6200, 0.5401, 0.6200, 0.6014, 0.3490, 0.3024],
        [0.0050, 0.0796, 0.3785, 0.6200, 0.3874, 0.4112, 0.2422],
        [0.0018, 0.0323, 0.0863, 0.2128, 0.6200, 0.2196, 0.1867],
        [0.0000, 0.0000, 0.0244, 0.0194, 0.2284, 0.0932, 0.3372],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0164, 0.0229, 0.1340],
        [0.0001, 0.0000, 0.0000, 0.0004, 0.0032, 0.0053, 0.0529],
    ]
    lapse_rate_plugin = OrogLapseRate()
    lapse_rate_plugin.orography_windows = lapse_rate_plugin._create_windows(
        orography_cube.data
    )
    lapse_rate_plugin.diagnostic_windows = lapse_rate_plugin._create_windows(
        snow_depth_cube.data
    )
    result = lapse_rate_plugin.process(
        snow_depth_cube, orography_cube, new_orography_cube
    )
    assert isinstance(result, Cube)
    assert result.units == cf_units.Unit("m")
    assert np.allclose(result.data, expected_data, equal_nan=True, atol=1e-4)
