# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests of precipitation_type utilities"""

import numpy as np
import pytest
from iris.exceptions import CoordinateNotFoundError

from improver.metadata.constants import FLOAT_DTYPE
from improver.precipitation_type.utilities import make_shower_condition_cube
from improver.synthetic_data.set_up_test_cubes import set_up_probability_cube


def set_up_test_cube(n_thresholds=1):
    """Set up a cube testing shower condition conversion"""
    thresholds = np.arange(n_thresholds)
    shape = [2, 2]
    shape = [n_thresholds, *shape] if n_thresholds > 0 else shape
    data = np.ones(shape, dtype=FLOAT_DTYPE)
    cube = set_up_probability_cube(
        data,
        thresholds,
        variable_name="texture_of_cloud_area_fraction",
        threshold_units=1,
        spatial_grid="equalarea",
    )
    return cube


def test_basic():
    """Test that with a valid input the cube is transformed into a shower
    condition cube."""

    cube = set_up_test_cube()
    result = make_shower_condition_cube(cube)
    threshold_coord = result.coord(var_name="threshold")

    assert result.name() == "probability_of_shower_condition_above_threshold"
    assert result.dtype == FLOAT_DTYPE
    assert (result.data == cube.data).all()
    assert threshold_coord.name() == "shower_condition"
    assert threshold_coord.units == 1


def test_no_threshold_coord():
    """Test an exception is raised if the proxy diagnostic cube does not have
    a threshold coordinate."""

    cube = set_up_test_cube()
    cube.remove_coord("texture_of_cloud_area_fraction")

    expected = "Input has no threshold coordinate and cannot be used"
    with pytest.raises(CoordinateNotFoundError, match=expected):
        make_shower_condition_cube(cube)


def test_multi_valued_threshold_coord():
    """Test an exception is raised if the proxy diagnostic cube has a multi
    valued threshold coordinate."""

    cube = set_up_test_cube(n_thresholds=2)

    expected = "Expected a single valued threshold coordinate.*"
    with pytest.raises(ValueError, match=expected):
        make_shower_condition_cube(cube)
