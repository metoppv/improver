# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
""" Tests of FieldTexture plugin."""

import iris
import numpy as np
import pytest

from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_probability_cube,
)
from improver.utilities.field_texture import FieldTexture

NB_RADIUS = 10000
TEXT_THRESH = 0.4
DIAG_THRESH = 0.8125


@pytest.fixture(name="multi_cloud_cube")
def multi_cloud_fixture():
    """Multi-realization/threshold cloud data cube."""
    cloud_area_fraction = np.zeros((3, 10, 10), dtype=np.float32)
    cloud_area_fraction[:, 1:4, 1:4] = 1.0
    thresholds = [0.265, 0.415, 0.8125]
    return cloud_probability_cube(cloud_area_fraction, thresholds)


@pytest.fixture(name="single_thresh_cloud_cube")
def single_thresh_cloud_fixture():
    """Multi-realization, single threshold cloud data cube."""
    cloud_area_fraction = np.zeros((1, 10, 10), dtype=np.float32)
    cloud_area_fraction[:, 1:4, 1:4] = 1.0
    thresholds = [0.8125]
    return cloud_probability_cube(cloud_area_fraction, thresholds)


@pytest.fixture(name="no_cloud_cube")
def no_cloud_fixture():
    """Multi-realization cloud data cube with no cloud present."""
    cloud_area_fraction = np.zeros((3, 10, 10), dtype=np.float32)
    thresholds = [0.265, 0.415, 0.8125]
    return cloud_probability_cube(cloud_area_fraction, thresholds)


@pytest.fixture(name="all_cloud_cube")
def all_cloud_fixture():
    """Multi-realization cloud data cube with all cloud present."""
    cloud_area_fraction = np.ones((3, 10, 10), dtype=np.float32)
    thresholds = [0.265, 0.415, 0.8125]
    return cloud_probability_cube(cloud_area_fraction, thresholds)


# Set up probability cubes for the pytest fixtures.


def cloud_probability_cube(data, thresholds):
    """Set up probability cube."""
    cube = set_up_probability_cube(
        data,
        variable_name="cloud_area_fraction",
        threshold_units="1",
        thresholds=thresholds,
        spatial_grid="equalarea",
    )
    cube = add_coordinate(cube, [0, 1, 2], "realization", dtype=np.int32)
    return cube


# Begin the unit tests.


def test_full_process(multi_cloud_cube):
    """Test the process function with multi realization/threshold input cube."""

    cube = multi_cloud_cube.extract(iris.Constraint(cloud_area_fraction=DIAG_THRESH))
    cube.remove_coord("cloud_area_fraction")
    expected_data = np.where(cube.data[0] == 0.0, 1.0, 0.0)

    plugin = FieldTexture(
        nbhood_radius=NB_RADIUS,
        textural_threshold=TEXT_THRESH,
        diagnostic_threshold=DIAG_THRESH,
    )
    clumpiness_result = plugin.process(multi_cloud_cube)
    np.testing.assert_almost_equal(clumpiness_result.data, expected_data, decimal=4)


def test__calculate_ratio(multi_cloud_cube):
    """Test the _calculate_ratio function with single realization of the input cube."""

    cube = multi_cloud_cube.extract(iris.Constraint(cloud_area_fraction=DIAG_THRESH))
    cube.remove_coord("cloud_area_fraction")
    expected_data = np.where(cube.data[0] == 0.0, 1.0, 0.3333)

    plugin = FieldTexture(
        nbhood_radius=NB_RADIUS,
        textural_threshold=TEXT_THRESH,
        diagnostic_threshold=DIAG_THRESH,
    )
    plugin.cube_name = "cloud_area_fraction"
    ratio_result = plugin._calculate_ratio(cube[0], NB_RADIUS)
    np.testing.assert_almost_equal(ratio_result.data, expected_data, decimal=4)


def test__calculate_transitions(multi_cloud_cube):
    """Test the _calculate_transitions function with a numpy array simulating
       the input cube."""

    cube = multi_cloud_cube.extract(iris.Constraint(cloud_area_fraction=DIAG_THRESH))
    cube.remove_coord("cloud_area_fraction")

    expected_data = np.zeros((10, 10), dtype=np.float32)
    expected_data[1:4, 1:4] = np.array(
        [[2.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 2.0]]
    )

    plugin = FieldTexture(
        nbhood_radius=NB_RADIUS,
        textural_threshold=TEXT_THRESH,
        diagnostic_threshold=DIAG_THRESH,
    )
    transition_result = plugin._calculate_transitions(cube[0].data)
    np.testing.assert_almost_equal(transition_result, expected_data, decimal=4)


def test_process_single_threshold(single_thresh_cloud_cube):
    """Test the process function with single threshold version of the multi
       realization input cube."""

    single_thresh_cloud_cube = iris.util.squeeze(single_thresh_cloud_cube)
    expected_data = np.where(single_thresh_cloud_cube.data[0] == 0.0, 1.0, 0.0)

    plugin = FieldTexture(
        nbhood_radius=NB_RADIUS,
        textural_threshold=TEXT_THRESH,
        diagnostic_threshold=DIAG_THRESH,
    )
    clumpiness_result = plugin.process(single_thresh_cloud_cube)
    np.testing.assert_almost_equal(clumpiness_result.data, expected_data, decimal=4)


def test_process_error(multi_cloud_cube):
    """Test the process function with a non-binary input cube to raise an error."""

    non_binary_cube = np.where(multi_cloud_cube.data == 0.0, 2.1, 0.0)
    plugin = FieldTexture(
        nbhood_radius=NB_RADIUS,
        textural_threshold=TEXT_THRESH,
        diagnostic_threshold=DIAG_THRESH,
    )

    with pytest.raises(Exception) as excinfo:
        plugin.process(non_binary_cube)
    assert str(excinfo.value) == "Incorrect input. Cube should hold binary data only"


def test_process_no_cloud(no_cloud_cube):
    """Test the FieldTexture plugin with multi realization input cube that has
       no cloud present in the field."""

    cube = no_cloud_cube.extract(iris.Constraint(cloud_area_fraction=DIAG_THRESH))
    cube.remove_coord("cloud_area_fraction")
    expected_data = np.ones(cube[0].shape)

    plugin = FieldTexture(
        nbhood_radius=NB_RADIUS,
        textural_threshold=TEXT_THRESH,
        diagnostic_threshold=DIAG_THRESH,
    )
    clumpiness_result = plugin.process(no_cloud_cube)
    np.testing.assert_almost_equal(clumpiness_result.data, expected_data, decimal=4)


def test_process_all_cloud(all_cloud_cube):
    """Test the process function with multi realization input cube that has
       all cloud occupying the field."""

    cube = all_cloud_cube.extract(iris.Constraint(cloud_area_fraction=DIAG_THRESH))
    cube.remove_coord("cloud_area_fraction")
    expected_data = np.zeros(cube[0].shape)

    plugin = FieldTexture(
        nbhood_radius=NB_RADIUS,
        textural_threshold=TEXT_THRESH,
        diagnostic_threshold=DIAG_THRESH,
    )
    clumpiness_result = plugin.process(all_cloud_cube)
    np.testing.assert_almost_equal(clumpiness_result.data, expected_data, decimal=4)


def test_output_metadata(multi_cloud_cube):
    """Test that the names of the output cubes follow expected conventions
       after each function in the plugin is called."""

    # ----------------- After _calculate_ratios function is performed -----------------------

    expected_name = "texture_of_cloud_area_fraction"

    cube = multi_cloud_cube.extract(iris.Constraint(cloud_area_fraction=DIAG_THRESH))
    cube.remove_coord("cloud_area_fraction")
    plugin = FieldTexture(
        nbhood_radius=NB_RADIUS,
        textural_threshold=TEXT_THRESH,
        diagnostic_threshold=DIAG_THRESH,
    )
    plugin.cube_name = "cloud_area_fraction"
    ratio = plugin._calculate_ratio(cube[0], NB_RADIUS)
    ratio_name = ratio.name()
    assert expected_name == ratio_name

    # ----------------- After process function is performed -----------------------

    expected_name = "probability_of_texture_of_cloud_area_fraction_above_threshold"

    process = plugin.process(multi_cloud_cube)
    process_name = process.name()
    assert expected_name == process_name
