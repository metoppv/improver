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
from iris.exceptions import CoordinateNotFoundError

from improver.field_texture import FieldTexture
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_probability_cube,
)

THRESHOLDS = [0.265, 0.415, 0.8125]
NB_RADIUS = 10000
TEXT_THRESH = 0.4
DIAG_THRESH = THRESHOLDS[2]


@pytest.fixture(name="multi_cloud_cube")
def multi_cloud_fixture():
    """Multi-realization/threshold cloud data cube."""
    cloud_area_fraction = np.zeros((3, 10, 10), dtype=np.float32)
    cloud_area_fraction[:, 1:4, 1:4] = 1.0
    return cloud_probability_cube(cloud_area_fraction, THRESHOLDS)


@pytest.fixture(name="thresholded_cloud_cube")
def thresholded_cloud_fixture():
    """Cloud data for a single realization at the relevant threshold."""
    cloud_area_fraction = np.zeros((1, 10, 10), dtype=np.float32)
    cloud_area_fraction[:, 1:4, 1:4] = 1.0
    multi_realization_cube = iris.util.squeeze(
        cloud_probability_cube(cloud_area_fraction, [DIAG_THRESH])
    )
    return next(multi_realization_cube.slices_over("realization"))


@pytest.fixture(name="no_realization_cloud_cube")
def no_realization_cloud_fixture():
    """No realization, multiple threshold cloud data cube."""
    cloud_area_fraction = np.zeros((3, 10, 10), dtype=np.float32)
    cloud_area_fraction[:, 1:4, 1:4] = 1.0
    multi_realization_cube = cloud_probability_cube(cloud_area_fraction, THRESHOLDS)
    no_realization_cube = next(multi_realization_cube.slices_over("realization"))
    no_realization_cube.remove_coord("realization")
    no_realization_cube.attributes.update(
        {
            "title": "UKV Forecast on 2 km Standard Grid",
            "mosg__model_configuration": "uk_det",
        }
    )
    return no_realization_cube


@pytest.fixture(name="no_cloud_cube")
def no_cloud_fixture():
    """Multi-realization cloud data cube with no cloud present."""
    cloud_area_fraction = np.zeros((3, 10, 10), dtype=np.float32)
    return cloud_probability_cube(cloud_area_fraction, THRESHOLDS)


@pytest.fixture(name="all_cloud_cube")
def all_cloud_fixture():
    """Multi-realization cloud data cube with all cloud present."""
    cloud_area_fraction = np.ones((3, 10, 10), dtype=np.float32)
    return cloud_probability_cube(cloud_area_fraction, THRESHOLDS)


def cloud_probability_cube(cloud_area_fraction, thresholds):
    """Set up probability cube."""
    cube = set_up_probability_cube(
        cloud_area_fraction,
        variable_name="cloud_area_fraction",
        threshold_units="1",
        thresholds=thresholds,
        spatial_grid="equalarea",
        attributes={
            "source": "Unified Model",
            "institution": "Met Office",
            "title": "MOGREPS-UK Forecast on 2 km Standard Grid",
            "mosg__model_configuration": "uk_ens",
        },
    )
    cube = add_coordinate(cube, [0, 1, 2], "realization", dtype=np.int32)
    return cube


def ftex_plugin():
    fieldtexture_instance = FieldTexture(
        nbhood_radius=NB_RADIUS,
        textural_threshold=TEXT_THRESH,
        diagnostic_threshold=DIAG_THRESH,
    )
    return fieldtexture_instance


def test_full_process(multi_cloud_cube):
    """Test the process function with multi realization/threshold input cube."""

    cube = multi_cloud_cube.extract(iris.Constraint(cloud_area_fraction=DIAG_THRESH))[0]
    expected_data = np.where(cube.data == 0.0, 1.0, 0.0)
    expected_dtype = "float32"

    result = ftex_plugin().process(multi_cloud_cube)
    np.testing.assert_almost_equal(result.data, expected_data, decimal=4)
    assert result.dtype == expected_dtype


def test__calculate_ratio(thresholded_cloud_cube):
    """Test the _calculate_ratio function with single realization of the input cube."""

    expected_data = np.where(thresholded_cloud_cube.data == 0.0, 1.0, 0.3333)
    result = ftex_plugin()._calculate_ratio(thresholded_cloud_cube, "cloud_area_fraction", NB_RADIUS)
    np.testing.assert_almost_equal(result.data, expected_data, decimal=4)


def test__calculate_transitions(thresholded_cloud_cube):
    """Test the _calculate_transitions function with a numpy array simulating
       the input cube."""

    expected_data = np.zeros((10, 10), dtype=np.float32)
    expected_data[1:4, 1:4] = np.array(
        [[2.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 2.0]]
    )
    result = ftex_plugin()._calculate_transitions(thresholded_cloud_cube.data)
    np.testing.assert_almost_equal(result, expected_data, decimal=4)


def test_process_single_threshold(multi_cloud_cube):
    """Test the process function with single threshold version of the multi
       realization input cube."""

    single_thresh_cloud_cube = multi_cloud_cube.extract(
        iris.Constraint(cloud_area_fraction=DIAG_THRESH)
    )
    expected_data = np.where(single_thresh_cloud_cube.data[0] == 0.0, 1.0, 0.0)
    result = ftex_plugin().process(single_thresh_cloud_cube)
    np.testing.assert_almost_equal(result.data, expected_data, decimal=4)


def test_process_no_realization(no_realization_cloud_cube):
    """Test the process function when the input cube does not contain a
       realization coordinate."""

    cube = no_realization_cloud_cube.extract(
        iris.Constraint(cloud_area_fraction=DIAG_THRESH)
    )
    expected_data = np.where(cube.data == 0.0, 1.0, 0.0)

    result = ftex_plugin().process(no_realization_cloud_cube)
    np.testing.assert_almost_equal(result.data, expected_data, decimal=4)


def test_process_error(multi_cloud_cube):
    """Test the process function with a non-binary input cube to raise an error."""

    non_binary_data = np.where(multi_cloud_cube.data == 0.0, 2.1, 0.0)
    non_binary_cube = multi_cloud_cube.copy(data=non_binary_data)

    with pytest.raises(Exception) as excinfo:
        ftex_plugin().process(non_binary_cube)
    assert str(excinfo.value) == "Incorrect input. Cube should hold binary data only"


def test_process_no_cloud(no_cloud_cube):
    """Test the FieldTexture plugin with multi realization input cube that has
       no cloud present in the field."""

    expected_data = np.ones(no_cloud_cube[0][0].shape)
    result = ftex_plugin().process(no_cloud_cube)
    np.testing.assert_almost_equal(result.data, expected_data, decimal=4)


def test_process_all_cloud(all_cloud_cube):
    """Test the process function with multi realization input cube that has
       all cloud occupying the field."""

    expected_data = np.zeros(all_cloud_cube[0][0].shape)
    result = ftex_plugin().process(all_cloud_cube)
    np.testing.assert_almost_equal(result.data, expected_data, decimal=4)


def test_no_threshold_cube(multi_cloud_cube):
    """Test the process function with multi_cloud_cube that has no thresholds"""

    multi_cloud_cube.remove_coord("cloud_area_fraction")
    with pytest.raises(CoordinateNotFoundError, match="No threshold coord found.*"):
        ftex_plugin().process(multi_cloud_cube)


def test_wrong_threshold(multi_cloud_cube):
    """Test the process function with multi_cloud_cube where user defined
        diagnostic_threshold variable does not match threshold available in
        the cube."""

    plugin = FieldTexture(
        nbhood_radius=NB_RADIUS,
        textural_threshold=TEXT_THRESH,
        diagnostic_threshold=0.235,
    )
    with pytest.raises(ValueError, match="Threshold 0.235 is not present.*"):
        plugin.process(multi_cloud_cube)


def test_metadata_name(multi_cloud_cube):
    """Test that the metadata of the output cube follows expected conventions
       after the plugin is complete and all old coordinates have been removed."""

    result = ftex_plugin().process(multi_cloud_cube)
    assert result.name() == "probability_of_texture_of_cloud_area_fraction_above_threshold"
    assert result.units == "1"


def test_metadata_coords(multi_cloud_cube):
    """Test that the coordinate metadata in the output cube follows expected
        conventions after that plugin has completed."""

    result = ftex_plugin().process(multi_cloud_cube)
    expected_dims = ["projection_y_coordinate", "projection_x_coordinate"]
    expected_scalar_coord = "texture_of_cloud_area_fraction"

    result_dims = [coord.name() for coord in result.coords(dim_coords=True)]
    result_scalar_coords = [coord.name() for coord in result.coords(dim_coords=False)]

    # check coordinates
    assert expected_dims == result_dims
    assert expected_scalar_coord in result_scalar_coords
    for coord in ["cloud_area_fraction", "realization"]:
        assert coord not in [crd.name() for crd in result.coords()]

    # check threshold coordinate units
    assert result.coord(expected_scalar_coord).units == "1"


def test_metadata_attributes(multi_cloud_cube):
    """Test that the metadata attributes in the output cube follows expected
        conventions after that plugin has completed."""

    expected_attributes = {
        "source": "Unified Model",
        "institution": "Met Office",
        "title": "MOGREPS-UK Forecast on 2 km Standard Grid",
    }

    result = ftex_plugin().process(multi_cloud_cube)
    assert result.attributes == expected_attributes


def test_metadata_attributes_with_model_id_attr(multi_cloud_cube):
    """Test that the metadata attributes in the output cube follows expected
    conventions when the model_id_attr argument is specified."""

    expected_attributes = {
        "source": "Unified Model",
        "institution": "Met Office",
        "title": "MOGREPS-UK Forecast on 2 km Standard Grid",
        "mosg__model_configuration": "uk_ens",
    }

    plugin = FieldTexture(
        nbhood_radius=NB_RADIUS,
        textural_threshold=TEXT_THRESH,
        diagnostic_threshold=DIAG_THRESH,
        model_id_attr="mosg__model_configuration",
    )

    result = plugin.process(multi_cloud_cube)
    assert result.attributes == expected_attributes
