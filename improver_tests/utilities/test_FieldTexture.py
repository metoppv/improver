# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
""" Tests of FieldTexture plugin"""

import iris
import numpy as np
import pytest
from iris.exceptions import CoordinateNotFoundError

from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_probability_cube,
)
from improver.utilities.textural import FieldTexture

THRESHOLDS = [0.265, 0.415, 0.8125]
NB_RADIUS = 10000
TEXT_THRESH = 0.4
DIAG_THRESH = THRESHOLDS[2]
CAF = "cloud_area_fraction"
REALIZATION = "realization"
COMMON_ATTRS = {
    "source": "Unified Model",
    "institution": "Met Office",
}
UK_ENS_ATTRS = {
    **COMMON_ATTRS,
    "title": "MOGREPS-UK Forecast on 2 km Standard Grid",
    "mosg__model_configuration": "uk_ens",
}
UK_DET_ATTRS = {
    **COMMON_ATTRS,
    "title": "UKV Forecast on 2 km Standard Grid",
    "mosg__model_configuration": "uk_det",
}


@pytest.fixture(name="multi_cloud_cube")
def multi_cloud_fixture():
    """Multi-realization/threshold cloud data cube"""
    cloud_area_fraction = np.zeros((3, 10, 10), dtype=np.float32)
    cloud_area_fraction[:, 1:4, 1:4] = 1.0
    return cloud_probability_cube(cloud_area_fraction, THRESHOLDS)


@pytest.fixture(name="thresholded_cloud_cube")
def thresholded_cloud_fixture():
    """Cloud data for a single realization at the relevant threshold"""
    cloud_area_fraction = np.zeros((1, 10, 10), dtype=np.float32)
    cloud_area_fraction[:, 1:4, 1:4] = 1.0
    multi_realization_cube = iris.util.squeeze(
        cloud_probability_cube(cloud_area_fraction, [DIAG_THRESH])
    )
    return next(multi_realization_cube.slices_over(REALIZATION))


@pytest.fixture(name="no_realization_cloud_cube")
def no_realization_cloud_fixture(multi_cloud_cube):
    """No realization, multiple threshold cloud data cube"""
    no_realization_cube = next(multi_cloud_cube.slices_over(REALIZATION))
    no_realization_cube.remove_coord(REALIZATION)
    no_realization_cube.attributes.update(UK_DET_ATTRS)
    return no_realization_cube


def cloud_probability_cube(cloud_area_fraction, thresholds):
    """Set up probability cube"""
    cube = set_up_probability_cube(
        cloud_area_fraction,
        variable_name=CAF,
        threshold_units="1",
        thresholds=thresholds,
        spatial_grid="equalarea",
        attributes=UK_ENS_ATTRS,
    )
    cube = add_coordinate(cube, [0, 1, 2], REALIZATION, dtype=np.int32)
    return cube


def ftex_plugin(
    nbhood_radius=NB_RADIUS,
    textural_threshold=TEXT_THRESH,
    diagnostic_threshold=DIAG_THRESH,
):
    """Create an instance of the FieldTexture plugin with standard arguments"""
    fieldtexture_instance = FieldTexture(
        nbhood_radius, textural_threshold, diagnostic_threshold,
    )
    return fieldtexture_instance


@pytest.mark.parametrize("textural_threshold, expected_value", ((0.1, 1.0), (0.5, 0.0)))
def test_full_process(multi_cloud_cube, textural_threshold, expected_value):
    """Test the process function with multi realization/threshold input cube"""
    cube = multi_cloud_cube.extract(iris.Constraint(cloud_area_fraction=DIAG_THRESH))[0]
    expected_data = np.where(cube.data == 0.0, 1.0, expected_value)
    result = ftex_plugin(textural_threshold=textural_threshold).process(
        multi_cloud_cube
    )
    np.testing.assert_allclose(result.data, expected_data)
    assert result.dtype == np.float32


def test__calculate_ratio(thresholded_cloud_cube):
    """Test the _calculate_ratio function with single realization of the input cube"""
    expected_data = np.where(thresholded_cloud_cube.data == 0.0, 1.0, 1.0 / 3.0)
    result = ftex_plugin()._calculate_ratio(thresholded_cloud_cube, CAF, NB_RADIUS)
    np.testing.assert_allclose(result.data, expected_data)


def test__calculate_transitions(thresholded_cloud_cube):
    """Test the _calculate_transitions function with a numpy array simulating
       the input cube"""
    expected_data = np.zeros((10, 10), dtype=np.float32)
    expected_data[1:4, 1:4] = np.array(
        [[2.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 2.0]]
    )
    result = ftex_plugin()._calculate_transitions(thresholded_cloud_cube.data)
    np.testing.assert_allclose(result.data, expected_data)


def test_process_single_threshold(multi_cloud_cube):
    """Test the process function with single threshold version of the multi
       realization input cube"""
    single_thresh_cloud_cube = multi_cloud_cube.extract(
        iris.Constraint(cloud_area_fraction=DIAG_THRESH)
    )
    expected_data = np.where(single_thresh_cloud_cube.data[0] == 0.0, 1.0, 0.0)
    result = ftex_plugin().process(single_thresh_cloud_cube)
    np.testing.assert_allclose(result.data, expected_data)


def test_process_no_realization(no_realization_cloud_cube):
    """Test the process function when the input cube does not contain a
       realization coordinate"""
    cube = no_realization_cloud_cube.extract(
        iris.Constraint(cloud_area_fraction=DIAG_THRESH)
    )
    expected_data = np.where(cube.data == 0.0, 1.0, 0.0)
    result = ftex_plugin().process(no_realization_cloud_cube)
    np.testing.assert_allclose(result.data, expected_data)


def test_process_error(multi_cloud_cube):
    """Test the process function with a non-binary input cube to raise an error"""
    non_binary_data = np.where(multi_cloud_cube.data == 0.0, 2.1, 0.0)
    non_binary_cube = multi_cloud_cube.copy(data=non_binary_data)
    with pytest.raises(ValueError, match="binary data"):
        ftex_plugin().process(non_binary_cube)


@pytest.mark.parametrize("cloud_frac, expected", ((0.0, 1.0), (1.0, 0.0)))
def test_process_constant_cloud(multi_cloud_cube, cloud_frac, expected):
    """Test the FieldTexture plugin with multi realization input cube that has
       no or all cloud present in the field"""
    multi_cloud_cube.data[:] = cloud_frac
    result = ftex_plugin().process(multi_cloud_cube)
    np.testing.assert_allclose(result.data, expected)


def test_no_threshold_cube(multi_cloud_cube):
    """Test the process function with multi_cloud_cube that has no thresholds"""
    multi_cloud_cube.remove_coord(CAF)
    with pytest.raises(CoordinateNotFoundError, match="threshold coord"):
        ftex_plugin().process(multi_cloud_cube)


def test_wrong_threshold(multi_cloud_cube):
    """Test the process function with multi_cloud_cube where user defined
        diagnostic_threshold variable does not match threshold available in
        the cube"""
    wrong_threshold = 0.235
    plugin = FieldTexture(
        nbhood_radius=NB_RADIUS,
        textural_threshold=TEXT_THRESH,
        diagnostic_threshold=wrong_threshold,
    )
    with pytest.raises(ValueError, match=str(wrong_threshold)):
        plugin.process(multi_cloud_cube)


def test_metadata_name(multi_cloud_cube):
    """Test that the metadata of the output cube follows expected conventions
       after the plugin is complete and all old coordinates have been removed"""
    result = ftex_plugin().process(multi_cloud_cube)
    assert result.name() == f"probability_of_texture_of_{CAF}_above_threshold"
    assert result.units == "1"


def test_metadata_coords(multi_cloud_cube):
    """Test that the coordinate metadata in the output cube follows expected
        conventions after that plugin has completed"""
    result = ftex_plugin().process(multi_cloud_cube)
    expected_scalar_coord = f"texture_of_{CAF}"

    result_dims = [coord.name() for coord in result.coords(dim_coords=True)]
    result_scalar_coords = [coord.name() for coord in result.coords(dim_coords=False)]

    # check coordinates
    assert result_dims == ["projection_y_coordinate", "projection_x_coordinate"]
    assert expected_scalar_coord in result_scalar_coords
    for coord in [CAF, REALIZATION]:
        assert coord not in [crd.name() for crd in result.coords()]

    # check threshold coordinate units
    assert result.coord(expected_scalar_coord).units == "1"


@pytest.mark.parametrize("model_config", (True, False))
def test_metadata_attributes(multi_cloud_cube, model_config):
    """Test that the metadata attributes in the output cube follows expected
        conventions after that plugin has completed"""
    expected_attributes = UK_ENS_ATTRS.copy()
    if model_config:
        fieldtexture_args = {"model_id_attr": "mosg__model_configuration"}
    else:
        expected_attributes.pop("mosg__model_configuration")
        fieldtexture_args = {}
    plugin = FieldTexture(
        nbhood_radius=NB_RADIUS,
        textural_threshold=TEXT_THRESH,
        diagnostic_threshold=DIAG_THRESH,
        **fieldtexture_args,
    )
    result = plugin.process(multi_cloud_cube)
    assert result.attributes == expected_attributes
