# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for ShowerConditionProbability plugin"""

from typing import Dict, List, Tuple, Union

import numpy as np
import pytest
from iris.cube import CubeList
from numpy import ndarray

from improver.metadata.constants import FLOAT_DTYPE
from improver.precipitation_type.shower_condition_probability import (
    ShowerConditionProbability,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

ATTRIBUTES = {
    "institution": "Met Office",
    "mosg__model_configuration": "gl_ens",
    "source": "Met Office Unified Model",
    "title": "MOGREPS-G Forecast on UK 2 km Standard Grid",
}

EXPECTED_ATTRIBUTES = {
    "institution": "Met Office",
    "source": "Met Office Unified Model",
    "title": "Post-Processed MOGREPS-G Forecast on UK 2 km Standard Grid",
}
MODEL_ID_ATTR_ATTRIBUTES = EXPECTED_ATTRIBUTES.copy()
MODEL_ID_ATTR_ATTRIBUTES.update({"mosg__model_configuration": "gl_ens"})


@pytest.fixture(name="test_cubes")
def cube_fixture(cube_properties: Tuple[Dict[str, Dict[str, Union[List, ndarray]]]]):
    """Create a test cube"""
    cubes = CubeList()
    for name, values in cube_properties.items():
        cubes.append(
            set_up_variable_cube(
                values["data"],
                name=name,
                units=1,
                realizations=values["realizations"],
                attributes=ATTRIBUTES,
            )
        )
    return cubes


@pytest.mark.parametrize(
    "cube_properties, kwargs, expected",
    (
        # Simple case with one realization, cloud dominates returned
        # probabilities (i.e. clear skies).
        (
            {
                "low_and_medium_type_cloud_area_fraction": {
                    "data": np.zeros((2, 2)).astype(FLOAT_DTYPE),
                    "realizations": [0],
                },
                "convective_ratio": {
                    "data": np.zeros((2, 2)).astype(FLOAT_DTYPE),
                    "realizations": [0],
                },
            },
            # Other plugin kwargs
            {"cloud_threshold": 0.5, "convection_threshold": 0.5},
            # Expected result
            (np.ones((2, 2)).astype(FLOAT_DTYPE), EXPECTED_ATTRIBUTES),
        ),
        # As above, but using the model_id_attr keyword to preserve the model
        # information.
        (
            {
                "low_and_medium_type_cloud_area_fraction": {
                    "data": np.zeros((2, 2)).astype(FLOAT_DTYPE),
                    "realizations": [0],
                },
                "convective_ratio": {
                    "data": np.zeros((2, 2)).astype(FLOAT_DTYPE),
                    "realizations": [0],
                },
            },
            # Other plugin kwargs
            {
                "model_id_attr": "mosg__model_configuration",
                "cloud_threshold": 0.5,
                "convection_threshold": 0.5,
            },
            # Expected result
            (np.ones((2, 2)).astype(FLOAT_DTYPE), MODEL_ID_ATTR_ATTRIBUTES),
        ),
        # Simple case with one realization, convection dominates returned
        # probabilities.
        (
            {
                "low_and_medium_type_cloud_area_fraction": {
                    "data": np.ones((2, 2)).astype(FLOAT_DTYPE),
                    "realizations": [0],
                },
                "convective_ratio": {
                    "data": np.ones((2, 2)).astype(FLOAT_DTYPE),
                    "realizations": [0],
                },
            },
            # Other plugin kwargs
            {"cloud_threshold": 0.5, "convection_threshold": 0.5},
            # Expected result
            (np.ones((2, 2)).astype(FLOAT_DTYPE), EXPECTED_ATTRIBUTES),
        ),
        # As above, but the convective_ratio includes masked values. This test
        # checks that they are ignored in setting the resulting probabilities
        # and that the output is not masked. One resulting value differs to the
        # above, corresponding to the masked point.
        (
            {
                "low_and_medium_type_cloud_area_fraction": {
                    "data": np.ones((2, 2)).astype(FLOAT_DTYPE),
                    "realizations": [0],
                },
                "convective_ratio": {
                    "data": np.ma.masked_array(
                        np.ones((2, 2)).astype(FLOAT_DTYPE),
                        mask=np.array([[0, 0], [0, 1]]),
                    ),
                    "realizations": [0],
                },
            },
            # Other plugin kwargs
            {"cloud_threshold": 0.5, "convection_threshold": 0.5},
            # Expected result
            (np.array([[1, 1], [1, 0]]).astype(FLOAT_DTYPE), EXPECTED_ATTRIBUTES),
        ),
        # Multi-realization case with a range of probabilities returned due
        # to variable cloud.
        (
            {
                "low_and_medium_type_cloud_area_fraction": {
                    "data": np.array(
                        [[[0.4, 0.6], [0.4, 0.6]], [[0.6, 0.6], [0.6, 0.6]]]
                    ).astype(FLOAT_DTYPE),
                    "realizations": [0, 1],
                },
                "convective_ratio": {
                    "data": np.zeros((2, 2, 2)).astype(FLOAT_DTYPE),
                    "realizations": [0, 1],
                },
            },
            # Other plugin kwargs
            {"cloud_threshold": 0.5, "convection_threshold": 0.5},
            # Expected result
            (np.array([[0.5, 0], [0.5, 0]]).astype(FLOAT_DTYPE), EXPECTED_ATTRIBUTES),
        ),
        # Same as above, but with different threshold values applied.
        # Cloud =< 0.7, which will result in probabilities all equal to 1.
        (
            {
                "low_and_medium_type_cloud_area_fraction": {
                    "data": np.array(
                        [[[0.4, 0.6], [0.4, 0.6]], [[0.6, 0.6], [0.6, 0.6]]]
                    ).astype(FLOAT_DTYPE),
                    "realizations": [0, 1],
                },
                "convective_ratio": {
                    "data": np.zeros((2, 2, 2)).astype(FLOAT_DTYPE),
                    "realizations": [0, 1],
                },
            },
            # Other plugin kwargs
            {"cloud_threshold": 0.7, "convection_threshold": 0.5},
            # Expected result
            (np.ones((2, 2)).astype(FLOAT_DTYPE), EXPECTED_ATTRIBUTES),
        ),
        # Multi-realization case with cloud and convection both providing a
        # showery probability of 1.
        (
            {
                "low_and_medium_type_cloud_area_fraction": {
                    "data": np.array([[[0, 1], [1, 1]], [[0, 1], [1, 1]]]).astype(
                        FLOAT_DTYPE
                    ),
                    "realizations": [0, 1],
                },
                "convective_ratio": {
                    "data": np.array([[[0, 0], [0, 1]], [[0, 0], [0, 1]]]).astype(
                        FLOAT_DTYPE
                    ),
                    "realizations": [0, 1],
                },
            },
            # Other plugin kwargs
            {"cloud_threshold": 0.5, "convection_threshold": 0.5},
            # Expected result
            (np.array([[1, 0], [0, 1]]).astype(FLOAT_DTYPE), EXPECTED_ATTRIBUTES),
        ),
    ),
)
def test_scenarios(test_cubes, kwargs, expected):
    """Test output type and metadata"""

    expected_shape = test_cubes[0].shape[-2:]
    result = ShowerConditionProbability(**kwargs)(test_cubes)

    assert result.name() == "probability_of_shower_condition_above_threshold"
    assert result.units == "1"
    assert result.shape == expected_shape
    assert result.data.dtype == FLOAT_DTYPE
    assert (result.data == expected[0]).all()
    assert result.attributes == expected[1]
    assert result.coord(var_name="threshold").name() == "shower_condition"
    assert result.coord(var_name="threshold").points == 1.0


def test_incorrect_inputs_exception():
    """Tests that the expected exception is raised for incorrectly named
    input cubes."""
    temperature = set_up_variable_cube(np.ones((2, 2)).astype(FLOAT_DTYPE))
    expected = (
        "A cloud area fraction and convective ratio are required, "
        f"but the inputs were: {temperature.name()}, {temperature.name()}"
    )

    with pytest.raises(ValueError, match=expected):
        ShowerConditionProbability()(CubeList([temperature, temperature]))


def test_mismatched_shape_exception():
    """Tests that the expected exception is raised for cloud and convection
    cubes of different shapes."""
    cloud = set_up_variable_cube(
        np.ones((2, 2)).astype(FLOAT_DTYPE),
        name="low_and_medium_type_cloud_area_fraction",
    )
    convection = set_up_variable_cube(
        np.ones((3, 3)).astype(FLOAT_DTYPE), name="convective_ratio"
    )

    expected = (
        "The cloud area fraction and convective ratio cubes are not the same "
        "shape and cannot be combined to generate a shower probability"
    )

    with pytest.raises(ValueError, match=expected):
        ShowerConditionProbability()(CubeList([cloud, convection]))
