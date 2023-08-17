# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
"""Unit tests for VisibilityCombineCloudBase plugin"""

from datetime import datetime

import iris
import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.synthetic_data.set_up_test_cubes import set_up_probability_cube
from improver.utilities.probability_manipulation import invert_probabilities
from improver.visibility.visibility_combine_cloud_base import VisibilityCombineCloudBase

cloud_name = (
    "cloud_base_assuming_only_consider_cloud_area_fraction_greater_than_4p5_oktas"
)


@pytest.fixture()
def visibility_gridded_cube() -> Cube:
    """
    Sets-up gridded visibility probability cube
    """
    data = np.full((5, 2, 3), dtype=np.float32, fill_value=0.5)
    cube = set_up_probability_cube(
        data,
        thresholds=[50, 1000, 2000, 5000, 7000],
        variable_name="visibility_in_air",
        threshold_units="m",
        spp__relative_to_threshold="less_than",
        time=datetime(2023, 11, 10, 4, 0),
    )
    return cube


@pytest.fixture()
def visibility_spot_cube() -> Cube:
    """
    Sets-up spot visibility probability cube
    """
    data = np.full((5, 2, 3), dtype=np.float32, fill_value=0.5)
    cube = set_up_probability_cube(
        data,
        thresholds=[50, 1000, 2000, 5000, 7000],
        variable_name="visibility_in_air",
        threshold_units="m",
        spp__relative_to_threshold="less_than",
        time=datetime(2023, 11, 10, 4, 0),
    )
    spot_cube = next(cube.slices_over("latitude"))
    spot_cube.remove_coord("latitude")
    spot_cube.coord("longitude").rename("spot_index")
    return spot_cube


@pytest.fixture()
def cloud_base_gridded_cube() -> Cube:
    """
    Sets-up gridded cloud base at ground level probability cube
    """
    data = np.full((2, 2, 3), dtype=np.float32, fill_value=0.4)
    data[0, 0, 0] = 0.7
    cube = set_up_probability_cube(
        data,
        thresholds=[10, 20],
        variable_name=cloud_name,
        threshold_units="m",
        spp__relative_to_threshold="less_than",
        time=datetime(2023, 11, 10, 4, 0),
    )
    cube = cube.extract(
        iris.Constraint(
            cloud_base_assuming_only_consider_cloud_area_fraction_greater_than_4p5_oktas=10
        )
    )
    return cube


@pytest.fixture()
def cloud_base_spot_cube() -> Cube:
    """
    Sets-up a spot cloud base at ground level probability cube
    """
    data = np.full((2, 2, 3), dtype=np.float32, fill_value=0.4)
    data[0, 0, 0] = 0.7
    cube = set_up_probability_cube(
        data,
        thresholds=[10, 20],
        variable_name=cloud_name,
        threshold_units="m",
        spp__relative_to_threshold="less_than",
        time=datetime(2023, 11, 10, 4, 0),
    )
    cube = cube.extract(
        iris.Constraint(
            cloud_base_assuming_only_consider_cloud_area_fraction_greater_than_4p5_oktas=10
        )
    )
    spot_cube = next(cube.slices_over("latitude"))
    spot_cube.remove_coord("latitude")
    spot_cube.coord("longitude").rename("spot_index")
    return spot_cube


@pytest.mark.parametrize("inverted", (True, False))
@pytest.mark.parametrize(
    "cloud_base_cube_name,visibility_cube_name",
    (
        ("cloud_base_spot_cube", "visibility_spot_cube"),
        ("cloud_base_gridded_cube", "visibility_gridded_cube"),
    ),
)
def test_basic(visibility_cube_name, cloud_base_cube_name, request, inverted):
    """Tests the plugin for spot and gridded inputs. Also tests with probabilities
    above or below thresholds"""

    visibility_cube = request.getfixturevalue(visibility_cube_name)
    cloud_base_cube = request.getfixturevalue(cloud_base_cube_name)

    expected_data = np.full_like(visibility_cube.data, fill_value=0.5)
    if "spot" in visibility_cube_name:
        expected_data[1, 0] = 0.585312
        expected_data[2, 0] = 0.66371197
        expected_data[3, 0] = 0.7
        expected_data[4, 0] = 0.7
    else:
        expected_data[1, 0, 0] = 0.585312
        expected_data[2, 0, 0] = 0.66371197
        expected_data[3, 0, 0] = 0.7
        expected_data[4, 0, 0] = 0.7

    if inverted:
        visibility_cube = invert_probabilities(visibility_cube)
        expected_data = 1 - expected_data
        expected_name = "probability_of_visibility_in_air_above_threshold"
    else:
        expected_name = "probability_of_visibility_in_air_below_threshold"

    result = VisibilityCombineCloudBase(
        first_unscaled_threshold=5000, initial_scaling_value=0.6
    )([visibility_cube, cloud_base_cube])
    assert np.allclose(result.data, expected_data)
    assert result.long_name == expected_name


@pytest.mark.parametrize(
    "initial_scaling_value,first_unscaled_threshold,expected_0",
    (
        (0.5, 5000, [0.5, 0.55663997, 0.65463996, 0.7, 0.7]),
        (0.5, 10000, [0.5, 0.5, 0.55663997, 0.67812496, 0.69716495]),
        (0.8, 5000, [0.5655166, 0.64265597, 0.68185604, 0.7, 0.7]),
    ),
)
def test_scaling_parameters(
    cloud_base_spot_cube,
    visibility_spot_cube,
    initial_scaling_value,
    first_unscaled_threshold,
    expected_0,
):
    """Tests plugin with different combinations of distribution constants"""
    expected = np.full_like(visibility_spot_cube.data, 0.5)
    expected[:, 0] = expected_0

    result = VisibilityCombineCloudBase(
        first_unscaled_threshold=first_unscaled_threshold,
        initial_scaling_value=initial_scaling_value,
    )([visibility_spot_cube, cloud_base_spot_cube])
    assert np.allclose(result.data, expected)


@pytest.mark.parametrize("prob_cube", ("visibility", "cloud_base_cube"))
def test_if_probability_cube_error(
    visibility_gridded_cube, cloud_base_gridded_cube, prob_cube
):
    """Tests an error is raised if either the visibility or cloud base at ground
    level cube aren't probability cubes"""

    if prob_cube == "visibility":
        visibility_gridded_cube = visibility_gridded_cube.extract(
            iris.Constraint(visibility_in_air=50)
        )
        visibility_gridded_cube.remove_coord("visibility_in_air")
    elif prob_cube == "cloud_base_cube":
        cloud_base_gridded_cube.remove_coord(cloud_name)

    error_msg = "must be a probability cube"
    with pytest.raises(ValueError, match=error_msg):
        VisibilityCombineCloudBase(
            first_unscaled_threshold=5000, initial_scaling_value=0.6
        )([visibility_gridded_cube, cloud_base_gridded_cube])


def test_too_many_cubes_error(visibility_gridded_cube, cloud_base_gridded_cube):
    """Tests an error is raised if too many cubes are provided"""

    error_msg = "Exactly two cubes should be provided"
    with pytest.raises(ValueError, match=error_msg):
        VisibilityCombineCloudBase(
            first_unscaled_threshold=5000, initial_scaling_value=0.6
        )(
            CubeList(
                [
                    visibility_gridded_cube,
                    cloud_base_gridded_cube,
                    visibility_gridded_cube,
                ]
            )
        )


@pytest.mark.parametrize("cube", ("visibility_gridded_cube", "cloud_base_gridded_cube"))
def test_incorrect_cubes_error(request, cube):
    """Tests an error is raised if a visibility and cloud base cube aren't provided"""

    cube = request.getfixturevalue(cube)
    error_msg = "A visibility and cloud base at ground level cube must be provided"
    with pytest.raises(ValueError, match=error_msg):
        VisibilityCombineCloudBase(
            first_unscaled_threshold=5000, initial_scaling_value=0.6
        )(CubeList([cube, cube]))
