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
"""Unit tests for the "cube_manipulation.normalise_to_reference" function."""
import pytest
import numpy as np
import iris
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import normalise_to_reference


@pytest.fixture
def shape():
    output = (2, 3, 3)
    return output


@pytest.fixture
def input_cubes(shape):
    rain_data = 0.5 * np.ones(shape, dtype=np.float32)
    sleet_data = 0.4 * np.ones(shape, dtype=np.float32)
    snow_data = 0.1 * np.ones(shape, dtype=np.float32)

    rain_cube = set_up_variable_cube(rain_data, name="rainfall_rate", units="m s-1")
    sleet_cube = set_up_variable_cube(sleet_data, name="lwe_sleetfall_rate", units="m s-1")
    snow_cube = set_up_variable_cube(snow_data, name="lwe_showfall_rate", units="m s-1")

    return iris.cube.CubeList([rain_cube, sleet_cube, snow_cube])


@pytest.fixture
def reference_cube(shape):
    precip_data = 2 * np.ones(shape, dtype=np.float32)

    return set_up_variable_cube(precip_data, name="lwe_precipitation_rate", units="m s-1")


@pytest.fixture()
def expected_cubes(shape):
    rain_data = 2 * 0.5 * np.ones(shape, dtype=np.float32)
    sleet_data = 2 * 0.4 * np.ones(shape, dtype=np.float32)
    snow_data = 2 * 0.1 * np.ones(shape, dtype=np.float32)

    rain_cube = set_up_variable_cube(rain_data, name="rainfall_rate", units="m s-1")
    sleet_cube = set_up_variable_cube(sleet_data, name="lwe_sleetfall_rate", units="m s-1")
    snow_cube = set_up_variable_cube(snow_data, name="lwe_showfall_rate", units="m s-1")

    return iris.cube.CubeList([rain_cube, sleet_cube, snow_cube])


def test_basic(input_cubes, reference_cube, expected_cubes):
    output = normalise_to_reference(input_cubes, reference_cube)

    assert output == expected_cubes


def test_zero_total(input_cubes, reference_cube, expected_cubes):
    for cube in input_cubes:
        cube.data[0, :, :] = 0
    for cube in expected_cubes:
        cube.data[0, :, :] = reference_cube.data[0, :, :] / 3

    output = normalise_to_reference(input_cubes, reference_cube)

    assert output == expected_cubes


def test_single_input_cube(input_cubes, reference_cube, expected_cubes):
    input_cube = input_cubes[0]
    output = normalise_to_reference(iris.cube.CubeList([input_cube]), reference_cube)

    # check that metadata is as expected
    assert input_cube == output[0].copy(data=input_cube.data)
    # check that data is as expected
    assert np.array_equal(output[0].data, reference_cube.data)
