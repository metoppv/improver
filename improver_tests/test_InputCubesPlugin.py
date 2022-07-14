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
"""Tests for the InputCubesPlugin base class and CubeDescriptor dataclass"""
import re
from datetime import datetime
from typing import List

import numpy as np
import pytest
from iris.cube import Cube

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.base_classes import InputCubesPlugin
from improver.utilities.cube_descriptor import CubeDescriptor


@pytest.mark.parametrize(
    "kwargs, msg",
    (
        ({"name": "invalid"}, r".* missing 1 required positional argument: 'units'"),
        (
            {"name": "air_temperature", "units": "K", "kittens": "cute"},
            r".* got an unexpected keyword argument 'kittens'",
        ),
        (
            {"name": "air_temperature", "units": 1},
            r"The field 'units' is <class 'int'> instead of <class 'str'>",
        ),
    ),
)
def test_bad_descriptors(kwargs, msg):
    """Test for known errors when a bad CubeDescriptor is declared"""
    with pytest.raises(TypeError, match=msg):
        CubeDescriptor(**kwargs)


@pytest.fixture(name="cubes")
def cubes_fixture(time_bounds) -> List[Cube]:
    """Set up matching r, y, x cubes matching Plugin requirements"""
    cubes = []
    data = np.ones((2, 3, 4), dtype=np.float32)
    kwargs = {}
    if time_bounds:
        kwargs["time_bounds"] = (
            datetime(2017, 11, 10, 3, 0),
            datetime(2017, 11, 10, 4, 0),
        )
    cube = set_up_variable_cube(data, **kwargs,)
    for descriptor in SimplePlugin.cube_descriptors.values():
        cube = cube.copy()
        if descriptor.partial_name:
            cube.rename(f"surface_{descriptor.name}")
        else:
            cube.rename(descriptor.name)
        cube.units = descriptor.units
        cubes.append(cube)
    return cubes


class SimplePlugin(InputCubesPlugin):
    """
    A simple implementation of the InputCubesPlugin.
    Sets just the cube_descriptors and expected cube attributes and contains
    a process method that just calls the parse_inputs method on the base class.
    """

    cube_descriptors = {
        "temperature": CubeDescriptor(name="air_temperature", units="K"),
        "pressure": CubeDescriptor(name="air_pressure", units="Pa", partial_name=True),
        "rel_humidity": CubeDescriptor(name="relative_humidity", units="kg kg-1"),
    }

    def process(self, inputs: List[Cube], time_bounds: bool = False):
        """Call parse_inputs class method."""
        self.parse_inputs(inputs, time_bounds=time_bounds)


def add_int_key(descriptor: dict):
    """Adds a descriptor key that fails the string test"""
    descriptor[0] = CubeDescriptor(name="invalid", units="kg")


def replace_with_dict(descriptor: dict):
    """Replaces a CubeDescriptor with a dict"""
    descriptor["temperature"] = {"name": "air_temperature", "units": "K"}


def empty_dict(descriptor: dict):
    """Empties the descriptor dictionary"""
    for key in list(descriptor.keys()):
        descriptor.pop(key)


@pytest.mark.parametrize(
    "breaking_function, error_type, msg",
    (
        (
            add_int_key,
            TypeError,
            "Keys in cube_descriptors must be 'str', not <class 'int'> for 0",
        ),
        (
            replace_with_dict,
            TypeError,
            "Values in cube_descriptors must be <CubeDescriptor>, not <class 'dict'> for",
        ),
        (empty_dict, ValueError, "Missing compulsory dictionary 'cube_descriptors'"),
    ),
)
def test_bad_descriptor_keys(breaking_function, error_type, msg):
    """Test for known errors when a bad descriptor is provided"""
    bad_descriptor = {
        "temperature": CubeDescriptor(name="air_temperature", units="K"),
        "pressure": CubeDescriptor(name="air_pressure", units="Pa"),
        "rel_humidity": CubeDescriptor(name="relative_humidity", units="kg kg-1"),
    }
    breaking_function(bad_descriptor)
    with pytest.raises(error_type, match=msg):

        class BadPlugin(InputCubesPlugin):
            cube_descriptors = bad_descriptor

            def process(self, inputs: List[Cube], time_bounds: bool = False):
                """Empty method to make abstract base class happy"""
                pass

        bp = BadPlugin()
        bp.cube_descriptors = bad_descriptor


def test_missing_descriptor():
    """Test for known errors when no descriptor is provided"""
    with pytest.raises(
        TypeError,
        match="Can't instantiate abstract class BadPlugin with abstract method",
    ):

        class BadPlugin(InputCubesPlugin):
            def process(self, inputs: List[Cube], time_bounds: bool = False):
                """Empty method to make abstract base class happy"""
                pass

        BadPlugin()


def test_calling_abstract_method():
    """Test for known error when calling the abstract method"""

    class BadPlugin(InputCubesPlugin):
        cube_descriptors = {
            "temperature": CubeDescriptor(name="air_temperature", units="K"),
            "pressure": CubeDescriptor(name="air_pressure", units="Pa"),
            "rel_humidity": CubeDescriptor(name="relative_humidity", units="kg kg-1"),
        }

        def process(self):
            """Uses super to access the abstract method"""
            super().cube_descriptors()

    with pytest.raises(NotImplementedError):
        BadPlugin()()


def metadata_ok(plugin):
    """Checks that the three cubes are in the right places with the right names and units"""
    assert plugin.get_cube("temperature").name() == "air_temperature"
    assert plugin.get_cube("temperature").units == "K"
    assert plugin.get_cube("pressure").name() == "surface_air_pressure"
    assert plugin.get_cube("pressure").units == "Pa"
    assert plugin.get_cube("rel_humidity").name() == "relative_humidity"
    assert plugin.get_cube("rel_humidity").units == "kg kg-1"


@pytest.mark.parametrize("time_bounds", (False, True))
@pytest.mark.parametrize("change_units", (False, True))
@pytest.mark.parametrize("reverse_order", (False, True))
def test_unit_conversion_and_cube_order(
    cubes, reverse_order, change_units, time_bounds
):
    """Check that we get the right behaviour regardless of cube list order, input cube units,
    with and without time bounds."""
    if change_units:
        cubes[0].units = "Celsius"
        cubes[1].units = "hPa"
        cubes[2].units = "g kg-1"
    if reverse_order:
        cubes.reverse()
    plugin = SimplePlugin()
    plugin(cubes, time_bounds=time_bounds)
    metadata_ok(plugin)
    if change_units:
        assert np.allclose(plugin.get_cube("temperature").data, 274.15)
        assert np.allclose(plugin.get_cube("pressure").data, 100.0)
        assert np.allclose(plugin.get_cube("rel_humidity").data, 0.001)
    else:
        assert np.allclose(plugin.get_cube("temperature").data, 1.0)
        assert np.allclose(plugin.get_cube("pressure").data, 1.0)
        assert np.allclose(plugin.get_cube("rel_humidity").data, 1.0)


@pytest.mark.parametrize("time_bounds", [False])
@pytest.mark.parametrize("model_id_attr", ("mosg__model_configuration", None))
def test_model_id_attr(cubes, model_id_attr):
    """Check that tests pass if model_id_attr is set on inputs and is applied or not"""
    for cube in cubes:
        cube.attributes["mosg__model_configuration"] = "gl_ens"
    plugin = SimplePlugin(model_id_attr=model_id_attr)
    plugin(cubes)
    metadata_ok(plugin)
    if model_id_attr:
        assert plugin.model_id_attr == "mosg__model_configuration"
        assert plugin.model_id_value == "gl_ens"
    else:
        assert plugin.model_id_attr is None
        assert plugin.model_id_value is None


def remove_a_cube(cubes: List[Cube]):
    """Removes the last cube from the cube list."""
    del cubes[-1]


def add_unexpected_cube(cubes: List[Cube]):
    """Copies first cube, renames it and adds it to the list of cubes."""
    cube = cubes[0].copy()
    cube.rename("kittens")
    cubes.append(cube)


def spatial_shift(cubes: List[Cube]):
    """Adjusts x-coord of first cube so it no longer matches the second cube."""
    coord = cubes[0].coord(axis="x")
    coord = coord.copy(coord.points + 1)
    cubes[0].replace_coord(coord)


def units_to_kg(cube: Cube):
    """Sets units of cube to kg"""
    cube.units = "kg"


def inconsistent_time_bounds(cubes: List[Cube]):
    """Adds time bounds only to the first cube"""
    time_point = cubes[0].coord("time").points[0]
    cubes[0].coord("time").bounds = (time_point - 3600, time_point)


def inconsistent_time_point(cubes: List[Cube]):
    """Moves time point of first cube back by one hour"""
    cubes[0].coord("time").points = cubes[0].coord("time").points - 3600


def inconsistent_frt(cubes: List[Cube]):
    """Moves forecast_reference_time point of first cube back by one hour"""
    cubes[0].coord("forecast_reference_time").points = (
        cubes[0].coord("forecast_reference_time").points - 3600
    )


def remove_one_time_bounds(cubes: List[Cube]):
    """Removes time bounds from first cube"""
    cubes[0].coord("time").bounds = None


def remove_two_time_bounds(cubes: List[Cube]):
    """Removes time bounds from first two cubes"""
    cubes[0].coord("time").bounds = None
    cubes[1].coord("time").bounds = None


def set_mismatched_model_ids(cubes: List[Cube]):
    """Sets model_id of first cube to be different"""
    cubes[0].attributes["mosg__model_configuration"] = "uk_ens"


@pytest.mark.parametrize(
    "modifier, time_bounds, error_match",
    (
        (lambda l: l[0].rename("kittens"), False, "Expected to find cube of "),
        (lambda l: l[1].rename("poodles"), False, "Expected to find cube of "),
        (remove_a_cube, False, "Expected to find cube of "),
        (
            add_unexpected_cube,
            False,
            re.escape("Unexpected Cube(s) found in inputs: "),
        ),
        (spatial_shift, False, "Spatial coords of input Cubes do not match: "),
        (lambda l: units_to_kg(l[0]), False, "Unable to convert from"),
        (lambda l: units_to_kg(l[1]), False, "Unable to convert from"),
        (inconsistent_time_bounds, False, "air_temperature must not have time bounds"),
        (inconsistent_time_point, False, "time coordinates do not match."),
        (inconsistent_frt, False, "forecast_reference_time coordinates do not match."),
        (remove_one_time_bounds, True, "air_temperature must have time bounds"),
        (
            remove_two_time_bounds,
            True,
            "air_temperature and surface_air_pressure must have time bounds",
        ),
        (
            set_mismatched_model_ids,
            False,
            "Attribute mosg__model_configuration does not match on input cubes",
        ),
    ),
)
def test_exceptions(cubes, modifier: callable, time_bounds: bool, error_match: str):
    """Check for things we know we should reject"""
    for cube in cubes:
        cube.attributes["mosg__model_configuration"] = "gl_ens"
    modifier(cubes)
    with pytest.raises(ValueError, match=error_match):
        SimplePlugin(model_id_attr="mosg__model_configuration")(
            cubes, time_bounds=time_bounds
        )


@pytest.mark.parametrize("time_bounds", [False])
def test_get_cube_error(cubes):
    """Checks that a non-cube inserted into the object raises an error"""
    plugin = SimplePlugin()
    plugin(cubes)
    assert isinstance(plugin.get_cube("temperature"), Cube)
    plugin._temperature = 1
    with pytest.raises(
        TypeError, match="_temperature should be a Cube, but found <class 'int'>"
    ):
        plugin.get_cube("temperature")
