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
"""Module containing plugin base classes."""
from abc import ABC, abstractmethod
from typing import List

from iris.cube import Cube, CubeList

import improver.utilities.cube_descriptor
from improver.utilities.cube_checker import spatial_coords_match


class BasePlugin(ABC):
    """An abstract class for IMPROVER plugins.
    Subclasses must be callable. We preserve the process
    method by redirecting to __call__.
    """

    def __call__(self, *args, **kwargs):
        """Makes subclasses callable to use process
        Args:
            *args:
                Positional arguments.
            **kwargs:
                Keyword arguments.
        Returns:
            Output of self.process()
        """
        return self.process(*args, **kwargs)

    @abstractmethod
    def process(self, *args, **kwargs):
        """Abstract class for rest to implement."""
        pass


class InputCubesPlugin(BasePlugin):
    """
    An abstract class for IMPROVER plugins that generate a new diagnostic.

    Sub-classes must set:
        cube_descriptors: Dict. Each key will be used with get_cube
        and the value is a CubeDescriptor describing the required cube name and units
        which the discovered cube will be converted to.
    """

    def __init__(self, model_id_attr: str = None):
        """
        Set up class

        Args:
            model_id_attr:
                Name of model ID attribute to be copied from source cubes to output cube
        """
        self._parse_cube_descriptors()

        self.model_id_attr = model_id_attr
        self.model_id_value = None

    def _parse_cube_descriptors(self):
        """Checks the cube_descriptors dict"""
        if not self.cube_descriptors:
            raise ValueError("Missing compulsory dictionary 'cube_descriptors'")
        for k, v in self.cube_descriptors.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"Keys in cube_descriptors must be 'str', not {type(k)} for {k}"
                )
            if not isinstance(v, improver.utilities.cube_descriptor.CubeDescriptor):
                raise TypeError(
                    f"Values in cube_descriptors must be <CubeDescriptor>, not {type(v)} for {k}"
                )

    @property
    @abstractmethod
    def cube_descriptors(self):
        """Classes must set this to a dict where each key will be used with get_cube
        and the value is a CubeDescriptor describing the required cube name and units
        which the discovered cube will be converted to."""
        raise NotImplementedError

    def get_cube(self, key: str) -> Cube:
        """Gets the named cube.

        Args:
            key:
                The cube identifier. Must match a key from cube_descriptors
        """
        cube = getattr(self, f"_{key}")
        if not isinstance(cube, Cube):
            raise TypeError(f"_{key} should be a Cube, but found {type(cube)}")
        return cube

    @staticmethod
    def assert_time_coords_ok(inputs: List[Cube], time_bounds: bool):
        """
        Raises appropriate ValueError if

        - Any input cube has or is missing time bounds (depending on time_bounds)
        - Input cube times and forecast_reference_times do not match

        Can be overloaded where only a subset of inputs are expected to match, or have
        specific offsets. Overloading functions can use self.get_cube()
        """
        if len(inputs) <= 1:
            return
        cubes_not_matching_time_bounds = [
            c.name() for c in inputs if c.coord("time").has_bounds() != time_bounds
        ]
        if cubes_not_matching_time_bounds:
            str_bool = "" if time_bounds else "not "
            msg = f"{' and '.join(cubes_not_matching_time_bounds)} must {str_bool}have time bounds"
            raise ValueError(msg)
        for time_coord_name in ["time", "forecast_reference_time"]:
            time_coords = [c.coord(time_coord_name) for c in inputs]
            if not all([tc == time_coords[0] for tc in time_coords[1:]]):
                raise ValueError(
                    f"{time_coord_name} coordinates do not match."
                    "\n  "
                    "\n  ".join(
                        [str(c.name()) + ": " + str(c.coord("time")) for c in inputs]
                    )
                )

    @staticmethod
    def assert_spatial_coords_ok(spatial_matching_cubes: List[Cube]):
        """
        Raises appropriate ValueError if

        - if the x and y coords are not exactly the same to the
            precision of the floating-point values (this should be true for
            any cubes derived using cube.regrid())

        Can be overloaded where only a subset of inputs are expected to match.
        Overloading functions can use self.get_cube()
        """
        if len(spatial_matching_cubes) <= 1:
            return
        if not spatial_coords_match(spatial_matching_cubes):
            raise ValueError(
                f"Spatial coords of input Cubes do not match: {spatial_matching_cubes}"
            )

    def parse_inputs(self, inputs: List[Cube], time_bounds=False) -> None:
        """Extracts input cubes as described by self.cube_descriptors.

        Args:
            inputs:
                List of Cubes containing exactly one of each input cube.
            time_bounds:
                True when all input cubes are expected to hate time bounds.
        Raises:
            ValueError:
                If additional cubes are found
        """
        cubes = CubeList(inputs)
        for desc in self.cube_descriptors.values():
            desc._matched_name = desc.name
            if desc.partial_name:
                # Replace descriptor name with any cube name that contains the partial name
                try:
                    (desc._matched_name,) = [
                        c.name() for c in cubes if desc.name in c.name()
                    ]
                except ValueError:
                    pass  # This is picked up with a better error message later
        expected_names = set(
            [desc._matched_name for desc in self.cube_descriptors.values()]
        )
        cubes_names = set([cube.name() for cube in cubes])
        diff = expected_names - cubes_names
        if diff:
            raise ValueError(f"Expected to find cube of {diff}, in {cubes_names}")
        diff = cubes_names - expected_names
        if diff:
            raise ValueError(f"Unexpected Cube(s) found in inputs: {diff}")

        spatial_matching_cubes = []
        for attr, cube_values in self.cube_descriptors.items():
            (cube,) = cubes.extract(cube_values._matched_name)
            cube.convert_units(cube_values.units)
            setattr(self, f"_{attr}", cube)
            if cube_values.spatial_match:
                spatial_matching_cubes.append(cube)

        self.assert_spatial_coords_ok(spatial_matching_cubes)
        self.assert_time_coords_ok(cubes, time_bounds)

        if self.model_id_attr:
            model_id_value = {cube.attributes[self.model_id_attr] for cube in cubes}
            if len(model_id_value) != 1:
                raise ValueError(
                    f"Attribute {self.model_id_attr} does not match on input cubes. "
                    f"{' != '.join(model_id_value)}"
                )
            (self.model_id_value,) = model_id_value


class PostProcessingPlugin(BasePlugin, ABC):
    """An abstract class for IMPROVER post-processing plugins.
    Makes generalised changes to metadata relating to post-processing.
    """

    def __call__(self, *args, **kwargs):
        """Makes subclasses callable to use process
        Args:
            *args:
                Positional arguments.
            **kwargs:
                Keyword arguments.

        Returns:
            Output of self.process() with updated title attribute
        """
        cube = super().__call__(*args, **kwargs)
        self.post_processed_title(cube)
        return cube

    @staticmethod
    def post_processed_title(cube):
        """Updates title attribute on output cube to include
        "Post-Processed"
        """
        from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS

        default_title = MANDATORY_ATTRIBUTE_DEFAULTS["title"]
        if (
            "title" in cube.attributes.keys()
            and cube.attributes["title"] != default_title
            and "Post-Processed" not in cube.attributes["title"]
        ):
            title = cube.attributes["title"]
            cube.attributes["title"] = f"Post-Processed {title}"
