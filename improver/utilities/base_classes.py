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


class InputCubesPlugin(BasePlugin, ABC):
    """An abstract class for IMPROVER plugins that generate a new diagnostic.
    """

    def __init__(self, model_id_attr: str = None):
        """
        Set up class

        Args:
            model_id_attr:
                Name of model ID attribute to be copied from source cubes to output cube
        """
        self.model_id_attr = model_id_attr
        self.model_id_value = None

    @classmethod
    @property
    @abstractmethod
    def cube_descriptors(cls):
        """Classes must set this to a dict where each key will become the class attribute name
        and the value is another dict containing "name" for the required cube name and "units"
        for the required cube units, which the discovered cube will be converted to."""
        raise NotImplementedError

    @staticmethod
    def assert_time_coords_ok(inputs: List[Cube], time_bounds: bool):
        """
        Raises appropriate ValueError if

        - Any input cube has or is missing time bounds (depending on time_bounds)
        - Input cube times and forecast_reference_times do not match
        """
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
        for attr, cube_values in self.cube_descriptors.items():
            try:
                (cube,) = cubes.extract(cube_values["name"])
            except ValueError as e:
                raise ValueError(
                    f"Expected to find cube of {cube_values['name']},"
                    f" in {[c.name() for c in cubes]}"
                ) from e
            cube.convert_units(cube_values["units"])
            setattr(self, attr, cube)
        if len(cubes) > len(self.cube_descriptors):
            expected_names = [i["name"] for i in self.cube_descriptors.values()]
            extras = [c.name() for c in cubes if c.name() not in expected_names]
            raise ValueError(f"Unexpected Cube(s) found in inputs: {extras}")
        if not spatial_coords_match(inputs):
            raise ValueError(f"Spatial coords of input Cubes do not match: {cubes}")
        self.assert_time_coords_ok(cubes, time_bounds)
        if self.model_id_attr:
            model_id_value = {cube.attributes[self.model_id_attr] for cube in inputs}
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
