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
"""Module containing the SnowFraction class."""


import iris

from improver import PostProcessingPlugin
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.cube_checker import spatial_coords_match
from improver.utilities.interpolation import interpolate_missing_data


class SnowFraction(PostProcessingPlugin):
    """
    Calculates a snow-fraction field from fields of snow and rain (rate or
    accumulation). Where no precipitation is present, the data are filled in from
    the nearest precipitating point.

    snow_fraction = snow / (snow + rain)
    """

    def _get_input_cubes(self, input_cubes):
        """
        Separates out the rain and snow cubes from the input list and checks that
            * No other cubes are present
            * Cubes represent the same time quantity (instantaneous or accumulation length)
            * Cubes have compatible units
            * Cubes have same dimensions

        Args:
            input_cubes (iris.cube.CubeList):
                Contains exactly two cubes, one of rain and one of snow. Both either
                rates or accumulations of the same length and of compatible units.

        Returns:
            None

        Raises:
            ValueError:
                If any of the criteria above are not met.

        """
        if len(input_cubes) != 2:
            raise ValueError(
                f"Expected exactly two input cubes, found {len(input_cubes)}"
            )
        rain_name, snow_name = self._get_input_cube_names(input_cubes)
        self.rain = input_cubes.extract(rain_name).merge_cube()
        self.snow = input_cubes.extract(snow_name).merge_cube()
        self.snow.convert_units(self.rain.units)
        if not spatial_coords_match(self.rain, self.snow):
            raise ValueError("Rain and Snow cubes are not on the same grid")
        if not self.rain.coord("time") == self.snow.coord("time"):
            raise ValueError("Rain and Snow cubes do not have the same time coord")

    @staticmethod
    def _get_input_cube_names(input_cubes):
        """
        Identifies the rain and snow cubes from the presence of "rain" or "snow" in
        the cube names.

        Args:
            input_cubes (iris.cube.CubeList):
                The unsorted rain and snow cubes.

        Returns:
            tuple:
                rain_name and snow_name, in that order.

        """
        cube_names = [cube.name() for cube in input_cubes]
        try:
            (rain_name,) = [x for x in cube_names if "rain" in x]
            (snow_name,) = [x for x in cube_names if "snow" in x]
        except ValueError:
            raise ValueError(f"Could not find both rain and snow in {cube_names}")
        if rain_name == snow_name:
            raise ValueError(
                f"Failed to find unique Rain and Snow cubes from {cube_names}"
            )
        return rain_name, snow_name

    def _calculate_snow_fraction(self):
        """
        Calculates the snow fraction data and interpolates to fill in the missing points

        Returns:
            iris.cube.Cube:
                Snow fraction cube

        """
        snow_fraction_cube = create_new_diagnostic_cube(
            "snow_fraction",
            "1",
            template_cube=self.rain,
            mandatory_attributes=generate_mandatory_attributes(
                iris.cube.CubeList([self.rain, self.snow])
            ),
            data=self.snow.data / (self.rain.data + self.snow.data),
        )

        spatial_dims = [snow_fraction_cube.coord(axis=n).name() for n in ["y", "x"]]
        snow_fraction_interpolated = iris.cube.CubeList()
        for snow_fraction_slice in snow_fraction_cube.slices(spatial_dims):
            snow_fraction_interpolated.append(
                snow_fraction_slice.copy(
                    interpolate_missing_data(snow_fraction_slice.data, method="nearest")
                )
            )
        return snow_fraction_interpolated.merge_cube()

    def process(self, input_cubes):
        """Check input cubes, then calculate and interpolate a snow fraction cube.

        Args:
            input_cubes (iris.cube.CubeList):
                Contains cubes of rain and snow, both must be either rates or accumulations.

        Returns:
            iris.cube.Cube:
                Cube of snow-fraction. The data within this
                cube will contain values between 0 and 1. Points where no precipitation
                is present will be filled using a nearest-neighbour interpolation.

                The cube meta-data will contain:
                * Input_cube name "snow_fraction"
                * Cube units set to (1).

        Raises:
            ValueError: if input cubes fail any comparison tests.

        """
        self._get_input_cubes(input_cubes)

        return self._calculate_snow_fraction()
