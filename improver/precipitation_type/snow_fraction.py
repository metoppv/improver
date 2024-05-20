# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing the SnowFraction class."""

from typing import Optional, Tuple

import iris
import numpy as np
from iris.cube import Cube, CubeList

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

    def __init__(self, model_id_attr: Optional[str] = None) -> None:
        """
        Initialise the class

        Args:
            model_id_attr:
                Name of the attribute used to identify the source model for
                blending.
        """
        self.model_id_attr = model_id_attr

    def _get_input_cubes(self, input_cubes: CubeList) -> None:
        """
        Separates out the rain and snow cubes from the input list and checks that
            * No other cubes are present
            * Cubes represent the same time quantity (instantaneous or accumulation length)
            * Cubes have compatible units
            * Cubes have same dimensions
            * Cubes are not masked (or are masked with an all-False mask)

        Args:
            input_cubes:
                Contains exactly two cubes, one of rain and one of snow. Both must be
                either rates or accumulations of the same length and of compatible units.

        Raises:
            ValueError:
                If any of the criteria above are not met.
        """
        if len(input_cubes) != 2:
            raise ValueError(
                f"Expected exactly 2 input cubes, found {len(input_cubes)}"
            )
        rain_name, snow_name = self._get_input_cube_names(input_cubes)
        self.rain = input_cubes.extract(rain_name).merge_cube()
        self.snow = input_cubes.extract(snow_name).merge_cube()
        self.snow.convert_units(self.rain.units)
        if not spatial_coords_match([self.rain, self.snow]):
            raise ValueError("Rain and snow cubes are not on the same grid")
        if not self.rain.coord("time") == self.snow.coord("time"):
            raise ValueError("Rain and snow cubes do not have the same time coord")
        if np.ma.is_masked(self.rain.data) or np.ma.is_masked(self.snow.data):
            raise ValueError("Unexpected masked data in input cube(s)")
        if isinstance(self.rain.data, np.ma.masked_array):
            self.rain.data = self.rain.data.data
        if isinstance(self.snow.data, np.ma.masked_array):
            self.snow.data = self.snow.data.data

    @staticmethod
    def _get_input_cube_names(input_cubes: CubeList) -> Tuple[str, str]:
        """
        Identifies the rain and snow cubes from the presence of "rain" or "snow" in
        the cube names.

        Args:
            input_cubes:
                The unsorted rain and snow cubes.

        Returns:
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
                f"Failed to find unique rain and snow cubes from {cube_names}"
            )
        return rain_name, snow_name

    def _calculate_snow_fraction(self) -> Cube:
        """
        Calculates the snow fraction data and interpolates to fill in the missing points.

        Returns:
            Snow fraction cube.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            snow_fraction = self.snow.data / (self.rain.data + self.snow.data)
        snow_fraction_cube = create_new_diagnostic_cube(
            "snow_fraction",
            "1",
            template_cube=self.rain,
            mandatory_attributes=generate_mandatory_attributes(
                iris.cube.CubeList([self.rain, self.snow]),
                model_id_attr=self.model_id_attr,
            ),
            data=snow_fraction,
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

    def process(self, input_cubes: CubeList) -> Cube:
        """Check input cubes, then calculate and interpolate a snow fraction cube.

        Args:
            input_cubes:
                Contains cubes of rain and snow, both must be either rates or accumulations.

        Returns:
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
