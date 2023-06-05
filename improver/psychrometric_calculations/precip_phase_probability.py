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
"""Module for calculating the probability of specific precipitation phases."""

import operator
from typing import List, Union

import iris
import numpy as np
from cf_units import Unit
from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.cube_checker import spatial_coords_match


class PrecipPhaseProbability(BasePlugin):
    """
    This plugin converts a falling-phase-change-level cube into the
    probability of a specific precipitation phase being found at the surface.

    Consider the case in which a snow-falling-level diagnostic is given as
    input. For a deterministic snow-falling-level diagnostic (i.e. no percentile
    coordinate), these falling levels are compared directly to the altitudes
    of interest to determine the binary probability (0 or 1) of snow at these
    altitudes. A 0 is returned if the altitude is below the snow-falling-level
    and a 1 if the altitude if above the snow-falling-level. If a probabilistic
    snow-falling-level is provided, this plugin will seek to use the 80th
    percentile falling-level to compare with the altitudes. This provides a
    greater level of certainty that the phase is snow, and in conjunction with
    the percentile chosen for rain (see below), provides for a greater depth
    of sleet (mixed phase precipitation).

    The plugin behaves in the same way for a rain-falling-level, but in the
    case of a probabalistic input, the 20th percentile is used. This lowers
    the falling-level within the atmosphere, providing for greater certainty
    that the phase is entirely rain.

    The altitudes used in these comparisons may be provided as gridded
    orographies or as spot forecast ancillaries that contain an altitude
    coordinate. In the former case the phase probability will be determined at
    each grid point at the orographic altitude. In the latter case, the phase
    probability will be determined at each site's specific altitude.
    """

    def _extract_input_cubes(self, cubes: Union[CubeList, List[Cube]]) -> None:
        """
        Separates the input list into the required cubes for this plugin,
        detects whether snow, rain from hail or rain are required from the input
        phase-level cube name, and as required, appropriately extracts the relevant
        percentile.

        Converts units of falling_level_cube to that of orography_cube / site
        altitudes if necessary. Sets flag for snow, rain from hail or rain depending
        on name of falling_level_cube.

        Args:
            cubes:
                Contains cubes of the altitude of the phase-change level (this
                can be snow->sleet, hail->rain or sleet->rain) and the altitude of the
                orography or sites. The name of the phase-change level cube must be
                "altitude_of_snow_falling_level", "altitude_of_rain_from_hail_falling_level" or
                "altitude_of_rain_falling_level". If a gridded orography is provided it must
                be named "surface_altitude". If a spot forecast ancillary is provided it
                must be named "grid_neighbours".

        Raises:
            ValueError: If cubes with the expected names cannot be extracted.
            ValueError: If cubes does not have the expected length of 2.
            ValueError: If a percentile cube does not contain the expected percentiles.
            ValueError: If the extracted cubes do not have matching spatial
                        coordinates.
        """
        if isinstance(cubes, list):
            cubes = iris.cube.CubeList(cubes)
        if len(cubes) != 2:
            raise ValueError(f"Expected 2 cubes, found {len(cubes)}")

        if not spatial_coords_match(cubes):
            raise ValueError(
                "Spatial coords mismatch between " f"{cubes[0]} and " f"{cubes[1]}"
            )

        definitions = {
            "snow": {"comparator": operator.ge, "percentile": 80},
            "rain": {"comparator": operator.lt, "percentile": 20},
            "rain_from_hail": {"comparator": operator.lt, "percentile": 20},
        }

        for diagnostic, definition in definitions.items():
            extracted_cube = cubes.extract(f"altitude_of_{diagnostic}_falling_level")
            if extracted_cube:
                # Once a falling-level cube has been extracted, exit this loop.
                break
        if not extracted_cube:
            raise ValueError(
                "Could not extract a rain, rain from hail or snow falling-level "
                f"cube from {', '.join([cube.name() for cube in cubes])}"
            )
        (self.falling_level_cube,) = extracted_cube
        self.param = diagnostic
        self.comparator = definition["comparator"]
        if self.falling_level_cube.coords("percentile"):
            constraint = iris.Constraint(percentile=definition["percentile"])
            required_percentile = self.falling_level_cube.extract(constraint)
            if not required_percentile:
                raise ValueError(
                    f"Cube {self.falling_level_cube.name()} does not "
                    "contain the required percentile "
                    f"{definition['percentile']}."
                )
            self.falling_level_cube = required_percentile
            self.falling_level_cube.remove_coord("percentile")

        orography_name = "surface_altitude"
        extracted_cube = cubes.extract(orography_name)
        if extracted_cube:
            self.altitudes = extracted_cube[0].data
            altitude_units = extracted_cube[0].units
        elif cubes.extract("grid_neighbours"):
            (extracted_cube,) = cubes.extract("grid_neighbours")
            self.altitudes = extracted_cube.coord("altitude").points
            altitude_units = extracted_cube.coord("altitude").units
        else:
            raise ValueError(
                f"Could not extract {orography_name} cube from "
                f"cube from {', '.join([cube.name() for cube in cubes])}"
            )

        if self.falling_level_cube.units != altitude_units:
            self.falling_level_cube = self.falling_level_cube.copy()
            self.falling_level_cube.convert_units(altitude_units)

    def process(self, cubes: Union[CubeList, List[Cube]]) -> Cube:
        """
        Derives the probability of a precipitation phase at the surface /
        site altitude. If the snow-sleet falling-level is supplied, this is
        the probability of snow at the surface / site altitude. If the sleet-rain
        falling-level is supplied, this is the probability of rain at the surface
        / site altitude. If the hail-rain falling-level is supplied, this is the
        probability of rain from hail at the surface / site altitude.

        Args:
            cubes:
                Contains cubes of the altitude of the phase-change level (this
                can be snow->sleet, hail->rain or sleet->rain) and the altitude
                of the orography or a spot forecast ancillary that defines site
                altitudes.

        Returns:
            Cube containing the probability of a specific precipitation phase
            reaching the surface orography or site altitude. If the
            falling_level_cube was snow->sleet, then this will be the probability
            of snow at the surface. If the falling_level_cube was sleet->rain,
            then this will be the probability of rain from sleet at the surface
            or site altitude. If the falling_level_cube was hail->rain, then this
            will be the probability of rain from hail at the surface or site
            altitude. The probabilities are categorical (1 or 0) allowing
            precipitation to be divided uniquely between snow, sleet and
            rain phases.
        """
        self._extract_input_cubes(cubes)

        result_data = np.where(
            self.comparator(self.altitudes, self.falling_level_cube.data), 1, 0,
        ).astype(np.int8)
        mandatory_attributes = generate_mandatory_attributes([self.falling_level_cube])

        cube = create_new_diagnostic_cube(
            f"probability_of_{self.param}_at_surface",
            Unit("1"),
            self.falling_level_cube,
            mandatory_attributes,
            data=result_data,
        )
        return cube
