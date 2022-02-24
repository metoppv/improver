# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""Module containing the FreezingRain class."""

from typing import Optional, Tuple

from iris.cube import Cube, CubeList

from improver import PostProcessingPlugin
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.cube_checker import spatial_coords_match
from improver.utilities.cube_extraction import extract_subcube
from improver.utilities.probability_manipulation import (
    comparison_operator_dict,
    invert_probabilities,
)


class FreezingRain(PostProcessingPlugin):
    """
    Calculates a probability of freezing rain using rain, sleet and temperature
    probabilities.
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
        Separates out the rain, sleet, and temperature cubes, checking that:
            * No other cubes are present
            * Cubes have same dimensions
            * Cubes represent the same time quantity (instantaneous or accumulation length)
            * Precipitation cube threshold units are compatible
            * Precipitation cubes have the same set of thresholds
            * A 273.15K (0 Celsius) temperature threshold is available

        The temperature cube is also modified if necessary to return probabilties
        below threshold values. This data is then thinned to return only the
        probabilities of temperature being below the freezing point of water,
        0 Celsius.

        Args:
            input_cubes:
                Contains exactly three cubes, a rain rate or accumulation, a
                sleet rate or accumulation, and an instantaneous or period
                temperature. Accumulations and periods must all represent the
                same length of time.

        Raises:
            ValueError:
                If any of the criteria above are not met.
        """
        if len(input_cubes) != 3:
            raise ValueError(
                f"Expected exactly 3 input cubes, found {len(input_cubes)}"
            )
        rain_name, sleet_name, temperature_name = self._get_input_cube_names(
            input_cubes
        )
        (self.rain,) = input_cubes.extract(rain_name)
        (self.sleet,) = input_cubes.extract(sleet_name)
        (self.temperature,) = input_cubes.extract(temperature_name)

        if not spatial_coords_match(self.rain, self.sleet):
            raise ValueError("Input cubes are not on the same grid")
        if (
            not self.rain.coord("time")
            == self.sleet.coord("time")
            == self.temperature.coord("time")
        ):
            raise ValueError("Input cubes do not have the same time coord")

        # Ensure rain and sleet cubes are compatible
        rain_threshold = self.rain.coord(var_name="threshold")
        sleet_threshold = self.sleet.coord(var_name="threshold")
        try:
            sleet_threshold.convert_units(rain_threshold.units)
        except ValueError:
            raise ValueError("Rain and sleet cubes have incompatible units")

        if not all(rain_threshold.points == sleet_threshold.points):
            raise ValueError("Rain and sleet cubes have different threshold values")

        # Ensure probabilities relate to temperatures below a threshold
        temperature_threshold = self.temperature.coord(var_name="threshold")
        inequality = temperature_threshold.attributes["spp__relative_to_threshold"]
        spp_lookup = comparison_operator_dict()
        greater_attr = [spp_lookup[ineq]["spp_string"] for ineq in ["ge", "gt"]]
        if inequality in greater_attr:
            self.temperature = invert_probabilities(self.temperature)

        # Simplify the temperature cube to the critical threshold of 273.15K,
        # the freezing point of water under typical pressures.
        self.temperature = extract_subcube(
            self.temperature, [f"{temperature_threshold.name()}=273.15"], units=["K"]
        )
        if self.temperature is None:
            raise ValueError(
                "No 0 Celsius or equivalent threshold is available "
                "in the temperature data"
            )

    @staticmethod
    def _get_input_cube_names(input_cubes: CubeList) -> Tuple[str, str, str]:
        """
        Identifies the rain, sleet, and temperature cubes from the diagnostic
        names.

        Args:
            input_cubes:
                The unsorted rain, sleet, and temperature cubes.

        Returns:
            rain_name, sleet_name, and temperature_name in that order.

        Raises:
            ValueError: If two input cubes have the same name.
            ValueError: If rain, sleet, and temperature cubes cannot be
                        distinquished by their names.
        """
        cube_names = [cube.name() for cube in input_cubes]
        if not sorted(list(set(cube_names))) == sorted(cube_names):
            raise ValueError(
                "Duplicate input cube provided. Unable to find unique rain, "
                f"sleet, and temperature cubes from {cube_names}"
            )

        try:
            (rain_name,) = [x for x in cube_names if "rain" in x]
            (sleet_name,) = [x for x in cube_names if "sleet" in x]
            (temperature_name,) = [x for x in cube_names if "temperature" in x]
        except ValueError:
            raise ValueError(
                "Could not find unique rain, sleet, and temperature diagnostics"
                f"in {cube_names}"
            )
        return rain_name, sleet_name, temperature_name

    def _calculate_freezing_rain_probability(self) -> Cube:
        """Calculate the probability of freezing rain from the probabilities
        of rain and sleet rates or accumulations, and the provided probabilities
        of temperature being below the freezing point of water.

        (probability of rain + probability of sleet) x (probability T < 0C)

        Returns:
            Cube of freezing rain probabilities.
        """

        freezing_rain_prob = (self.rain.data + self.sleet.data) * self.temperature.data
        diagnostic_name = self.sleet.name().replace("sleet", "freezing_rain")
        threshold_name = (
            self.sleet.coord(var_name="threshold")
            .name()
            .replace("sleet", "freezing_rain")
        )
        freezing_rain_cube = create_new_diagnostic_cube(
            diagnostic_name,
            "1",
            template_cube=self.sleet,
            mandatory_attributes=generate_mandatory_attributes(
                CubeList([self.rain, self.sleet]), model_id_attr=self.model_id_attr,
            ),
            data=freezing_rain_prob,
        )
        freezing_rain_cube.coord(var_name="threshold").rename(threshold_name)
        freezing_rain_cube.coord(threshold_name).var_name = "threshold"
        return freezing_rain_cube

    def process(self, input_cubes: CubeList) -> Cube:
        """Check input cubes, then calculate a probability of freezing rain
        diagnostic.

        Args:
            input_cubes:
                Contains exactly three cubes, a rain rate or accumulation, a
                sleet rate or accumulation, and an instantaneous or period
                temperature. Accumulations and periods must all represent the
                same length of time.

        Returns:
            Cube of freezing rain probabilties.
        """
        self._get_input_cubes(input_cubes)
        return self._calculate_freezing_rain_probability()
