# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing the FreezingRain class."""

from typing import Optional, Tuple, Union

import iris
import numpy as np
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError

from improver import PostProcessingPlugin
from improver.metadata.amend import get_unique_attributes
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.common_input_handle import as_cubelist
from improver.utilities.cube_checker import spatial_coords_match
from improver.utilities.cube_extraction import extract_subcube
from improver.utilities.probability_manipulation import to_threshold_inequality


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

        if not spatial_coords_match([self.rain, self.sleet, self.temperature]):
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
        self.temperature = to_threshold_inequality(self.temperature, above=False)

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
                        distinguished by their names.
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

    def _extract_common_realizations(self) -> None:
        """Picks out the realizations that are common to the rain, sleet, and
        temperature cubes.

        Raises:
            ValueError: If the input cubes have no shared realizations.
        """

        def _match_realizations(target):
            constraint = iris.Constraint(realization=common_realizations)
            matched = target.extract(constraint)
            return matched

        cubes = [self.rain, self.sleet, self.temperature]
        # If not working with multi-realization data, return immediately.
        try:
            [cube.coord("realization") for cube in cubes]
        except CoordinateNotFoundError:
            return

        common_realizations = set(cubes[0].coord("realization").points)
        for cube in cubes[1:]:
            common_realizations.intersection_update(cube.coord("realization").points)
        if not common_realizations:
            raise ValueError("Input cubes share no common realizations.")

        del cubes
        self.rain = _match_realizations(self.rain)
        self.sleet = _match_realizations(self.sleet)
        self.temperature = _match_realizations(self.temperature)

    def _generate_template_cube(self, n_realizations: Optional[int]) -> Cube:
        """Generate a freezing rain cube with appropriate coordinates and
        metadata. The sleet cube is used as a basis for this to ensure that
        the lwe (liquid water equivalent) prefix is present in the output cube
        name.

        Args:
            n_realizations:
                The number of realizations if using multi-realization data,
                else None.

        Returns:
            freezing_rain_cube
        """
        if n_realizations is not None:
            template = next(self.sleet.slices_over("realization")).copy()
            template.remove_coord("realization")
        else:
            template = self.sleet.copy()

        diagnostic_name = self.sleet.name().replace("sleet", "freezing_rain")
        threshold_name = (
            self.sleet.coord(var_name="threshold")
            .name()
            .replace("sleet", "freezing_rain")
        )
        mandatory_attributes = generate_mandatory_attributes(
            CubeList([self.rain, self.sleet])
        )
        optional_attributes = {}
        if self.model_id_attr:
            # Rain and sleet will always be derived from the same model, but temperature
            # may be diagnosed from a different model when creating a nowcast forecast.
            # The output in such a case is fundamentally a nowcast product, so we exclude
            # the temperature diagnostic when determining the model_id_attr.
            optional_attributes = get_unique_attributes(
                CubeList([self.rain, self.sleet]), self.model_id_attr
            )
        freezing_rain_cube = create_new_diagnostic_cube(
            diagnostic_name,
            "1",
            template_cube=template,
            mandatory_attributes=mandatory_attributes,
            optional_attributes=optional_attributes,
            data=np.zeros(template.shape).astype(np.float32),
        )
        freezing_rain_cube.coord(var_name="threshold").rename(threshold_name)
        freezing_rain_cube.coord(threshold_name).var_name = "threshold"

        # Adds a cell method only if the time coordinate has bounds, to avoid
        # application to a cube of instantaneous data.
        if freezing_rain_cube.coord("time").has_bounds():
            cell_method = iris.coords.CellMethod("sum", coords="time")
            freezing_rain_cube.add_cell_method(cell_method)

        return freezing_rain_cube

    def _calculate_freezing_rain_probability(
        self, n_realizations: Optional[int]
    ) -> Cube:
        """Calculate the probability of freezing rain from the probabilities
        of rain and sleet rates or accumulations, and the provided probabilities
        of temperature being below the freezing point of water.

        If multiple realizations are present, the contribution of each realization
        is scaled by a (1 / n_realizations) factor to compute the mean across realizations.
        This approach is taken, as opposed to collapsing the realization coordinate later,
        to minimise the memory required.

        (probability of rain + probability of sleet) x (probability T < 0C)

        Args:
            n_realizations:
                The number of realizations if using multi-realization data,
                else None.
        Returns:
            Cube of freezing rain probabilities.
        """
        freezing_rain = self._generate_template_cube(n_realizations)
        if n_realizations is not None:
            rslices = self.rain.slices_over("realization")
            sslices = self.sleet.slices_over("realization")
            tslices = self.temperature.slices_over("realization")
            denominator = n_realizations
        else:
            rslices = [self.rain]
            sslices = [self.sleet]
            tslices = [self.temperature]
            denominator = 1.0

        for rslice, sslice, tslice in zip(rslices, sslices, tslices):
            freezing_rain.data += ((rslice.data + sslice.data) * tslice.data) * (
                1.0 / denominator
            )

        return freezing_rain

    def process(self, *input_cubes: Union[Cube, CubeList]) -> Cube:
        """Check input cubes, then calculate a probability of freezing rain
        diagnostic. Collapses the realization coordinate if present.

        Args:
            input_cubes:
                Contains exactly three cubes, a rain rate or accumulation, a
                sleet rate or accumulation, and an instantaneous or period
                temperature. Accumulations and periods must all represent the
                same length of time.

        Returns:
            Cube of freezing rain probabilties.
        """
        input_cubes = as_cubelist(*input_cubes)
        self._get_input_cubes(input_cubes)

        try:
            n_realizations = len(list(self.rain.slices_over("realization")))
        except CoordinateNotFoundError:
            n_realizations = None

        if n_realizations is not None:
            self._extract_common_realizations()

        freezing_rain_cube = self._calculate_freezing_rain_probability(n_realizations)

        return freezing_rain_cube
