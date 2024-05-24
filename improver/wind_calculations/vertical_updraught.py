# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""This module contains the VerticalUpdraught plugin"""

from typing import List

import numpy as np
from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.cube_checker import spatial_coords_match


class VerticalUpdraught(BasePlugin):
    """
    Methods to calculate the maximum vertical updraught from CAPE and precipitation rate as
    defined in Hand (2002) and Golding (1998) with the precipitation rate modifier found in
    the UKPP CDP code.

    Hand, W. 2002. "The Met Office Convection Diagnosis Scheme." Meteorological Applications
        9(1): 69-83. doi:10.1017/S1350482702001081.
    Golding, B.W. 1998. "Nimrod: A system for generating automated very short range forecasts."
        Meteorol. Appl. 5: 1-16. doi:https://doi.org/10.1017/S1350482798000577.
    """

    def __init__(self, model_id_attr: str = None):
        """
        Set up class

        Args:
            model_id_attr:
                Name of model ID attribute to be copied from source cubes to output cube
        """
        self.model_id_attr = model_id_attr
        self.cape = Cube(None)
        self.precip = Cube(None)
        self.cube_names = [
            "atmosphere_convective_available_potential_energy",
            "lwe_precipitation_rate_max",
        ]
        self._minimum_cape = 10.0  # J kg-1. Minimum value to diagnose updraught from
        self._minimum_precip = 5.0  # mm h-1. Minimum value to diagnose updraught from

    def _parse_inputs(self, inputs: List[Cube]) -> None:
        """
        Separates input CubeList into CAPE and precipitation rate objects with standard units
        and raises Exceptions if it can't, or finds excess data.

        Args:
            inputs:
                List of Cubes containing exactly one of CAPE and Precipitation rate.
        Raises:
            ValueError:
                If additional cubes are found
        """
        cubes = CubeList(inputs)
        try:
            (self.cape, self.precip) = cubes.extract(self.cube_names)
        except ValueError as e:
            raise ValueError(
                f"Expected to find cubes of {self.cube_names}, not {[c.name() for c in cubes]}"
            ) from e
        if len(cubes) > 2:
            extras = [c.name() for c in cubes if c.name() not in self.cube_names]
            raise ValueError(f"Unexpected Cube(s) found in inputs: {extras}")
        if not spatial_coords_match(inputs):
            raise ValueError(f"Spatial coords of input Cubes do not match: {cubes}")
        time_error_msg = self._input_times_error()
        if time_error_msg:
            raise ValueError(time_error_msg)
        self.cape.convert_units("J kg-1")
        self.precip.convert_units("mm h-1")
        if self.model_id_attr:
            if (
                self.cape.attributes[self.model_id_attr]
                != self.precip.attributes[self.model_id_attr]
            ):
                raise ValueError(
                    f"Attribute {self.model_id_attr} does not match on input cubes. "
                    f"{self.cape.attributes[self.model_id_attr]} != "
                    f"{self.precip.attributes[self.model_id_attr]}"
                )

    def _input_times_error(self) -> str:
        """
        Returns appropriate error message string if

        - CAPE cube time is unbounded
        - CAPE time point is lower bound of precip cube time point
        - CAPE and precip cubes have different forecast reference times
        """
        cape_time = self.cape.coord("time")
        if cape_time.has_bounds():
            return "CAPE cube must not have time bounds"
        if self.cape.coord("forecast_reference_time") != self.precip.coord(
            "forecast_reference_time"
        ):
            return "Forecast reference times do not match"
        if not self.precip.coord("time").has_bounds():
            return "Precip cube must have time bounds"
        if cape_time.cell(0).point != self.precip.coord("time").cell(0).bound[0]:
            return "CAPE time must match precip cube's lower time bound"
        return ""

    def _updraught_from_cape(self) -> np.ndarray:
        """
        Calculate the updraught from CAPE data

        Calculation is 0.25 * sqrt(2 * cape)

        Returns zero where CAPE < 10 J kg-1
        """
        updraught = 0.25 * (2 * self.cape.data) ** 0.5
        updraught[self.cape.data < self._minimum_cape] = 0.0
        return updraught.astype(np.float32)

    def _updraught_increment_from_precip(self) -> np.ndarray:
        """
        Calculate the updraught increment from the precipitation rate.

        Calculation is 7.33 * (precip / 28.7)^0.22
        Where precipitation rate < 5 mm h-1, increment is zero.
        """
        increment = 7.33 * (self.precip.data / 28.7) ** 0.22
        increment[self.precip.data < self._minimum_precip] = 0.0
        return increment.astype(np.float32)

    def _make_updraught_cube(self, data: np.ndarray) -> Cube:
        """Puts the data array into a CF-compliant cube"""
        attributes = {}
        if self.model_id_attr:
            attributes[self.model_id_attr] = self.precip.attributes[self.model_id_attr]
        cube = create_new_diagnostic_cube(
            "maximum_vertical_updraught",
            "m s-1",
            self.precip,
            mandatory_attributes=generate_mandatory_attributes(
                [self.precip, self.cape]
            ),
            optional_attributes=attributes,
            data=data,
        )
        return cube

    def process(self, inputs: List[Cube]) -> Cube:
        """Executes methods to calculate updraught from CAPE and precipitation rate
        and packages this as a Cube with appropriate metadata.

        Args:
            inputs:
                List of CAPE and precipitation rate cubes (any order)

        Returns:
            Cube:
                Containing maximum vertical updraught
        """
        self._parse_inputs(inputs)
        return self._make_updraught_cube(
            self._updraught_from_cape() + self._updraught_increment_from_precip()
        )
