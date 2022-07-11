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
"""Module to contain CloudCondensationLevel plugin."""
from typing import List, Tuple

import numpy as np
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList
from scipy.optimize import newton

from improver import BasePlugin
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.psychrometric_calculations.psychrometric_calculations import (
    dry_adiabatic_temperature,
    saturated_humidity,
)
from improver.utilities.cube_checker import spatial_coords_match


class CloudCondensationLevel(BasePlugin):
    """
    Derives the temperature and pressure of the convective cloud condensation
    level from near-surface values of temperature, pressure and humidity mixing
    ratio.
    """

    def __init__(self, model_id_attr: str = None):
        """
        Set up class

        Args:
            model_id_attr:
                Name of model ID attribute to be copied from source cubes to output cube
        """
        self.model_id_attr = model_id_attr
        self.cube_names = [
            "air_temperature",
            "surface_air_pressure",
            "humidity_mixing_ratio",
        ]

    def _input_times_error(self) -> str:
        """
        Returns appropriate error message string if

        - Any input cube time is bounded
        - Input cube times and forecast_reference_times do not match
        """
        inputs = [self.temperature, self.pressure, self.humidity]
        cubes_with_time_bounds = [
            c.name() for c in inputs if c.coord("time").has_bounds()
        ]
        if cubes_with_time_bounds:
            return f"{' and '.join(cubes_with_time_bounds)} must not have time bounds"
        for time_coord_name in ["time", "forecast_reference_time"]:
            time_coords = [c.coord(time_coord_name) for c in inputs]
            if not all([tc == time_coords[0] for tc in time_coords[1:]]):
                return (
                    f"{time_coord_name} coordinates do not match."
                    "\n  "
                    "\n  ".join(
                        [str(c.name()) + ": " + str(c.coord("time")) for c in inputs]
                    )
                )
        return ""

    def _parse_inputs(self, inputs: List[Cube]) -> None:
        """Extracts temperature, pressure and humidity mixing ratio cubes from
        the cube list and ensures units are as this plugin expects.

        Args:
            inputs:
                List of Cubes containing exactly one of CAPE and Precipitation rate.
        Raises:
            ValueError:
                If additional cubes are found
        """
        cubes = CubeList(inputs)
        try:
            (self.temperature, self.pressure, self.humidity) = cubes.extract(
                self.cube_names
            )
        except ValueError as e:
            raise ValueError(
                f"Expected to find cubes of {self.cube_names}, not {[c.name() for c in cubes]}"
            ) from e
        if len(cubes) > 3:
            extras = [c.name() for c in cubes if c.name() not in self.cube_names]
            raise ValueError(f"Unexpected Cube(s) found in inputs: {extras}")
        if not spatial_coords_match(inputs):
            raise ValueError(f"Spatial coords of input Cubes do not match: {cubes}")
        time_error_msg = self._input_times_error()
        if time_error_msg:
            raise ValueError(time_error_msg)
        self.temperature.convert_units("K")
        self.pressure.convert_units("Pa")
        self.humidity.convert_units("kg kg-1")
        if self.model_id_attr:
            model_id_value = {cube.attributes[self.model_id_attr] for cube in inputs}
            if len(model_id_value) != 1:
                raise ValueError(
                    f"Attribute {self.model_id_attr} does not match on input cubes. "
                    f"{' != '.join(model_id_value)}"
                )

    def _make_ccl_cube(self, data: np.ndarray, pressure: np.ndarray) -> Cube:
        """Puts the data array into a CF-compliant cube"""
        attributes = {}
        if self.model_id_attr:
            attributes[self.model_id_attr] = self.temperature.attributes[
                self.model_id_attr
            ]
        cube = create_new_diagnostic_cube(
            "temperature_at_cloud_condensation_level",
            "K",
            self.temperature,
            mandatory_attributes=generate_mandatory_attributes(
                [self.temperature, self.pressure, self.humidity]
            ),
            optional_attributes=attributes,
            data=data,
        )
        pressure_coord = AuxCoord(
            standard_name="air_pressure", units="Pa", points=pressure
        )
        spatial_dims = range(self.temperature.ndim)
        cube.add_aux_coord(pressure_coord, spatial_dims)
        return cube

    def _iterate_to_ccl(self) -> Tuple[np.ndarray, np.ndarray]:
        """Uses a Newton iterator to find the pressure level where the
        adiabatically-adjusted temperature equals the saturation temperature.
        Returns pressure and temperature arrays."""

        def humidity_delta(p2, p, t, q):
            """For a target pressure guess, p2, and origin p, t and q, return the
            difference between q and q_sat(t2, p2)"""
            t2 = dry_adiabatic_temperature(t, p, p2)
            return q - saturated_humidity(t2, p2)

        ccl_pressure = newton(
            humidity_delta,
            self.pressure.data.copy(),
            args=(self.pressure.data, self.temperature.data, self.humidity.data),
            tol=1e-6,
            maxiter=100,
        ).astype(np.float32)
        ccl_temperature = dry_adiabatic_temperature(
            self.temperature.data, self.pressure.data, ccl_pressure
        )
        return ccl_pressure, ccl_temperature

    def process(self, cubes: List[Cube]) -> Cube:
        """
        Calculates the cloud condensation level from the near-surface inputs.

        Args:
            cubes:
                Cubes of temperature, pressure and humidity mixing ratio

        Returns:
            Cube of cloud condensation level

        """
        self._parse_inputs(cubes)
        ccl_pressure, ccl_temperature = self._iterate_to_ccl()
        return self._make_ccl_cube(ccl_temperature, ccl_pressure)
