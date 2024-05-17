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
from typing import List, Tuple, Union

import numpy as np
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError
from scipy.optimize import newton

from improver import PostProcessingPlugin
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.psychrometric_calculations.psychrometric_calculations import (
    dry_adiabatic_temperature,
    saturated_humidity,
)
from improver.utilities.flatten import flatten


class MetaPluginCloudCondensationLevel(PostProcessingPlugin):
    """
    Meta-plugin which handles the calling of HumidityMixingRatio followed by CloudCondensationLevel.
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
        from improver.psychrometric_calculations.psychrometric_calculations import (
            HumidityMixingRatio,
        )
        model_id_attr = model_id_attr
        self._humidity_plugin = HumidityMixingRatio(model_id_attr=model_id_attr)
        self._cloud_condensation_level_plugin = CloudCondensationLevel(model_id_attr=model_id_attr)

    def process(self, *cubes: Union[Cube,CubeList]) -> Tuple[Cube, Cube]:
        """
        Calls the HumidityMixingRatio plugin to calculate humidity mixing ratio from relative humidity.
        Calls the CloudCondensationLevel plugin to calculate cloud condensation level.

        Args:
            cubes:
                Cubes, of temperature (K), pressure (Pa) and humidity (1).

        Returns:
            Cubes of air_temperature_at_cloud_condensation_level and
            air_pressure_at_cloud_condensation_level

        """
        humidity = self._humidity_plugin(cubes)
        return self._cloud_condensation_level_plugin(
            self._humidity_plugin.temperature, self._humidity_plugin.pressure, humidity)


class CloudCondensationLevel(PostProcessingPlugin):
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
        self.temperature, self.pressure, self.humidity = None, None, None

    def _make_ccl_cube(self, data: np.ndarray, is_temperature: bool) -> Cube:
        """Puts the data array into a CF-compliant cube"""
        attributes = {}
        if self.model_id_attr:
            attributes[self.model_id_attr] = self.temperature.attributes[
                self.model_id_attr
            ]
        if is_temperature:
            name = "air_temperature_at_condensation_level"
            units = "K"
        else:
            name = "air_pressure_at_condensation_level"
            units = "Pa"
        cube = create_new_diagnostic_cube(
            name,
            units,
            self.temperature,
            mandatory_attributes=generate_mandatory_attributes(
                [self.temperature, self.pressure, self.humidity]
            ),
            optional_attributes=attributes,
            data=data,
        )
        # The template cube may have had a height coord describing it as screen-level.
        # This needs removing:
        try:
            cube.remove_coord("height")
        except CoordinateNotFoundError:
            pass
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
            tol=100,
            maxiter=20,
            disp=False,
        ).astype(np.float32)
        ccl_temperature = dry_adiabatic_temperature(
            self.temperature.data, self.pressure.data, ccl_pressure
        )
        return ccl_pressure, ccl_temperature

    def process(self, *cubes: Union[Cube,CubeList]) -> Tuple[Cube, Cube]:
        """
        Calculates the cloud condensation level from the near-surface inputs.

        Args:
            cubes:
                Cubes, of temperature (K), pressure (Pa)
                and humidity mixing ratio (kg kg-1)

        Returns:
            Cubes of air_temperature_at_cloud_condensation_level and
            air_pressure_at_cloud_condensation_level

        """
        cubes = flatten(cubes)
        (self.temperature, self.pressure, self.humidity) = CubeList(cubes).extract(
            ["air_temperature", "surface_air_pressure", "humidity_mixing_ratio"]
        )
        ccl_pressure, ccl_temperature = self._iterate_to_ccl()
        return (
            self._make_ccl_cube(ccl_temperature, is_temperature=True),
            self._make_ccl_cube(ccl_pressure, is_temperature=False),
        )
