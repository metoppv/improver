# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to contain CloudCondensationLevel plugin."""
from typing import Tuple, Union

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
    HumidityMixingRatio,
    dry_adiabatic_temperature,
    saturated_humidity,
)
from improver.utilities.common_input_handle import as_cubelist


class MetaCloudCondensationLevel(PostProcessingPlugin):
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
        self._humidity_plugin = HumidityMixingRatio(model_id_attr=model_id_attr)
        self._cloud_condensation_level_plugin = CloudCondensationLevel(
            model_id_attr=model_id_attr
        )

    def process(self, *cubes: Union[Cube, CubeList]) -> Tuple[Cube, Cube]:
        """
        Call HumidityMixingRatio followed by CloudCondensationLevel to calculate cloud
        condensation level.

        Args:
            cubes:
                Cubes of temperature (K), pressure (Pa) and humidity (1).

        Returns:
            Cubes of air_temperature_at_cloud_condensation_level and
            air_pressure_at_cloud_condensation_level

        """
        humidity = self._humidity_plugin(*cubes)
        return self._cloud_condensation_level_plugin(
            self._humidity_plugin.temperature, self._humidity_plugin.pressure, humidity
        )


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

    def process(self, *cubes: Union[Cube, CubeList]) -> Tuple[Cube, Cube]:
        """
        Calculates the cloud condensation level from the near-surface inputs.

        Args:
            cubes:
                Cubes of temperature (K), pressure (Pa)
                and humidity mixing ratio (kg kg-1)

        Returns:
            Cubes of air_temperature_at_cloud_condensation_level and
            air_pressure_at_cloud_condensation_level

        """
        cubes = as_cubelist(cubes)
        (self.temperature, self.pressure, self.humidity) = CubeList(cubes).extract(
            ["air_temperature", "surface_air_pressure", "humidity_mixing_ratio"]
        )
        ccl_pressure, ccl_temperature = self._iterate_to_ccl()
        return (
            self._make_ccl_cube(ccl_temperature, is_temperature=True),
            self._make_ccl_cube(ccl_pressure, is_temperature=False),
        )
