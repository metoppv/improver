# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to generate precipitable water"""

import iris
import numpy as np
from iris.cube import Cube
from numpy import ndarray

from improver import BasePlugin
from improver.constants import EARTH_SURFACE_GRAVITY_ACCELERATION, WATER_DENSITY


def ensure_no_float_64(cube: Cube) -> Cube:
    """
    Convert float64 data and coordinates in the cube to float32 to ensure workflow compatibility.
    """
    if cube.data.dtype == np.float64:
        cube.data = cube.data.astype(np.float32)
    return cube


class PrecipitableWater(BasePlugin):
    """
    Plugin to calculate precipitable water from a humidity mixing ratio cube.

    This plugin integrates the humidity mixing ratio over pressure levels to compute
    the precipitable water in metres. It assumes pressure levels are in
    Pascals and uses constants for gravity and water density to convert the result
    into liquid water equivalent thickness.
    """

    @staticmethod
    def calculate_pressure_thickness(
        layer_bounds: ndarray, cube_data: ndarray
    ) -> ndarray:
        """
        Calculate the pressure thickness of each atmospheric layer.

        Args:
            layer_bounds:
                1D array of pressure layer boundaries in Pascals.
            cube_data:
                N-dimensional array of humidity mixing ratio values.

        Returns:
            N-dimensional array of pressure thickness values reshaped to match
            the dimensions of the input data.
        """
        delta_p = np.abs(np.diff(layer_bounds))
        reshaped_delta_p = delta_p.reshape(
            (len(delta_p),) + (1,) * (cube_data.ndim - 1)
        )
        return reshaped_delta_p

    @staticmethod
    def calculate_precipitable_water(
        humidity_data: ndarray, delta_p: ndarray
    ) -> ndarray:
        """
        Calculate precipitable water using sorted humidity data and pressure thickness.

        Args:
            humidity_data:
                Humidity mixing ratio data sorted by pressure levels.
            delta_p:
                Pressure thickness values reshaped to match the data dimensions.

        Returns:
            Array of precipitable water values in metres.
        """
        pw_data = (
            humidity_data
            * delta_p
            / (EARTH_SURFACE_GRAVITY_ACCELERATION * WATER_DENSITY)
        )
        return pw_data

    def calculate_layer_bounds(self, sorted_pressure: ndarray) -> ndarray:
        """
        Calculate pressure layer boundaries from a 1D array of pressure levels.

        This method estimates the boundaries between pressure levels by computing midpoints
        between adjacent levels. It assumes that the topmost and bottommost layers are
        half as thick as the internal layers, and extrapolates their bounds accordingly.

        Args:
            sorted_pressure:
                1D array of pressure levels in Pascals.

        Returns:
            1D array of pressure layer boundaries in Pascals.
        """

        midpoints = 0.5 * (sorted_pressure[:-1] + sorted_pressure[1:])
        layer_bounds = np.empty(len(sorted_pressure) + 1)
        layer_bounds[1:-1] = midpoints
        layer_bounds[0] = sorted_pressure[0] + (sorted_pressure[0] - midpoints[0])
        layer_bounds[-1] = sorted_pressure[-1] - (midpoints[-1] - sorted_pressure[-1])

        return layer_bounds

    def process(self, humidity_mixing_ratio_cube: Cube) -> Cube:
        """
        Calculate precipitable water on pressure levels.

        Args:
            humidity_mixing_ratio_cube:
                Cube containing humidity mixing ratio data with pressure levels.

        Returns:
            Cube containing precipitable water in metres, with updated metadata, on pressure levels.
        """
        cube = humidity_mixing_ratio_cube.copy()

        try:
            pressure_coord = cube.coord("pressure")
        except iris.exceptions.CoordinateNotFoundError:
            raise ValueError("Cube must have a 'pressure' coordinate.")

        pressure_coord.convert_units("Pa")

        pressure_levels = pressure_coord.points
        data = cube.data

        layer_bounds = self.calculate_layer_bounds(pressure_levels)
        reshaped_delta_p = self.calculate_pressure_thickness(layer_bounds, data)
        pw_data = self.calculate_precipitable_water(data, reshaped_delta_p)

        pw_cube = cube.copy(data=pw_data)
        pw_cube.rename("precipitable_water")
        pw_cube.units = "m"
        pw_cube.standard_name = "lwe_thickness_of_precipitation_amount"
        pw_cube.attributes = cube.attributes.copy()

        # Enforce float32 precision to avoid dtype errors in downstream processing
        pw_cube = ensure_no_float_64(pw_cube)

        return pw_cube
