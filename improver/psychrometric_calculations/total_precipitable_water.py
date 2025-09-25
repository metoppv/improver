# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to generate total precipitable water"""

import iris
import numpy as np
from iris.cube import Cube
from numpy import ndarray

from improver import BasePlugin
from improver.constants import EARTH_SURFACE_GRAVITY_ACCELERATION, WATER_DENSITY


class PrecipitableWater(BasePlugin):
    """
    Plugin to calculate total precipitable water from a humidity mixing ratio cube.

    This plugin integrates the humidity mixing ratio over pressure levels to compute
    the total precipitable water (TPW) in metres. It assumes pressure levels are in
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
        sorted_data: ndarray, reshaped_delta_p: ndarray
    ) -> ndarray:
        """
        Calculate total precipitable water using sorted humidity data and pressure thickness.

        Args:
            sorted_data:
                Humidity mixing ratio data sorted by pressure levels.
            reshaped_delta_p:
                Pressure thickness values reshaped to match the data dimensions.

        Returns:
            Array of total precipitable water values in metres.
        """
        pw_data = (
            sorted_data
            * reshaped_delta_p
            / (EARTH_SURFACE_GRAVITY_ACCELERATION * WATER_DENSITY)
        )
        total_pw_data = np.maximum(pw_data, 0)
        return total_pw_data

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
        Main method to calculate total precipitable water.

        Args:
            humidity_mixing_ratio_cube:
                Cube containing humidity mixing ratio data with pressure levels.

        Returns:
            Cube containing total precipitable water in metres, with updated metadata.
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
        total_pw_data = self.calculate_precipitable_water(data, reshaped_delta_p)

        total_pw_cube = cube.copy(data=total_pw_data)
        total_pw_cube.rename("total_precipitable_water")
        total_pw_cube.units = "m"
        total_pw_cube.standard_name = "lwe_thickness_of_precipitation_amount"
        total_pw_cube.attributes = cube.attributes.copy()

        return total_pw_cube
