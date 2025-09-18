# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to generate total precipitable water"""

from typing import Tuple

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

    Attributes:
        model_type (str): Type of model used, e.g., 'global' or 'ukv'.
    """

    def __init__(self, model_type: str = "global") -> None:
        """
        Initialise the PrecipitableWater plugin.

        Args:
            model_type:
                Type of model used, e.g., 'global' or 'ukv'.
        """
        self.model_type = model_type.lower()

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
        delta_p = -np.diff(layer_bounds)
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
        pw_data = np.maximum(pw_data, 0)
        total_pw_data = np.sum(pw_data, axis=0)
        return total_pw_data

    def sort_pressure_levels(self, cube: Cube) -> Tuple[ndarray, ndarray]:
        """
        Sort pressure levels in descending order and reorder the data accordingly.

        Args:
            cube:
                Cube containing humidity mixing ratio data with a pressure coordinate.

        Returns:
            Tuple of sorted pressure levels and corresponding sorted data.

        Raises:
            ValueError: If the pressure coordinate is missing or not in Pascals.
        """
        try:
            pressure_coord = cube.coord("pressure")
        except iris.exceptions.CoordinateNotFoundError:
            raise ValueError("Cube must have a 'pressure' coordinate.")

        if pressure_coord.units != "Pa":
            raise ValueError("Pressure units must be in Pascals (Pa).")

        pressure_levels = pressure_coord.points
        sorted_indices = np.argsort(pressure_levels)[::-1]
        sorted_pressure = pressure_levels[sorted_indices]
        sorted_data = cube.data[sorted_indices]

        return sorted_pressure, sorted_data

    def calculate_layer_bounds(self, sorted_pressure: ndarray) -> ndarray:
        """
        Calculate pressure layer boundaries from sorted pressure levels.

        Args:
            sorted_pressure:
                1D array of pressure levels sorted in descending order.

        Returns:
            1D array of pressure layer boundaries.

        Raises:
            ValueError: If the resulting layer bounds are not strictly decreasing.
        """
        midpoints = 0.5 * (sorted_pressure[:-1] + sorted_pressure[1:])
        layer_bounds = np.empty(len(sorted_pressure) + 1)
        layer_bounds[1:-1] = midpoints
        layer_bounds[0] = sorted_pressure[0] + (sorted_pressure[0] - midpoints[0])
        layer_bounds[-1] = sorted_pressure[-1] - (midpoints[-1] - sorted_pressure[-1])

        if not (np.diff(layer_bounds) < 0).all():
            raise ValueError("Layer bounds must be strictly decreasing.")

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

        sorted_pressure, sorted_data = self.sort_pressure_levels(cube)
        layer_bounds = self.calculate_layer_bounds(sorted_pressure)

        reshaped_delta_p = self.calculate_pressure_thickness(layer_bounds, cube.data)
        total_pw_data = self.calculate_precipitable_water(sorted_data, reshaped_delta_p)

        total_pw_cube = cube[0].copy(total_pw_data)
        total_pw_cube.remove_coord("pressure")
        total_pw_cube.rename("total_precipitable_water")
        total_pw_cube.units = "m"
        total_pw_cube.standard_name = "lwe_thickness_of_precipitation_amount"
        total_pw_cube.attributes = cube.attributes.copy()

        return total_pw_cube
