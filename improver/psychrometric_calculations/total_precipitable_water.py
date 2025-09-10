# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to generate total precipitable water"""

import iris
import numpy as np
from iris.cube import Cube

from improver import BasePlugin

# Constants used in the calculation:

# Acceleration due to gravity, Earth's surface (m s-2)
EARTH_SURFACE_GRAVITY_ACCELERATION = 9.81  # m/s²

# Density of water: (kg m-3)
WATER_DENSITY = 1000  # kg/m³


class PrecipitableWater(BasePlugin):
    """
    Class to calculate total precipitable water from a humidity mixing ratio cube.
    """

    def __init__(self, model_type="global"):
        self.model_type = model_type.lower()

    def process(self, humidity_mixing_ratio_cube: Cube):
        cube = humidity_mixing_ratio_cube.copy()

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

        midpoints = 0.5 * (sorted_pressure[:-1] + sorted_pressure[1:])
        layer_bounds = np.empty(len(sorted_pressure) + 1)
        layer_bounds[1:-1] = midpoints
        layer_bounds[0] = sorted_pressure[0] + (sorted_pressure[0] - midpoints[0])
        layer_bounds[-1] = sorted_pressure[-1] - (midpoints[-1] - sorted_pressure[-1])

        if not np.all(np.diff(layer_bounds) < 0):
            raise ValueError("Layer bounds must be strictly decreasing.")

        reshaped_delta_p = self.calculate_pressure_thickness(layer_bounds, cube.data)
        total_pw_data = self.calculate_precipitable_water(sorted_data, reshaped_delta_p)

        total_pw_cube = cube[0].copy(total_pw_data)
        total_pw_cube.remove_coord("pressure")
        total_pw_cube.rename("total_precipitable_water")
        total_pw_cube.units = "m"
        total_pw_cube.standard_name = "lwe_thickness_of_precipitation_amount"
        total_pw_cube.attributes = cube.attributes.copy()

        return total_pw_cube

    @staticmethod
    def calculate_pressure_thickness(layer_bounds, cube_data):
        delta_p = -np.diff(layer_bounds)
        reshaped_delta_p = delta_p.reshape(
            (len(delta_p),) + (1,) * (cube_data.ndim - 1)
        )
        return reshaped_delta_p

    @staticmethod
    def calculate_precipitable_water(sorted_data, reshaped_delta_p):
        pw_data = (
            sorted_data
            * reshaped_delta_p
            / (EARTH_SURFACE_GRAVITY_ACCELERATION * WATER_DENSITY)
        )
        pw_data = np.maximum(pw_data, 0)
        total_pw_data = np.sum(pw_data, axis=0)
        return total_pw_data
