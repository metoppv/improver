# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to generate total precipitable water"""

import iris
import numpy as np
from iris.cube import Cube

# constants used in the calculation
GRAVITY = 9.81  # m/s²
WATER_DENSITY = 1000.0  # kg/m³


class PrecipitableWater:
    """
    Class to calculate total precipitable water from a humidity mixing ratio cube.

    This method integrates the water vapour content over pressure levels to
    compute the total precipitable water in metres. It assumes pressure levels
    are in Pascals and strictly decreasing.

    Attributes:
        model_type (str): Type of model used ('global' or 'ukv').
    """

    def __init__(self, model_type="global"):
        """
        Initialise the PrecipitableWater class.

        Args:
            model_type (str): Model type, e.g., 'global' or 'ukv'. Determines metadata
                              added to the output cube.
        """

        self.model_type = model_type.lower()

    def process(self, humidity_mixing_ratio_cube: Cube):
        # Create copy of input cube
        cube = humidity_mixing_ratio_cube.copy()

        """
         Calculate total precipitable water from a humidity mixing ratio cube.

         Args:
             humidity_mixing_ratio_cube (Cube): An Iris cube containing specific humidity
                                                data with an 'air_pressure' coordinate.

         Returns:
             Cube: A new Iris cube containing total precipitable water (in metres),
                   with appropriate metadata and attributes.
         """

        # Extract the air pressure coordinate
        try:
            pressure_coord = cube.coord("air_pressure")
        except iris.exceptions.CoordinateNotFoundError:
            raise ValueError("Cube must have an 'air_pressure' coordinate.")

        # Ensure pressure units are in Pascals
        if pressure_coord.units != "Pa":
            raise ValueError("Pressure units must be in Pascals (Pa).")

        # Extract pressure levels and sort them into descending order
        pressure_levels = pressure_coord.points
        sorted_indices = np.argsort(pressure_levels)[::-1]
        sorted_pressure = pressure_levels[sorted_indices]
        sorted_data = cube.data[sorted_indices]

        # Calculate midpoints between pressure levels to define layer boundaries
        midpoints = 0.5 * (sorted_pressure[:-1] + sorted_pressure[1:])
        layer_bounds = np.empty(len(sorted_pressure) + 1)
        layer_bounds[1:-1] = midpoints
        # Estimate top and bottom bounds by extrapolating from midpoints
        layer_bounds[0] = sorted_pressure[0] + (sorted_pressure[0] - midpoints[0])
        layer_bounds[-1] = sorted_pressure[-1] - (midpoints[-1] - sorted_pressure[-1])

        # Ensure bounds are strictly decreasing
        if not np.all(np.diff(layer_bounds) < 0):
            raise ValueError("Layer bounds must be strictly decreasing.")

        # Calculate pressure thickness of each layer and reshape for broadcasting with cube data
        delta_p = -np.diff(layer_bounds)  # Negate to ensure positive thickness
        reshaped_delta_p = delta_p.reshape((len(delta_p),) + (1,) * (cube.ndim - 1))

        # Calculate precipitable water per layer using the following formula
        # PW = (mixing ratio * pressure thickness) / (GRAVITY * WATER_DENSITY)
        pw_data = sorted_data * reshaped_delta_p / (GRAVITY * WATER_DENSITY)
        # Ensure no negative values and sum across the vertical layers to get the total precipitable water
        pw_data = np.maximum(pw_data, 0)
        total_pw_data = np.sum(pw_data, axis=0)

        # Create a new cube for the total precipitable water
        total_pw_cube = cube[0].copy(total_pw_data)
        total_pw_cube.remove_coord("air_pressure")
        total_pw_cube.rename("total_precipitable_water")
        total_pw_cube.units = "m"
        total_pw_cube.standard_name = "lwe_thickness_of_precipitation_amount"
        total_pw_cube.attributes = cube.attributes.copy()
        total_pw_cube.attributes["least_significant_digit"] = 3

        # Add model-specific metadata
        if self.model_type == "global":
            total_pw_cube.attributes["title"] = (
                "Global Enhanced Model Forecast on Global 10 km Standard Grid"
            )
        elif self.model_type == "ukv":
            total_pw_cube.attributes["title"] = (
                "UKV Enhanced Model Forecast on UK 2 km Standard Grid"
            )

        return total_pw_cube
