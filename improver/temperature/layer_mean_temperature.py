# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import iris
import numpy as np
from cf_units import Unit
from iris.cube import Cube

from improver import BasePlugin


class LayerTemperatureInterpolation(BasePlugin):
    """
    Plugin to interpolate temperature values at specified layer boundaries.

    This plugin extracts all temperature levels within a specified vertical layer
    (between `bottom` and `top` heights, in feet), and interpolates temperature
    at the exact base and top of the layer. The output is a cube containing
    temperature at all interior levels plus the interpolated base and top.
    """

    def process(
        self, temp_cube: Cube, bottom: float, top: float, verbosity: int = 0
    ) -> Cube:
        """
        Interpolate temperature values at layer boundaries.

        Args:
            temp_cube: Input temperature cube with a height coordinate in metres.
            bottom: Lower boundary of the layer in feet.
            top: Upper boundary of the layer in feet.
            verbosity: Verbosity level for printing debug information.

        Returns:
            Cube containing temperature at all layer heights
            (base, interior, and top).
        """
        if bottom >= top:
            raise ValueError(f"Bottom ({bottom} ft) must be less than top ({top} ft).")

        # Convert layer bounds from feet to metres using cf_units
        bottom_m = Unit("ft").convert(bottom, "m")
        top_m = Unit("ft").convert(top, "m")

        if verbosity:
            print(f"Interpolating temperature at base: {bottom} ft")

        # Extract cube of temperature levels within layer
        between_layer_temp_cube = temp_cube.extract(
            iris.Constraint(height=lambda point: bottom_m < point < top_m)
        )

        # Interpolate temperature at base of layer
        base_temp = temp_cube.interpolate(
            [("height", np.array([bottom_m], dtype=np.float32))],
            iris.analysis.Linear(),
            collapse_scalar=False,
        )

        if verbosity:
            print(f"Interpolating temperature at top: {top} ft")

        # Interpolate temperature at top of layer
        top_temp = temp_cube.interpolate(
            [("height", np.array([top_m], dtype=np.float32))],
            iris.analysis.Linear(),
            collapse_scalar=False,
        )

        # Merge cubes of temperature at top, bottom and within layer
        cubes_to_merge = [base_temp, between_layer_temp_cube, top_temp]
        cubes_to_merge = [cube for cube in cubes_to_merge if cube is not None]
        layer_levels_temp_cube = iris.cube.CubeList(cubes_to_merge).concatenate_cube()

        return layer_levels_temp_cube


class CalculateLayerMeanTemperature(BasePlugin):
    """Calculate the vertically weighted mean temperature for a layer."""

    def process(self, layer_cube: Cube, verbosity: int = 0) -> Cube:
        """
        Calculate the altitude-weighted mean temperature across the layer.

        Args:
            layer_cube: Cube containing temperature at all heights within
            the specified layer (including interpolated base and top).
            verbosity: Set level of output to print.

        Returns:
            2D cube of layer mean temperature.
        """
        # Set up array for holding sum of products of temperature and vertical distance
        layer_temp_product = np.zeros(layer_cube.data.shape[1:])

        # Estimate mean temperature of layers and
        # Weight by vertical extent of layer
        altitude_array = layer_cube.coord("height").points
        for alt_index in range(1, len(altitude_array) - 1):
            layer_thickness = (
                altitude_array[alt_index + 1] - altitude_array[alt_index - 1]
            ) / 2
            layer_temp_product += layer_cube.data[alt_index, :, :] * layer_thickness

        # Add contributions from base and top
        layer_temp_product += (
            layer_cube.data[0, :, :] * (altitude_array[1] - altitude_array[0]) / 2
        )
        layer_temp_product += (
            layer_cube.data[-1, :, :] * (altitude_array[-1] - altitude_array[-2]) / 2
        )

        # Divide by total thickness to get mean
        lmt_array = layer_temp_product / (altitude_array[-1] - altitude_array[0])

        if verbosity:
            print("Layer mean temperature array:", lmt_array)

        # Wrap result in a cube and add required metadata
        lmt_cube = iris.cube.Cube(
            lmt_array,
            var_name="air_temperature",
            units="K",
            dim_coords_and_dims=(
                (layer_cube.coord("projection_y_coordinate"), 0),
                (layer_cube.coord("projection_x_coordinate"), 1),
            ),
            aux_coords_and_dims=(
                (layer_cube.coord("forecast_period"), ()),
                (layer_cube.coord("forecast_reference_time"), ()),
                (layer_cube.coord("time"), ()),
            ),
        )
        return lmt_cube
