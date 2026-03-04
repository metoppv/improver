# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import iris
import numpy as np

from improver import BasePlugin


class LayerExtractionAndInterpolation(BasePlugin):
    """
    Plugin to extract and interpolate temperature values at specified layer boundaries.

    This plugin extracts all temperature levels within a specified vertical layer
    (between `bottom` and `top` heights, in feet), and interpolates temperature
    at the exact base and top of the layer. The output is a cube containing
    temperature at all interior levels plus the interpolated base and top.

    """

    def __init__(self, metres_to_ft=3.28084):
        """
        Initialise the plugin.

        Args:
            metres_to_ft (float): Conversion factor from metres to feet.
        """
        self.metres_to_ft = metres_to_ft

    def process(self, temp_cube, bottom, top, verbosity=0):
        """
        Extract and interpolate temperature values at layer boundaries.

        Args:
            temp_cube (iris.cube.Cube): Input temperature cube with a height coordinate.
            bottom (float): Lower boundary of the layer (in feet).
            top (float): Upper boundary of the layer (in feet).
            verbosity (int): Verbosity level for printing debug information.

        Returns:
            iris.cube.Cube: Cube containing temperature at all layer heights
                            (base, interior, and top).
        """
        if verbosity:
            print(f"Extracting/interpolating levels from {bottom} to {top} ft")
        # Extract cube of temperature levels within layer
        between_layer_temp_cube = temp_cube.extract(
            iris.Constraint(
                height=lambda point: bottom / self.metres_to_ft
                < point
                < top / self.metres_to_ft
            )
        )
        # Interpolate temperature at top and base of layer
        base_temp = temp_cube.interpolate(
            [("height", np.array([bottom / self.metres_to_ft], dtype=np.float32))],
            iris.analysis.Linear(),
            collapse_scalar=False,
        )
        top_temp = temp_cube.interpolate(
            [("height", np.array([top / self.metres_to_ft], dtype=np.float32))],
            iris.analysis.Linear(),
            collapse_scalar=False,
        )
        # Merge cubes of temperature at top, bottom and within layer
        layer_levels_temp_cube = iris.cube.CubeList(
            [base_temp, between_layer_temp_cube, top_temp]
        ).concatenate_cube()
        if verbosity > 1:
            print(layer_levels_temp_cube)

        return layer_levels_temp_cube


class CalculateLayerMeanTemperature(BasePlugin):
    """Calculate the vertically weighted mean temperature for a layer."""

    def process(self, layer_cube, verbosity=0):
        """
        Calculate the mean temperature between the lowest and highest heights in the input cube.

        Args:
            layer_cube (iris.cube.Cube): Cube containing temperature at heights within the specified layer
            verbosity (int): Set level of output to print.

        Returns:
            iris.cube.Cube: 2D cube of layer mean temperature.
        """

        # Set up array for holding sum of products of temperature and vertical distance
        layer_temp_product = np.zeros(layer_cube.data.shape[1:])

        # Estimate mean temperature of layers between 2000-3000ft and
        # weight by vertical extent of layer
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

        # Wrap result in a cube (add metadata as needed)
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
