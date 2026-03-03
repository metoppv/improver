# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import iris
import numpy as np


class LayerExtractionAndInterpolation:
    def __init__(self, metres_to_ft=3.28084):
        self.metres_to_ft = metres_to_ft

    def process(self, temp_cube, bottom, top, verbosity=0):
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
