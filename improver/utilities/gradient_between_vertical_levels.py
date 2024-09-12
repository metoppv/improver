# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Calculate the gradient between two vertical levels."""

from typing import Optional

import iris
import numpy as np
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError

from improver import BasePlugin
from improver.utilities.common_input_handle import as_cubelist


class GradientBetweenVerticalLevels(BasePlugin):
    """Calculate the gradient between two vertical levels. The gradient is calculated as the
    difference between the input cubes divided by the difference in height."""

    @staticmethod
    def extract_cube_from_list(cubes: CubeList, name: str) -> Cube:
        """Extract a cube from a cubelist based on the name if it exists. If the cube is found
        it is removed from the cubelist.

        Args:
            cubes:
                A cube list containing cubes.
            name:
                The name of the cube to be extracted.
        Returns:
            The extracted cube or None if there is no cube with the specified name. Also returns
            the input cubelist with the extracted cube removed.
        """
        try:
            extracted_cube = cubes.extract_cube(iris.Constraint(name))
        except iris.exceptions.ConstraintMismatchError:
            extracted_cube = None
        else:
            cubes.remove(extracted_cube)

        return extracted_cube, cubes

    def gradient_over_vertical_levels(
        self,
        cubes: CubeList,
        geopotential_height: Optional[Cube],
        orography: Optional[Cube],
    ) -> Cube:
        """Calculate the gradient between two vertical levels. The gradient is calculated as the
        difference between the two cubes in cubes divided by the difference in height.

        If the cubes are provided at height levels this is assumed to be a height above ground level
        and the height above sea level is calculated by adding the height of the orography to the
        height coordinate.If the cubes are provided at pressure levels, the height above sea level
        is extracted from a geopotential_height cube.

        Args:
            cubes:
                Two cubes containing a diagnostic at two different vertical levels. The cubes must
                contain either a height or pressure scalar coordinate. If the cubes contain a height
                scalar coordinate this is assumed to be a height above ground level.
            geopotential_height:
                Optional cube that contains the height above sea level of pressure levels. This cube
                is required if any input cube is defined at pressure level. This is used to extract
                the height above sea level at the pressure level of the input cubes.
            orography:
                Optional cube containing the orography height above sea level. This cube is required
                if any input cube is defined at height levels and is used to convert the height
                above ground level to height above sea level.

        Returns:
            A cube containing the gradient between the cubes between two vertical levels.

        Raises:
            ValueError: If either input cube is defined at height levels and no orography cube is
                        provided.
            ValueError: If either input cube is defined at pressure levels and no
                        geopotential_height cube is provided.
        """

        cube_heights = []

        for cube in cubes:
            try:
                cube_height = np.array(cube.coord("height").points)
            except CoordinateNotFoundError:
                if geopotential_height:
                    height_ASL = geopotential_height.extract(
                        iris.Constraint(pressure=cube.coord("pressure").points)
                    )
                else:
                    raise ValueError(
                        """No geopotential height cube provided but one of the inputs cubes has a
                        pressure coordinate"""
                    )
            else:
                if orography:
                    height_ASL = orography + cube_height
                else:
                    raise ValueError(
                        """No orography cube provided but one of the input cubes has height
                        coordinate"""
                    )
            cube_heights.append(height_ASL)

        height_diff = cube_heights[0] - cube_heights[1]
        height_diff.data = np.ma.masked_where(height_diff.data == 0, height_diff.data)

        diff = cubes[0] - cubes[1]
        gradient = diff / height_diff

        return gradient

    def process(self, *cubes: CubeList) -> Cube:
        """
            Process the input cubes to calculate the gradient between two vertical levels.

            Args:
                cubes:
                    A cubelist of two cubes containing a diagnostic at two vertical levels.
                    The cubes must contain either a height or pressure scalar coordinate.
                    If either of the cubes contain a height scalar coordinate this is assumed
                    to be a height above ground level and a cube with the name "surface_altitude"
                    must also be provided. If either cube contains a pressure scalar coordinate
                    a cube with the name "geopotential_height" must be provided.

            Returns:
                A cube containing the gradient between two vertical levels. The cube will be
                names gradient_of_ followed by the name of the input cubes.
            """
        cubes = as_cubelist(cubes)

        orography, cubes = self.extract_cube_from_list(cubes, "surface_altitude")
        geopotential_height, cubes = self.extract_cube_from_list(
            cubes, "geopotential_height"
        )

        gradient = self.gradient_over_vertical_levels(
            cubes, geopotential_height, orography
        )

        gradient.rename(f"gradient_of_{cubes[0].name()}")
        gradient.units = f"{cubes[0].units} m-1"
        return gradient
