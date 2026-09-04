# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing plugin for generating extra realizations from a cube with fewer
realizations than required."""

from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.utilities.cube_manipulation import enforce_coordinate_ordering


class ExpandRealizationDimension(BasePlugin):
    """Plugin to expand the realization dimension of a cube to the required number of
    realizations."""

    def __init__(self, n_realizations_required: int):
        """
        Args:
            n_realizations_required:
                The number of realizations required in the output cube.
        """
        self.n_realizations_required = n_realizations_required

    def process(self, cube: Cube) -> Cube:
        """
        Expand the realization dimension of a cube by repeating the existing
        realizations as necessary. E.g. if the input cube has 18 realizations and 24 are
        required, the first 18 realizations will be repeated and the first 6 of these
        will be used to create the additional 6 realizations needed to reach the
        required 24. Realization points will be renumbered from 0 to
        n_realizations_required - 1 in the output cube, regardless of the original
        realization points in the input cube.

        Args:
            cube:
                A cube with a realization coordinate that has fewer realizations than
                required.

        Returns:
            Expanded cube. Dimensions are the same as input cube, with the realization
            dimension expanded to the specified size.

        Raises:
            ValueError: if realization is not a dimension coordinate.
        """
        if not cube.coords("realization", dim_coords=True):
            raise ValueError(
                "The input cube does not contain a realization coordinate."
            )

        realizations_available = cube.coord("realization").points.size

        extended_realization_cubelist = CubeList([])

        # Ensure the realization coordinate is the first dimension coordinate
        enforce_coordinate_ordering(cube, ["realization"])

        for index in range(self.n_realizations_required):
            r_index = index % realizations_available
            realization_slice = cube[r_index]
            realization_slice.coord("realization").points = index
            extended_realization_cubelist.append(realization_slice)

        return extended_realization_cubelist.merge_cube()
