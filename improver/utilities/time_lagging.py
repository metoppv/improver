# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Provide support utilities for time lagging ensembles"""

import warnings
from typing import Union

import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.metadata.forecast_times import rebadge_forecasts_as_latest_cycle
from improver.utilities.common_input_handle import as_cubelist
from improver.utilities.cube_manipulation import MergeCubes


class GenerateTimeLaggedEnsemble(BasePlugin):
    """Combine realizations from different forecast cycles into one cube::

    * If a cube has no ``realization`` coordinate, one is added.
    * For fully deterministic input (no realization coordinates on any cube),
      added realizations are assigned sequentially in input order (0, 1, 2, ...).
    * For mixed input (some cubes have realizations and others do not), missing
      coordinates are added only to cubes that need them.
    * If this creates duplicate realization numbers across the combined inputs,
      all realization numbers are rebadged to a unique sequential set in input
      order before merging.
    """

    def __init__(self, rebadge_realizations: bool = False) -> None:
        """Initialise plugin.

        Args:
            rebadge_realizations:
                If True, rebadge realization points on the final merged cube
                to sequential values starting at 0.
        """
        self.rebadge_realizations = rebadge_realizations

    def _check_validity_times_match(self, cubelist: CubeList) -> None:
        """Raise ValueError if cubes have mismatched validity times.

        Args:
            cubelist:
                List of input forecasts.

        Raises:
            ValueError: If input cubes have mismatched validity times.
        """
        time_coords = [cube.coord("time") for cube in cubelist]
        time_coords_match = [coord == time_coords[0] for coord in time_coords]
        if not all(time_coords_match):
            raise ValueError("Cubes with mismatched validity times are not compatible.")

    def add_realization_coord(self, cubelist: CubeList) -> Cube:
        """Add a realization coordinate to each cube in the cubelist if not
        already present. This facilitates the creation of a time-lagged ensemble
        from deterministic forecasts.

        Args:
            cubelist: List of input forecasts.
        Returns:
            Cubelist with a realization coordinate added to each cube if not already
            present.
        """
        index = 0
        for cube in cubelist:
            if not cube.coords("realization"):
                cube.add_aux_coord(
                    DimCoord(
                        np.array([index], dtype=np.int32),
                        standard_name="realization",
                        units="1",
                    ),
                )
                index += 1

        return cubelist

    def _rebadge_duplicate_realizations(self, cubelist: CubeList) -> CubeList:
        """
        If duplicate realization numbers exist across the cubelist,
        rebadge all realization numbers to a unique sequential set in input order.

        Args:
            cubelist: List of input forecasts.

        Returns:
            Cubelist with unique realization numbers across all cubes.
        """
        all_realizations = [cube.coord("realization").points for cube in cubelist]
        all_realizations = np.concatenate(all_realizations)
        unique_realizations = np.unique(all_realizations)

        if len(unique_realizations) < len(all_realizations):
            first_realization = 0
            for cube in cubelist:
                n_realization = len(cube.coord("realization").points)
                cube.coord("realization").points = np.arange(
                    first_realization, first_realization + n_realization, dtype=np.int32
                )
                first_realization += n_realization
        return cubelist

    def process(self, *cubes: Union[Cube, CubeList]) -> Cube:
        """
        Take an input cubelist containing forecasts from different cycles and
        merges them into a single cube.

        The steps taken are:
            1. Update forecast reference time and period to match the latest
               contributing cycle.
            2. Check for duplicate realization numbers. If a duplicate is
               found, renumber all of the realizations uniquely.
            3. Concatenate into one cube along the realization axis.

        Args:
            cubelist: List of input forecasts

        Returns:
            Concatenated forecasts

        Warns:
            UserWarning: If only a single cube is provided, so time lagging will have
            no effect.
        """
        cubelist = as_cubelist(cubes)
        cubelist = self.add_realization_coord(cubelist)

        if len(cubelist) == 1:
            warnings.warn(
                "Only a single cube input, so time lagging will have no effect."
            )
            return cubelist[0]

        # raise error if validity times are not all equal
        self._check_validity_times_match(cubelist)

        # Update all forecasts to have the same forecast reference time and
        # forecast period as the latest cycle. This faciliates merging.
        cubelist = rebadge_forecasts_as_latest_cycle(cubelist)

        # Check for duplicate realization numbers across the cubelist and rebadge
        # if necessary.
        cubelist = self._rebadge_duplicate_realizations(cubelist)

        # slice over realization to deal with cases where direct concatenation
        # would result in a non-monotonic coordinate
        lagged_ensemble = MergeCubes(slice_over_realization=True)(cubelist)

        if self.rebadge_realizations:
            n_realization = len(lagged_ensemble.coord("realization").points)
            lagged_ensemble.coord("realization").points = np.arange(
                n_realization, dtype=np.int32
            )

        return lagged_ensemble
