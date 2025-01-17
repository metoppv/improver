# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Provide support utilities for time lagging ensembles"""

import warnings
from typing import Union

import numpy as np
from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.metadata.forecast_times import rebadge_forecasts_as_latest_cycle
from improver.utilities.common_input_handle import as_cubelist
from improver.utilities.cube_manipulation import MergeCubes


class GenerateTimeLaggedEnsemble(BasePlugin):
    """Combine realizations from different forecast cycles into one cube"""

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
            cubelist:
                List of input forecasts

        Returns:
            Concatenated forecasts
        """
        cubelist = as_cubelist(cubes)
        if len(cubelist) == 1:
            warnings.warn(
                "Only a single cube input, so time lagging will have no effect."
            )
            return cubelist[0]

        # raise error if validity times are not all equal
        time_coords = [cube.coord("time") for cube in cubelist]
        time_coords_match = [coord == time_coords[0] for coord in time_coords]
        if not all(time_coords_match):
            raise ValueError("Cubes with mismatched validity times are not compatible.")

        cubelist = rebadge_forecasts_as_latest_cycle(cubelist)

        # Take all the realizations from all the input cube and
        # put in one array
        all_realizations = [cube.coord("realization").points for cube in cubelist]
        all_realizations = np.concatenate(all_realizations)
        # Find unique realizations
        unique_realizations = np.unique(all_realizations)

        # If we have fewer unique realizations than total realizations we have
        # duplicate realizations so we rebadge all realizations in the cubelist
        if len(unique_realizations) < len(all_realizations):
            first_realization = 0
            for cube in cubelist:
                n_realization = len(cube.coord("realization").points)
                cube.coord("realization").points = np.arange(
                    first_realization, first_realization + n_realization, dtype=np.int32
                )
                first_realization = first_realization + n_realization

        # slice over realization to deal with cases where direct concatenation
        # would result in a non-monotonic coordinate
        lagged_ensemble = MergeCubes()(cubelist, slice_over_realization=True)

        return lagged_ensemble
