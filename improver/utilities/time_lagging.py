# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Provide support utilities for time lagging ensembles"""

import numpy as np

from improver import BasePlugin
from improver.metadata.forecast_times import rebadge_forecasts_as_latest_cycle
from improver.utilities.cube_manipulation import concatenate_cubes


class GenerateTimeLaggedEnsemble(BasePlugin):
    """Combine realizations from different forecast cycles into one cube"""

    def process(self, cubelist):
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
            cubelist (iris.cube.CubeList or list of iris.cube.Cube):
                List of input forecasts

        Returns:
            iris.cube.Cube:
                Concatenated forecasts
        """
        cubelist = rebadge_forecasts_as_latest_cycle(cubelist)

        # Take all the realizations from all the input cube and
        # put in one array
        all_realizations = [
            cube.coord("realization").points for cube in cubelist]
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
                    first_realization, first_realization + n_realization,
                    dtype=np.int32)
                first_realization = first_realization + n_realization

        # slice over realization to deal with cases where direct concatenation
        # would result in a non-monotonic coordinate
        lagged_ensemble = concatenate_cubes(
            cubelist, master_coord="realization",
            coords_to_slice_over=["realization"])

        return lagged_ensemble
