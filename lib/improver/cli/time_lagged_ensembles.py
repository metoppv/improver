#!/usr/bin/env python
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
"""Script to run time-lagged ensembles."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube):
    """Module to time-lag ensembles.

    Combines the realization from different forecast cycles into one cube.
    Takes an input CubeList containing forecasts from different cycles and
    merges them into a single cube.

    Args:
        cubes (list of iris.cube.Cube):
            List of individual ensemble cubes

    Returns:
        iris.cube.Cube:
            Merged cube

    Raises:
        ValueError: If ensembles have mismatched validity times
    """
    import warnings
    from improver.utilities.time_lagging import GenerateTimeLaggedEnsemble

    if len(cubes) == 1:
        warnings.warn('Only a single cube input, so time lagging will have '
                      'no effect.')
        return cubes[0]

    # raise error if validity times are not all equal
    time_coords = [cube.coord("time") for cube in cubes]
    time_coords_match = [coord == time_coords[0] for coord in time_coords]
    if not all(time_coords_match):
        raise ValueError(
            "Cubes with mismatched validity times are not compatible.")

    return GenerateTimeLaggedEnsemble().process(cubes)
