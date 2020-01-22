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
"""Script to collapse cube coordinates and calculate percentiled data."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube,
            *,
            coordinates: cli.comma_separated_list = None,
            ignore_ecc_bounds=False,
            percentiles: cli.comma_separated_list = None,
            percentiles_count: int = None):
    r"""Collapses cube coordinates and calculate percentiled data.

    Calculate percentiled data over a given coordinate by collapsing that
    coordinate. Typically used to convert realization data into percentiled
    data, but may calculate over any dimension coordinate. Alternatively
    calling this with a dataset containing probabilities will convert those
    to percentiles using the ensemble coupla coupling plugin. If no particular
    percentiles are given at which to calculate values and no
    'number of percentiles' to calculate are specified, the
    following defaults will be used.
    '[0, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 100]'

    Args:
        cube (iris.cube.Cube):
            A Cube for processing.
        coordinates (str or list):
            Coordinate or coordinates over which to collapse data and
            calculate percentiles; e.g. 'realization' or 'latitude,longitude'.
            This argument must be provided when collapsing a coordinate or
            coordinates to create percentiles, but is redundant when
            converting probabilities to percentiles and may be omitted. This
            coordinate(s) will be removed and replaced by a percentile
            coordinate.
        ignore_ecc_bounds (bool):
            If True, where calculated percentiles are outside the ECC bounds
            range, raises a warning rather than an exception.
        percentiles (list):
            Optional definition of percentiles at which to calculate data.
        percentiles_count (int):
            Optional definition of the number of percentiles to be generated,
            these distributed regularly with the aim of dividing into blocks
            of equal probability.

    Returns:
        iris.cube.Cube:
            The processed Cube.

    Raises:
        ValueError:
            If the cube name does not contain 'probability_of\_' and
            coordinates isn't used.

    Warns:
        Warning:
            If 'probability_of\_' is in the cube name and coordinates is used.

    """
    import warnings

    import numpy as np

    from improver.ensemble_copula_coupling.ensemble_copula_coupling import \
        ConvertProbabilitiesToPercentiles
    from improver.ensemble_copula_coupling.utilities \
        import choose_set_of_percentiles
    from improver.percentile import PercentileConverter

    if percentiles is not None:
        percentiles = [float(p) for p in percentiles]

    if percentiles_count is not None:
        percentiles = choose_set_of_percentiles(percentiles_count,
                                                sampling="quantile")
    # TODO: Correct when formal cf-standards exists
    if 'probability_of_' in cube.name():
        result = ConvertProbabilitiesToPercentiles(
            ecc_bounds_warning=ignore_ecc_bounds).process(
            cube, percentiles=percentiles)
        if coordinates:
            warnings.warn("Converting probabilities to percentiles. The "
                          "provided COORDINATES_TO_COLLAPSE variable will "
                          "not be used.")
    else:
        if not coordinates:
            raise ValueError("To collapse a coordinate to calculate "
                             "percentiles, a coordinate or list of "
                             "coordinates must be provided.")

        # Switch back to use the slow scipy method if the cube contains masked
        # data which the numpy method cannot handle.
        fast_percentile_method = True

        if np.ma.is_masked(cube.data):
            # Check for masked points:
            fast_percentile_method = False
        elif np.ma.isMaskedArray(cube.data):
            # Check if we have a masked array with an empty mask. If so,
            # replace it with a non-masked array:
            cube.data = cube.data.data

        result = PercentileConverter(
            coordinates, percentiles=percentiles,
            fast_percentile_method=fast_percentile_method).process(cube)
    return result
