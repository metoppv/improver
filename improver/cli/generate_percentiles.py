#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to collapse cube coordinates and calculate percentiled data."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube,
    *,
    coordinates: cli.comma_separated_list = None,
    percentiles: cli.comma_separated_list = None,
    ignore_ecc_bounds_exceedance: bool = False,
    skip_ecc_bounds: bool = False,
    mask_percentiles: bool = False,
    optimal_crps_percentiles: bool = False,
):
    r"""Collapses cube coordinates and calculate percentiled data.

    Calculate percentiled data over a given coordinate by collapsing that
    coordinate. Typically used to convert realization data into percentiled
    data, but may calculate over any dimension coordinate. If no coordinate for
    collapsing over is provided, the realization data will instead be rebadged
    as percentile data. Alternatively calling this with a dataset containing
    probabilities will convert those to percentiles using the ensemble copula
    coupling plugin. If no particular percentiles are given at which to calculate
    values and no 'number of percentiles' to calculate are specified, the
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
            converting probabilities to percentiles and may be omitted. Not
            providing the coordinate(s) with realization data will cause the
            realizations to be rebadged as percentiles instead. This
            coordinate(s) will be removed and replaced by a percentile
            coordinate.
        percentiles (list):
            Optional definition of percentiles at which to calculate data.
        ignore_ecc_bounds_exceedance (bool):
            If True, where calculated percentiles are outside the ECC bounds
            range, raises a warning rather than an exception.
        skip_ecc_bounds (bool):
            If True, ECC bounds are not included when probabilities
            are converted to percentiles. This has the effect that percentiles
            outside of the range given by the input percentiles will be computed
            by nearest neighbour interpolation from the nearest available percentile,
            rather than using linear interpolation between the nearest available
            percentile and the ECC bound.
        mask_percentiles (bool):
            A boolean determining whether the final percentiles should
            be masked. If True then where the percentile is higher than
            the probability of the diagnostic existing the outputted
            percentile will be masked.
            This masking acts to cap the maximum percentile
            that can be generated for any grid point relative to the
            probability of exceeding the highest threshold at that point.
            For example, if the probability of a temperature exceeding 40C
            is 5%, the 95th and above percentiles will all be masked.
            Likewise, if at some grid point the probability of the cloud base
            being below 15000m (the highest threshold) is 70% then every
            percentile above the 70th would be masked.
            This is only implemented to work when converting probabilities to
            percentiles.
        optimal_crps_percentiles (bool):
            If True, percentiles are computed following the
            recommendation of Bröcker, 2012 for optimising the CRPS using
            the equation: q = (i-0.5)/N, i=1,...,N, where N is the number
            of realizations. If False, percentiles are computed as equally
            spaced following the equation: q = i/(1+N), i=1,...,N.
            Defaults to False.

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
        Warning:
            If 'probability_of\_' is not in the cube name and mask_percentiles is True.

    """
    import warnings

    import numpy as np

    from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
        ConvertProbabilitiesToPercentiles,
        RebadgeRealizationsAsPercentiles,
    )
    from improver.metadata.probabilistic import is_probability
    from improver.percentile import PercentileConverter

    if percentiles is not None:
        percentiles = [float(p) for p in percentiles]

    if mask_percentiles and not is_probability(cube):
        warnings.warn(
            "The option mask_percentiles is only implemented for generating percentiles from"
            "probability cubes and so will not be used."
        )

    if is_probability(cube):
        result = ConvertProbabilitiesToPercentiles(
            ecc_bounds_warning=ignore_ecc_bounds_exceedance,
            skip_ecc_bounds=skip_ecc_bounds,
            mask_percentiles=mask_percentiles,
        )(cube, percentiles=percentiles)
        if coordinates:
            warnings.warn(
                "Converting probabilities to percentiles. The "
                "provided COORDINATES_TO_COLLAPSE variable will "
                "not be used."
            )
    elif coordinates:
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
            coordinates,
            percentiles=percentiles,
            fast_percentile_method=fast_percentile_method,
        )(cube)
    else:
        result = RebadgeRealizationsAsPercentiles(
            optimal_crps_percentiles=optimal_crps_percentiles
        )(cube)

    return result
