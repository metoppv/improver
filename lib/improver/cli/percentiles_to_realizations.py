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

"""Script to run Ensemble Copula Coupling processing."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube,
            raw_cube: cli.inputcube = None,
            *,
            realizations_count: int = None,
            sampling_method='quantile',
            ignore_ecc_bounds=False,
            randomise=False,
            random_seed: int = None,
            realizations: cli.comma_separated_list = None):
    """Convert percentiles to ensemble realizations using Ensemble Coupla
    Coupling.

    Percentiles are either rebadged as realizations or reordered if the
    raw_cube argument is given.

    Args:
        cube (iris.cube.Cube):
            Cube expected to contain a percentiles coordinate.
        raw_cube (iris.cube.Cube):
            Cube of raw (not post processed) weather data.
            If this argument is given ensemble realizations will be created
            from percentiles by reshuffling them in correspondance to the rank
            order of the raw ensemble. Otherwise, the percentiles are rebadged
            as realizations.
        realizations_count (int):
            The number of percentiles to be generated. This is also equal to
            the number of ensemble realizations that will be generated.
        sampling_method (str):
            Method to be used for generating the list of percentiles with
            forecasts generated at each percentile. The options are "quantile"
            and "random".
            The quantile option produces equally spaced percentiles which is
            the preferred option for full ensemble couple coupling with
            reorder enabled.
        ignore_ecc_bounds (bool):
            If True where percentiles (calculated as an intermediate output
            before realization) exceed the ECC bounds range, raises a
            warning rather than an exception.
        randomise (bool):
            Reorder randomly, rather than using the rank order of the raw_cube.
        random_seed (int):
            Option to specify a value for the random seed for testing purposes,
            otherwise, the default random seed behaviour is utilised.
            The random seed is used in the generation of the random numbers
            used for either the randomise option to order the input
            percentiles randomly, rather than use the ordering from the
            raw ensemble, or for splitting tied values within the raw ensemble
            so that the values from the input percentiles can be ordered to
            match the raw ensemble.
        realizations (list of ints):
            A list of ensemble realization numbers to use when rebadging the
            percentiles into realizations.

    Returns:
        iris.cube.Cube:
            The processed Cube.
    """
    from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
        RebadgePercentilesAsRealizations, ResamplePercentiles,
        EnsembleReordering)

    if realizations:
        realizations = [int(x) for x in realizations]

    result = ResamplePercentiles(
        ecc_bounds_warning=ignore_ecc_bounds).process(
        cube, no_of_percentiles=realizations_count,
        sampling=sampling_method)

    if raw_cube:
        result = EnsembleReordering().process(
            result, raw_cube, random_ordering=randomise,
            random_seed=random_seed)
    else:
        result = RebadgePercentilesAsRealizations().process(
            result, ensemble_realization_numbers=realizations)

    return result
