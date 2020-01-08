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
            raw_forecast: cli.inputcube = None,
            *,
            no_of_percentiles: int = None,
            sampling_method='quantile',
            ecc_bounds_warning=False,
            reordering=False,
            rebadging=False,
            random_ordering=False,
            random_seed: int = None,
            realization_numbers: cli.comma_separated_list = None):
    """Runs Ensemble Copula Coupling processing.

    Converts a dataset containing percentiles into one containing ensemble
    realizations using Ensemble Coupla Coupling.

    Args:
        cube (iris.cube.Cube):
            Cube expected to contain a percentiles coordinate.
        raw_forecast (iris.cube.Cube):
            Cube of raw (not post processed) weather data.
            This option is compulsory, if the reordering option is selected.
        no_of_percentiles (int):
            The number of percentiles to be generated. This is also equal to
            the number of ensemble realizations that will be generated.
        sampling_method (str):
            Method to be used for generating the list of percentiles with
            forecasts generated at each percentile. The options are "quantile"
            and "random".
            The quantile option produces equally spaced percentiles which is
            the preferred option for full ensemble couple coupling with
            reordering enabled.
        ecc_bounds_warning (bool):
            If True where percentiles (calculated as an intermediate output
            before realization) exceed the ECC bounds range, raises a
            warning rather than an exception.
        reordering (bool):
            The option used to create ensemble realizations from percentiles
            by reordering the input percentiles based on the order of the
            raw ensemble forecast.
        rebadging (bool):
            The option used to create ensemble realizations from percentiles
            by rebadging the input percentiles.
        random_ordering (bool):
            If random_ordering is True, the post-processed forecasts are
            reordered randomly, rather than using the ordering of the
            raw ensemble.
        random_seed (int):
            Option to specify a value for the random seed for testing purposes,
            otherwise, the default random seed behaviour is utilised.
            The random seed is used in the generation of the random numbers
            used for either the random_ordering option to order the input
            percentiles randomly, rather than use the ordering from the
            raw ensemble, or for splitting tied values within the raw ensemble
            so that the values from the input percentiles can be ordered to
            match the raw ensemble.
        realization_numbers (list of ints):
            A list of ensemble realization numbers to use when rebadging the
            percentiles into realizations.

    Returns:
        iris.cube.Cube:
            The processed Cube.
    """
    from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
        RebadgePercentilesAsRealizations, ResamplePercentiles,
        EnsembleReordering)

    if reordering:
        if realization_numbers is not None:
            raise RuntimeError('realization_numbers cannot be used with '
                               'reordering.')
    if rebadging:
        if raw_forecast is not None:
            raise RuntimeError('rebadging cannot be used with raw_forecast.')
        if random_ordering is not False:
            raise RuntimeError('rebadging cannot be used with '
                               'random_ordering.')

    if realization_numbers:
        realization_numbers = [int(x) for x in realization_numbers]

    result = ResamplePercentiles(
        ecc_bounds_warning=ecc_bounds_warning).process(
        cube, no_of_percentiles=no_of_percentiles,
        sampling=sampling_method)
    if reordering:
        result = EnsembleReordering().process(
            result, raw_forecast, random_ordering=random_ordering,
            random_seed=random_seed)
    elif rebadging:
        result = RebadgePercentilesAsRealizations().process(
            result, ensemble_realization_numbers=realization_numbers)
    return result
