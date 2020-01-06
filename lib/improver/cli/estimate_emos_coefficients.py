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
"""Script to estimate coefficients for Ensemble Model Output
Statistics (EMOS), otherwise known as Non-homogeneous Gaussian
Regression (NGR)."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube,
            distribution,
            truth_attribute,
            cycletime=None,
            units=None,
            predictor_of_mean='mean',
            tolerance: float = 1,
            max_iterations: int = 1000):
    """Estimate coefficients for Ensemble Model Output Statistics.

    Loads in arguments for estimating coefficients for Ensemble Model
    Output Statistics (EMOS), otherwise known as Non-homogeneous Gaussian
    Regression (NGR). Two sources of input data must be provided: historical
    forecasts and historical truth data (to use in calibration).
    The estimated coefficients are output as a cube.

    Args:
        cubes (list of iris.cube.Cube):
            A list of cubes containing the historical forecasts and
            corresponding truth used for calibration. Optionally this may also
            contain a land-sea mask cube on the same domain as the historic
            forecasts and truth (where land points are set to one and sea
            points are set to zero).
        distribution (str):
            The distribution that will be used for calibration. This will be
            dependant upon the input phenomenon.
        truth_attribute (str):
            A string of the form "attribute=value" which specifies which
            attribute, value pair in the list of cubes corresponds to the
            forecast truth.
        cycletime (str):
            This denotes the cycle at which forecasts will be calibrated using
            the calculated EMOS coefficients. The validity time in the output
            coefficients cube will be calculated relative to this cycletime.
            This cycletime is in the format YYYYMMDDTHHMMZ.
        units (str):
            The units that calibration should be undertaken in. The historical
            forecast and truth will be converted as required.
            Default is None.
        predictor_of_mean (str):
            String to specify the input to calculate the calibrated mean.
            Currently the ensemble mean ("mean") and the ensemble realizations
            ("realizations") are supported as the predictors.
            Default is 'mean'.
        tolerance (float):
            The tolerance for the Continuous Ranked Probability Score (CRPS)
            calculated by the minimisation. Once multiple iterations result in
            a CRPS equal to the same value within the specified tolerance, the
            minimisation will terminate.
        max_iterations (int):
            The maximum number of iterations allowed until the minimisation has
            converged to a stable solution. If the maximum number of iterations
            is reached but the minimisation has not yet converged to a stable
            solution, then the available solution is used anyway, and a warning
            is raised.
            If the predictor_of_mean is "realizations", then the number of
            iterations may require increasing, as there will be more
            coefficients to solve.
            Default is 1000.

    Returns:
        iris.cube.Cube:
            Cube containing the coefficients estimated using EMOS. The cube
            contains a coefficient_index dimension coordinate and a
            coefficient_name auxiliary coordinate.

    Raises:
        RuntimeError:
            An unexpected number of distinct cube names were passed in.
        RuntimeError:
            More than one cube was identified as a land-sea mask.
        RuntimeError:
            Missing truth and/or historical forcast in input cubes.

    """

    from collections import OrderedDict
    from improver.utilities.cube_manipulation import MergeCubes
    from improver.ensemble_calibration.ensemble_calibration import (
        EstimateCoefficientsForEnsembleCalibration)

    grouped_cubes = {}
    for cube in cubes:
        grouped_cubes.setdefault(cube.name(), []).append(cube)
    if len(grouped_cubes) == 1:
        # Only one group - all forecast/truth cubes
        landsea_mask = None
        diag_name = list(grouped_cubes.keys())[0]
    elif len(grouped_cubes) == 2:
        # Two groups - the one with exactly one cube matching a name should
        # be the landmask, since we require >= 2 cubes in the forecast/truth
        # group
        grouped_cubes = OrderedDict(sorted(grouped_cubes.items(),
                                           key=lambda kv: len(kv[1])))
        # landsea name should be the key with the lowest number of cubes (1)
        landsea_name, diag_name = list(grouped_cubes.keys())
        landsea_mask = grouped_cubes[landsea_name][0]
        if len(grouped_cubes[landsea_name]) != 1:
            raise RuntimeError('Expected one cube for land-sea mask.')
    else:
        raise RuntimeError('Must have cubes with 1 or 2 distinct names.')

    # split non-landmask cubes on forecast vs truth
    truth_key, truth_value = truth_attribute.split('=')
    input_cubes = grouped_cubes[diag_name]
    grouped_cubes = {'truth': [], 'historical forecast': []}
    for cube in input_cubes:
        if cube.attributes.get(truth_key) == truth_value:
            grouped_cubes['truth'].append(cube)
        else:
            grouped_cubes['historical forecast'].append(cube)

    missing_inputs = ' and '.join(k for k, v in grouped_cubes.items() if not v)
    if missing_inputs:
        raise RuntimeError('Missing ' + missing_inputs + ' input.')

    truth = MergeCubes()(grouped_cubes['truth'])
    forecast = MergeCubes()(grouped_cubes['historical forecast'])

    return EstimateCoefficientsForEnsembleCalibration(
        distribution, cycletime, desired_units=units,
        predictor_of_mean_flag=predictor_of_mean,
        tolerance=tolerance, max_iterations=max_iterations).process(
            forecast, truth, landsea_mask=landsea_mask)
