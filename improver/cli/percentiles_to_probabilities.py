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
"""
Calculate probability values from percentile data and a 2D threshold field.
"""

from improver import cli


@cli.clizefy
@cli.with_output
def process(percentiles_cube: cli.inputcube,
            threshold_cube: cli.inputcube,
            *,
            output_diagnostic_name):
    r"""Probability from a percentiled field at a 2D threshold level.

    Probabilities are generated at a fixed threshold (height) from a set of
    (height) percentiles. E.g. for 2D percentile levels at different heights,
    calculate probability that height is at ground level, where the threshold
    cube contains a 2D topography field.

    Example::

        Snow-fall level:

            Reference field: Percentiled snow fall level (m ASL)
            Other field: Orography (m ASL)

            300m ----------------- 30th Percentile snow fall level
            200m ----_------------ 20th Percentile snow fall level
            100m ---/-\----------- 10th Percentile snow fall level
            000m --/---\----------  0th Percentile snow fall level
            ______/     \_________ Orogaphy

    The orography heights are compared against the heights that correspond with
    percentile values to find the band in which they fall, then interpolated
    linearly to obtain a probability at / below the ground surface.

    Args:
        percentiles_cube (iris.cube.Cube):
            The percentiled field from which probabilities will be obtained
            using the input cube.
            This cube should contain a percentiles dimension, with fields of
            values that correspond to these percentiles. The cube passed to
            the process method will contain values of the same diagnostic.
        threshold_cube (iris.cube.Cube):
            A cube of values that effectively behave as thresholds, for which
            it is desired to obtain probability values from a percentiled
            reference cube.
        output_diagnostic_name (str):
            The name of the cube being created, e.g
            'probability_of_snow_falling_level_below_ground_level'

    Returns:
        iris.cube.Cube:
            A cube of probabilities obtained by interpolating between
            percentile values at the "threshold" level.
    """
    from improver.utilities.statistical_operations import \
        ProbabilitiesFromPercentiles2D
    result = ProbabilitiesFromPercentiles2D(percentiles_cube,
                                            output_diagnostic_name)
    probability_cube = result.process(threshold_cube)
    return probability_cube
