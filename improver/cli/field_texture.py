# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
"""Script to calculate whether or not the input field texture exceeds a given threshold."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube,
    *,
    nbhood_radius: float,
    textural_threshold: float,
    diagnostic_threshold: float,
):

    """Calculates field texture for a given neighbourhood radius.

    Args:
        cube (iris.cube.Cube):
            Input data cube that will have a mixture of sunny and cloudy intervals.

        nbhood_radius (float):
            The neighbourhood radius in metres within which the number of potential
            transitions should be calculated. This forms the denominator in the
            calculation of the ratio of actual to potential transitions that indicates a
            field's texture. A larger radius should be used for diagnosing larger-scale
            textural features.

        textural_threshold (float):
            A unit-less threshold value that defines the ratio value above which
            the field is considered rough and below which the field is considered
            smoother.

        diagnostic_threshold (float):
            A user defined threshold value related either to cloud or precipitation,
            used to extract the corresponding dimensional cube with assumed units of 1.

    Returns:
        iris.cube.Cube:
                A cube containing the mean of the thresholded ratios in cube
                format.

    """

    from improver.field_texture import FieldTexture

    field_texture = FieldTexture(
        nbhood_radius, textural_threshold, diagnostic_threshold
    )(cube)
    return field_texture
