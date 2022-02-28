# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2022 Met Office.
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
    nbhood_radius: float = 20000.0,
    textural_threshold: float = 0.05,
    diagnostic_threshold: float = 0.8125,
    model_id_attr: str = None,
):

    """Calculates field texture for a given neighbourhood radius.

       The field texture is an assessment of the transitions/edges within a
       neighbourhood of a grid point to indicate whether the field is rough
       or smooth.

    Args:
        cube (iris.cube.Cube):
            The diagnostic for which texture is to be assessed. For example cloud
            area fraction where transitions between cloudy regions and cloudless
            regions will be diagnosed. Defaults set assuming cloud area fraction
            cube.

        nbhood_radius (float):
            The neighbourhood radius in metres within which the number of potential
            transitions should be calculated. This forms the denominator in the
            calculation of the ratio of actual to potential transitions that indicates a
            field's texture. A larger radius should be used for diagnosing larger-scale
            textural features. Default value set to 10000.0 assuming cloud area fraction
            cube.

        textural_threshold (float):
            A unit-less threshold value that defines the ratio value above which
            the field is considered rough and below which the field is considered
            smoother. Default value set to 0.05 assuming cloud area fraction cube.

        diagnostic_threshold (float):
            The diagnostic threshold for which field texture will be calculated.
            A ValueError is raised if this threshold is not present on the input
            cube. Default value set to 0.8125 corresponding to 6 oktas, assuming
            cloud area fraction cube.

        model_id_attr (str):
            Name of the attribute used to identify the source model for
            blending.

    Returns:
        iris.cube.Cube:
            A field texture cube containing values between 0 and 1, where 0
            indicates no transitions and 1 indicates the maximum number of
            possible transitions has been achieved, within the neighbourhood of
            the grid point.
    """

    from improver.utilities.textural import FieldTexture

    field_texture = FieldTexture(
        nbhood_radius,
        textural_threshold,
        diagnostic_threshold,
        model_id_attr=model_id_attr,
    )(cube)
    return field_texture
