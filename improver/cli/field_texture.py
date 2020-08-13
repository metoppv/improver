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
"""Script to calculate cloud clumpiness using field texture and edge transitions"""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube, *, nbhood_radius: float, ratio_threshold: float = 0.05
):

    """Calculates cloud texture for a given neighbourhood radius.

    Args:
        cube (iris.cube.Cube):
            Input data cube that will have a mixture of sunny and cloudy intervals.

        nbhood_radius (float):
                A neighbourhood radius of sufficient size to capture the region and
                all actual transitions, in metres.

        ratio_threshold (float):
                A threshold to re-normalise values about a sensible value.

    Returns:
        clumpiness (iris.cube.Cube):
            A cube of binary data, where 1 represents sunlight and 0 represents cloud.

    """

    from improver.utilities.field_texture import FieldTexture

    field_texture = FieldTexture()(cube, nbhood_radius, ratio_threshold)

    return field_texture
