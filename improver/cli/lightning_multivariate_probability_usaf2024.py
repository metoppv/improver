#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
"""Script to create lightning probabilities from multi-parameter datasets."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube,
    model_id_attr: str = None,
):
    """
    From the supplied following cubes:
    Convective Available Potential Energy (CAPE in J/kg),
    Lifted Index (liftind in K),
    Precipitable Water (pwat in kg m-2 or mm. This is used as mm in the regression equations),
    Convective Inhibition (CIN in J/kg),
    3-hour Accumulated Precipitation (apcp in kg m-2 or millimetres),
    calculate a probability of lightning cube using relationships developed using regression
    statistics.

    The cubes for CAPE, lifted index, precipitable water, and CIN must be valid for the beginning
    of the 3-hr accumulated precipitation window.

    Does not collapse a realization coordinate.

    Args:
        cubes (list of iris.cube.Cube):
            Cubes to be processed.
        model_id_attr (str):
            Name of the attribute used to identify the source model for
            blending.

    Returns:
        iris.cube.Cube:
            Cube of probabilities of lightning
    """
    from iris.cube import CubeList

    from improver.lightning import LightningMultivariateProbability_USAF2024

    result = LightningMultivariateProbability_USAF2024()(
        CubeList(cubes), model_id_attr=model_id_attr
    )

    return result
