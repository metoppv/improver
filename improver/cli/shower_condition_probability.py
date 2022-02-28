#!/usr/bin/env python
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
"""Script to calculate a probability of precipitation being showery if present."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube,
    cloud_threshold: float,
    convection_threshold: float,
    model_id_attr: str = None,
):
    """
    Create a shower condition diagnostic that provides the probability that
    precipitation, if present, should be classified as showery. This shower
    condition is created from cloud area fraction and convective ratio fields.

    Args:
        cubes (iris.cube.CubeList):
            Cubes of cloud area fraction and convective ratio that are used
            to calculate a proxy probability that conditions are suitable for
            showery precipitation.
        cloud_threshold (float):
            The cloud area fraction value at which to threshold the input cube.
            This sets the amount of cloud coverage below which conditions are
            likely to be considered showery (i.e. precipitation between sunny
            spells).
        convection_threshold (float):
            The convective ratio value above which, despite the cloud conditions
            not suggesting showers, the precipitation is so clearly derived
            from convection that it should be classified as showery.
        model_id_attr (str):
            Name of the attribute used to identify the source model for
            blending (optional).

    Returns:
        iris.cube.Cube:
            Probability of any precipitation, if present, being classified as
            showery.
    """
    from iris.cube import CubeList

    from improver.precipitation_type.shower_condition_probability import (
        ShowerConditionProbability,
    )

    return ShowerConditionProbability(
        cloud_threshold=cloud_threshold,
        convection_threshold=convection_threshold,
        model_id_attr=model_id_attr,
    )(CubeList(cubes))
