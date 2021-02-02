#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""Script to apply lapse rates to temperature data."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    temperature: cli.inputcube,
    lapse_rate: cli.inputcube,
    source_orography: cli.inputcube,
    target_orography: cli.inputcube,
):
    """Apply downscaling temperature adjustment using calculated lapse rate.

    Args:
        temperature (iris.cube.Cube):
            Input temperature cube.
        lapse_rate (iris.cube.Cube):
            Lapse rate cube.
        source_orography (iris.cube.Cube):
            Source model orography.
        target_orography (iris.cube.Cube):
            Target orography to which temperature will be downscaled.

    Returns:
        iris.cube.Cube:
            Temperature cube after lapse rate adjustment has been applied.
    """
    from improver.lapse_rate import ApplyGriddedLapseRate

    # apply lapse rate to temperature data
    result = ApplyGriddedLapseRate()(
        temperature, lapse_rate, source_orography, target_orography
    )
    return result
