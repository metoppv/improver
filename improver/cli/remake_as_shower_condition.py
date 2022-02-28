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
"""Script to modify a suitable shower condition proxy diagnostic into a shower
condition cube."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube):
    """
    Modify the name and threshold coordinate of another diagnostic to create
    a shower condition cube. Such a cube provides the probability that any
    precipitation, should it be present, should be classified as showery. Only
    suitable proxies for identifying showery conditions should be modified in
    this way. By modifying cubes in this way it is possible to blend different
    proxies from different models as though they are equivalent diagnostics.
    The user must be satisfied that the proxies are suitable for blending.

    Args:
        cube (iris.cube.Cube):
            A cube containing the diagnostic that is a proxy for showery
            conditions, e.g. cloud texture.

    Returns:
        iris.cube.Cube:
            Probability of any precipitation, if present, being classified as
            showery.
    """
    from improver.precipitation_type.utilities import make_shower_condition_cube

    return make_shower_condition_cube(cube)
