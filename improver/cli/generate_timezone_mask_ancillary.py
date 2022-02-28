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
"""Script to run GenerateTimezoneMask ancillary generation."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube,
    *,
    include_dst=False,
    time=None,
    groupings: cli.inputjson = None,
):
    """Generate a timezone mask ancillary.

    Create masks for regions of a geographic grid that are on different
    timezones. The resulting masks can be used to isolate data for specific
    timezones from a grid of the same shape as the cube used in this plugin
    to generate the masks.

    Args:
        cube (iris.cube.Cube):
            A cube with the desired grid. If no 'time' argument is provided
            to the CLI, the time on this cube will be used for determining the
            validity time of the calculated UTC offsets (this is only relevant
            if daylight savings times are being included).
        include_dst (bool):
            If set, find and use the UTC offset to a grid point including
            daylight savings.
        time (str):
            A datetime specified in the format YYYYMMDDTHHMMZ at which to
            calculate the mask (UTC). If daylight savings are not included
            this will have no impact on the resulting masks.
        groupings (dict):
            A dictionary specifying how timezones should be grouped if so
            desired. This dictionary takes the form::

                {-9: [-12, -6], 0: [-5, 5], 6: [6, 12]}

            The keys indicate the UTC offset that should provide the data for a
            group. The numbers in the lists denote the inclusive limits of the
            timezones for which that data should be used. This is of use if data
            is not available at hourly intervals.

    Returns:
        iris.cube.Cube:
            A cube containing timezone masks for each timezone found in the
            grid within the input cube, grouped into larger areas if so desired.
            Mask values of 1 indicate that a point is masked out, and values of
            0 indicate the point is unmasked.
    """
    from improver.generate_ancillaries.generate_timezone_mask import (
        GenerateTimezoneMask,
    )

    return GenerateTimezoneMask(
        include_dst=include_dst, time=time, groupings=groupings
    )(cube)
