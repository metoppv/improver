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
"""Script to map multiple forecast times into a local time grid"""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    timezone_cube: cli.inputcube, local_time: str, *cubes: cli.inputcube,
):
    """Calculates timezone-offset data for the specified UTC output times

    Args:
        timezone_cube (iris.cube.Cube):
            Cube describing the UTC offset for the local time at each grid location.
            Must have the same spatial coords as input_cube.
            Use generate-timezone-mask-ancillary to create this.
        local_time (str):
            The "local" time of the output cube as %Y%m%dT%H%M. This will form a
            scalar "time_in_local_timezone" coord on the output cube, while the "time"
            coord will be auxillary to the spatial coords and will show the UTC time
            that matches the local_time at each point.
        cubes (list of iris.cube.Cube):
            Source data to be remapped onto time-zones. Must contain an exact 1-to-1
            mapping of times to time-zones. Multiple input files will be merged into one
            cube.

    Returns:
        iris.cube.Cube:
            Processed cube.
    """
    from datetime import datetime

    from improver.utilities.temporal import TimezoneExtraction

    local_datetime = datetime.strptime(local_time, "%Y%m%dT%H%M")
    return TimezoneExtraction()(cubes, timezone_cube, local_datetime)
