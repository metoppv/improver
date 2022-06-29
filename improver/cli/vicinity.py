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
"""Script to apply vicinity neighbourhoods to data."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube, vicinity: float):
    """Module to apply vicinity processing to data.

    Calculate the neighbourhood maximum value about each point in the field
    within a vicinity radius. This step must be applied prior to collapsing
    any realization coordinate if being used to calculate a neighbourhood
    maximum ensemble probability.

    The threshold CLI is typically used to threshold, vicinity process, and
    collapse realizations in a single call. This CLI provides an alternative
    for non-standard cases where the threshold CLI is not used.

    Users should ensure they do not inadvertently apply vicinity processing
    twice, once within the threshold CLI and then again using this CLI.

    Args:
        cube (iris.cube.Cube):
            A cube containing data to which a vicinity it to be applied.
        vicinity (float):
            Distance in metres used to define the vicinity within which to
            search for an occurrence

    Returns:
        iris.cube.Cube:
            Cube with the vicinity processed data.
    """
    from improver.metadata.probabilistic import in_vicinity_name_format
    from improver.utilities.spatial import OccurrenceWithinVicinity

    result = OccurrenceWithinVicinity(radius=vicinity).process(cube)
    result.rename(in_vicinity_name_format(result.name()))

    return result
