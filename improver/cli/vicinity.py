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
def process(cube: cli.inputcube, vicinity: cli.comma_separated_list = None):
    """Module to apply vicinity processing to data.

    Calculate the maximum value within a vicinity radius about each point
    in each x-y slice of the input cube.

    If working with thresholded data, using this CLI to calculate a
    neighbourhood maximum ensemble probability, the vicinity process must
    be applied prior to averaging across the ensemble, i.e. collapsing the
    realization coordinate. A typical chain might look like:

      threshold --> vicinity --> realization collapse

    Note that the threshold CLI can be used to perform this series of steps
    in a single call without intermediate output.

    Users should ensure they do not inadvertently apply vicinity processing
    twice, once within the threshold CLI and then again using this CLI.

    Args:
        cube (iris.cube.Cube):
            A cube containing data to which a vicinity is to be applied.
        vicinity (list of float / int):
            List of distances in metres used to define the vicinities within
            which to search for an occurrence. Each vicinity provided will
            lead to a different gridded field.

    Returns:
        iris.cube.Cube:
            Cube with the vicinity processed data.
    """
    from improver.utilities.spatial import OccurrenceWithinVicinity

    vicinity = [float(x) for x in vicinity]
    return OccurrenceWithinVicinity(radii=vicinity).process(cube)
