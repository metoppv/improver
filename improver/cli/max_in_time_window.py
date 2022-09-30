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
"""Script to compute the maximum within a time window for a period diagnostic."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube, minimum_realizations=None):
    """Find the maximum probability or maximum diagnostic value within a time window
    for a period diagnostic. For example, find the maximum probability of exceeding
    a given accumulation threshold in a period e.g. 20 mm in 3 hours, over the course
    of a longer interval e.g. a 24 hour time window.

    Args:
        cubes (iris.cube.CubeList or list of iris.cube.Cube):
            Cubes over which to find the maximum.
        minimum_realizations (int):
            If specified, the input cubes will be filtered to ensure that only
            realizations that include all available lead times are combined. If the
            number of realizations that meet this criteria are fewer than this integer,
            an error will be raised.

    Returns:
        result (iris.cube.Cube):
            Returns a cube that is representative of a maximum within a time window
            for the period diagnostic supplied.
    """
    from iris.cube import CubeList

    from improver.cube_combiner import MaxInTimeWindow

    return MaxInTimeWindow(minimum_realizations=minimum_realizations)(CubeList(cubes))
