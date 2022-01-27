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
import iris
from iris.cube import Cube, CubeList

from improver import BasePlugin


class FilterRealizations(BasePlugin):
    """For a given list of cubes, identifies the set of times, filters out any realizations
    that are not present at all times and returns a merged cube of the result."""

    def process(self, cubes: CubeList) -> Cube:
        """For a given list of cubes, identifies the set of times, filters out any realizations
        that are not present at all times and returns a merged cube of the result."""
        times = set()
        realizations = set()
        for cube in cubes:
            times.update([c.point for c in cube.coord("time").cells()])
            realizations.update(cube.coord("realization").points)
        filtered_cubes = CubeList()
        for realization in realizations:
            realization_cube = cubes.extract(
                iris.Constraint(realization=realization)
            ).merge_cube()
            if set([c.point for c in realization_cube.coord("time").cells()]) == times:
                filtered_cubes.append(realization_cube)
        return filtered_cubes.merge_cube()
