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
"""Calculation of expected value from a probability distribution."""

import iris.analysis
from iris.cube import Cube
from iris.coords import CellMethod

from improver import PostProcessingPlugin
from improver.metadata.probabilistic import is_percentile, is_probability
from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    RebadgePercentilesAsRealizations,
)
from improver.utilities.cube_manipulation import collapsed


class ExpectedValue(PostProcessingPlugin):
    """Calculation of expected value from a probability distribution"""

    def process(self, cube: Cube) -> Cube:
        """Expected value calculation and metadata updates.

        Args:
            cube:
                Probabilistic data with a realization or percentile representation.

        Returns:
            Expected value of probability distribution. Same shape as input cube
            but with realization/percentile coordinate removed.
        """
        if is_percentile(cube):
            cube = RebadgePercentilesAsRealizations().process(cube)
        if is_probability(cube):
            raise NotImplementedError(
                "Handling of threshold cubes is not yet implemented"
            )
        mean_cube = collapsed(cube, "realization", iris.analysis.MEAN)
        mean_cube.remove_coord("realization")
        mean_cube.add_cell_method(CellMethod("mean", coords="realization"))
        return mean_cube
