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
"""Module containing a plugin to calculate the modal weather symbol in a period."""

import numpy as np
from iris.analysis import Aggregator
from iris.cube import Cube
from numpy import ndarray
from scipy import stats

from improver import BasePlugin

SYMBOL_MAX = 100


# @dataclass
class ModalSymbol(BasePlugin):
    """Plugin that returns the modal symbol over the period spanned by the
    input data. In cases of a tie in the mode values, scipy returns the smaller
    value. The opposite is desirable in this case as the significance /
    importance of the weather symbols generally increases with the value. To
    achieve this the symbol codes are subtracted from an arbitrarily larger
    number prior to calculating the mode, and this operation reversed in the
    final output."""

    @staticmethod
    def mode_aggregator(data: ndarray, axis: int):
        # Iris aggregators support indexing from the end of the array.
        if axis < 0:
            axis += data.ndim
        # Aggregation coordinate is moved to the -1 position in initialisation;
        # move this back to the leading coordinate
        data = np.moveaxis(data, [axis], [0])
        mode_result, _ = stats.mode(SYMBOL_MAX - data, axis=0)
        return SYMBOL_MAX - np.squeeze(mode_result)

    def process(self, cube: Cube):
        """Calculate the modal symbol."""

        mode_aggregator = Aggregator("mode", self.mode_aggregator)
        result = cube.collapsed("time", mode_aggregator)
        result.coord('time').points = result.coord('time').bounds[0][-1]
        return result
