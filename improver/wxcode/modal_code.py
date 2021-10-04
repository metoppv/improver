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
"""Module containing a plugin to calculate the modal weather code in a period."""

import numpy as np
from iris.analysis import Aggregator
from iris.cube import Cube, CubeList
from numpy import ndarray
from scipy import stats

from improver import BasePlugin
from improver.utilities.cube_manipulation import MergeCubes

from .utilities import DAYNIGHT_CODES, GROUPED_CODES

CODE_MAX = 100
UNSET_CODE_INDICATOR = -99


class ModalWeatherCode(BasePlugin):
    """Plugin that returns the modal code over the period spanned by the
    input data. In cases of a tie in the mode values, scipy returns the smaller
    value. The opposite is desirable in this case as the significance /
    importance of the weather codes generally increases with the value. To
    achieve this the codes are subtracted from an arbitrarily larger
    number prior to calculating the mode, and this operation reversed in the
    final output.

    If there are many different codes for a single point over the time
    spanned by the input cubes it may be that the returned mode is not robust.
    Given the preference to return more significant codes explained above,
    a 12 hour period with 12 different codes, one of which is thunder, will
    return a thunder code to describe the whole period. This is likely not a
    good representation. In these cases grouping is used to try and select
    a suitable weather code (e.g. a rain shower if the codes include a mix of
    rain showers and dynamic rain) by providing a more robust mode.
    """

    def __init__(self):
        """Create an aggregator instance for reuse"""
        self.aggregator_instance = Aggregator("mode", self.mode_aggregator)

    @staticmethod
    def unify_day_and_night(cube: Cube):
        """Remove distinction between day and night codes so they can each
        contribute when calculating the modal code. The cube of weather
        codes is modified in place with all night codes made into their
        daytime equivalents.

        Args:
            A cube of weather codes.
        """
        night_codes = np.array(DAYNIGHT_CODES) - 1
        for code in night_codes:
            cube.data[cube.data == code] += 1

    @staticmethod
    def group_codes(modal: Cube, cube: Cube):
        """In instances where the mode returned is not significant, i.e. the
        weather code chosen occurs infrequently in the period, the codes can be
        grouped to yield a more definitive period code. Given the uncertainty
        the least significant weather type (lowest number in a group that is
        found in the data) is used to replace the other data values that belong
        to that group prior to recalculating the modal code.

        The modal code is modified in place.

        Args:
            modal:
                The modal weather code cube which contains UNSET_CODE_INDICATOR
                values that need to be replaced with a more definitive period
                code.
            cube:
                The original input data. Data relating to unset points will be
                grouped and the mode recalculated."""

        undecided_points = np.argwhere(modal.data == UNSET_CODE_INDICATOR)

        for y, x in undecided_points:
            data = cube.data[:, y, x].copy()

            for _, codes in GROUPED_CODES.items():
                default_code = sorted([code for code in data if code in codes])
                if default_code:
                    data[np.isin(data, codes)] = default_code[0]
            mode_result, counts = stats.mode(CODE_MAX - data)
            modal.data[y, x] = CODE_MAX - mode_result

    @staticmethod
    def mode_aggregator(data: ndarray, axis: int) -> ndarray:
        """An aggregator for use with iris to calculate the mode along the
        specified axis. If the modal value selected comprises less than 10%
        of data along the dimension being collapsed, the value is set to the
        UNSET_CODE_INDICATOR to indicate that the uncertainty was too high to
        return a mode.

        Args:
            data:
                The data for which a mode is to be calculated.
            axis:
                The axis / dimension over which to calculate the mode.

        Returns:
            The data array collapsed over axis, containing the calculated modes.
        """
        # Iris aggregators support indexing from the end of the array.
        if axis < 0:
            axis += data.ndim
        # Aggregation coordinate is moved to the -1 position in initialisation;
        # move this back to the leading coordinate
        data = np.moveaxis(data, [axis], [0])
        minimum_significant_count = 0.1 * data.shape[0]
        mode_result, counts = stats.mode(CODE_MAX - data, axis=0)
        mode_result[counts < minimum_significant_count] = (
            CODE_MAX - UNSET_CODE_INDICATOR
        )
        return CODE_MAX - np.squeeze(mode_result)

    def process(self, cubes: CubeList) -> Cube:
        """Calculate the modal weather code, with handling for edge cases.

        Args:
            cubes:
                A list of weather code cubes at different times. A modal
                code will be calculated over the time coordinate to return
                the most comon code, which is taken to be the best
                representation of the whole period.

        Returns:
            A single weather code cube with time bounds that span those of
            the input weather code cubes.
        """
        # Handle case in which a single time is provided.
        if len(cubes) == 1:
            return cubes[0]

        cube = MergeCubes()(cubes)
        self.unify_day_and_night(cube)

        result = cube.collapsed("time", self.aggregator_instance)
        result.coord("time").points = result.coord("time").bounds[0][-1]

        # Handle any unset points where it was hard to determine a suitable mode
        if (result.data == UNSET_CODE_INDICATOR).any():
            self.group_codes(result, cube)

        return result
