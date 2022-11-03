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
"""Module containing a plugin to calculate the modal weather code in a period."""

from typing import Optional

import iris
import numpy as np
from iris.analysis import Aggregator
from iris.cube import Cube, CubeList
from numpy import ndarray
from scipy import stats

from improver import BasePlugin
from improver.blending import RECORD_COORD
from improver.blending.utilities import apply_record_run_attr, store_record_run_attr
from improver.utilities.cube_manipulation import MergeCubes

from ..metadata.forecast_times import forecast_period_coord
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
    rain showers and dynamic rain) by providing a more robust mode. The lowest
    number (least significant) member of the group is returned as the code.
    Use of the least significant member reflects the lower certainty in the
    forecasts.

    Where there are different weather codes available for night and day, the
    modal code returned is always a day code, regardless of the times
    covered by the input files.
    """

    def __init__(
        self, model_id_attr: Optional[str] = None, record_run_attr: Optional[str] = None
    ):
        """
        Set up plugin and create an aggregator instance for reuse

        Args:
            model_id_attr:
                Name of attribute recording source models that should be
                inherited by the output cube. The source models are expected as
                a space-separated string.
            record_run_attr:
                Name of attribute used to record models and cycles used in
                constructing the weather symbols.
        """
        self.aggregator_instance = Aggregator("mode", self.mode_aggregator)

        self.model_id_attr = model_id_attr
        self.record_run_attr = record_run_attr

    @staticmethod
    def _unify_day_and_night(cube: Cube):
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
    def _group_codes(modal: Cube, cube: Cube):
        """In instances where the mode returned is not significant, i.e. the
        weather code chosen occurs infrequently in the period, the codes can be
        grouped to yield a more definitive period code. Given the uncertainty,
        the least significant weather type (lowest number in a group that is
        found in the data) is used to replace the other data values that belong
        to that group prior to recalculating the modal code.

        The modal cube is modified in place.

        Args:
            modal:
                The modal weather code cube which contains UNSET_CODE_INDICATOR
                values that need to be replaced with a more definitive period
                code.
            cube:
                The original input data. Data relating to unset points will be
                grouped and the mode recalculated."""

        undecided_points = np.argwhere(modal.data == UNSET_CODE_INDICATOR)

        for point in undecided_points:
            data = cube.data[(..., *point)].copy()

            for _, codes in GROUPED_CODES.items():
                default_code = sorted([code for code in data if code in codes])
                if default_code:
                    data[np.isin(data, codes)] = default_code[0]
            mode_result, counts = stats.mode(CODE_MAX - data)
            modal.data[tuple(point)] = CODE_MAX - mode_result

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

    @staticmethod
    def _set_blended_times(cube: Cube) -> None:
        """Updates time coordinates so that time point is at the end of the time bounds,
        blend_time and forecast_reference_time (if present) are set to the end of the
        bound period and bounds are removed, and forecast_period is updated to match."""
        cube.coord("time").points = cube.coord("time").bounds[0][-1]

        for coord_name in ["blend_time", "forecast_reference_time"]:
            if coord_name in [c.name() for c in cube.coords()]:
                coord = cube.coord(coord_name)
                if coord.has_bounds():
                    coord = coord.copy(coord.bounds[0][-1])
                    cube.replace_coord(coord)

        if "forecast_period" in [c.name() for c in cube.coords()]:
            calculated_coord = forecast_period_coord(
                cube, force_lead_time_calculation=True
            )
            new_coord = cube.coord("forecast_period").copy(
                points=calculated_coord.points, bounds=calculated_coord.bounds
            )
            cube.replace_coord(new_coord)

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
        # Store the information for the record_run attribute on the cubes.
        if self.record_run_attr and self.model_id_attr:
            store_record_run_attr(cubes, self.record_run_attr, self.model_id_attr)

        cube = MergeCubes()(cubes)

        # Create the expected cell method. The aggregator adds a cell method
        # but cannot include an interval, so we create it here manually,
        # ensuring to preserve any existing cell methods.
        cell_methods = list(cube.cell_methods)
        try:
            (input_data_period,) = np.unique(np.diff(cube.coord("time").bounds)) / 3600
        except ValueError as err:
            raise ValueError(
                "Input diagnostics do not have consistent periods."
            ) from err
        cell_methods.append(
            iris.coords.CellMethod(
                "mode", coords="time", intervals=f"{int(input_data_period)} hour"
            )
        )

        self._unify_day_and_night(cube)

        # Handle case in which a single time is provided.
        if len(cube.coord("time").points) == 1:
            result = cube
        else:
            result = cube.collapsed("time", self.aggregator_instance)
        self._set_blended_times(result)

        result.cell_methods = None
        for cell_method in cell_methods:
            result.add_cell_method(cell_method)

        if self.model_id_attr:
            # Update contributing models
            contributing_models = set()
            for source_cube in cubes:
                for model in source_cube.attributes[self.model_id_attr].split(" "):
                    contributing_models.update([model])
            result.attributes[self.model_id_attr] = " ".join(
                sorted(list(contributing_models))
            )

        if self.record_run_attr and self.model_id_attr:
            apply_record_run_attr(
                result, cube, self.record_run_attr, discard_weights=True
            )
            result.remove_coord(RECORD_COORD)

        # Handle any unset points where it was hard to determine a suitable mode
        if (result.data == UNSET_CODE_INDICATOR).any():
            self._group_codes(result, cube)

        return result
