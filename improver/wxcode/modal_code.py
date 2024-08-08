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

from typing import Dict, Optional

import iris
import numpy as np
from iris.analysis import Aggregator
from iris.cube import Cube, CubeList
from numpy import ndarray
from scipy import stats

from improver import BasePlugin
from improver.blending import RECORD_COORD
from improver.blending.utilities import (
    record_run_coord_to_attr,
    store_record_run_as_coord,
)
from improver.utilities.cube_manipulation import MergeCubes

from ..metadata.forecast_times import forecast_period_coord
from .utilities import DAYNIGHT_CODES, GROUPED_CODES

CODE_MAX = 100
UNSET_CODE_INDICATOR = -99


class BaseModalCategory(BasePlugin):
    """Base plugin for modal weather symbol plugins."""

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

    def _prepare_input_cubes(
        self,
        cubes: CubeList,
        record_run_attr: Optional[str] = None,
        model_id_attr: Optional[str] = None,
    ) -> Cube:
        """
        Args:
            cubes: Input cubes on which to store the metadata required.
            record_run_attr: Attribute to record the run information.
                Defaults to None.
            model_id_attr: Attribute to record the model_id information.
                Defaults to None.
        Returns:
            Merged cube with metadata added as required.
        """
        # Store the information for the record_run attribute on the cubes.
        if record_run_attr and model_id_attr:
            store_record_run_as_coord(cubes, record_run_attr, model_id_attr)
        return MergeCubes()(cubes)

    def _prepare_result_cube(
        self,
        cube: Cube,
        cubes: CubeList,
        result: Cube,
        record_run_attr: Optional[str] = None,
        model_id_attr: Optional[str] = None,
    ) -> Cube:
        """Update the result cube with metadata from the input cubes.
        Args:
            cube: Input cube
            cubes: Input cubelist.
            result: Result cube.
            record_run_attr: Attribute to record the run information.
                Defaults to None.
            model_id_attr: Attribute to record the model_id information.
                Defaults to None.
        Raises:
            ValueError: If the time coordinate on the input cube does not represent
                consistent periods.
        Returns:
            Cube with updated metadata.
        """
        # Create the expected cell method. Manually create a cell method ensuring to
        # preserve any existing cell methods.
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
        result.cell_methods = None
        for cell_method in cell_methods:
            result.add_cell_method(cell_method)

        if model_id_attr:
            # Update contributing models
            contributing_models = set()
            for source_cube in cubes:
                for model in source_cube.attributes[model_id_attr].split(" "):
                    contributing_models.update([model])
            result.attributes[model_id_attr] = " ".join(
                sorted(list(contributing_models))
            )

        if record_run_attr and model_id_attr:
            record_run_coord_to_attr(
                result, cube, record_run_attr, discard_weights=True
            )
            result.remove_coord(RECORD_COORD)
        return result


class ModalWeatherCode(BaseModalCategory):
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
        specified axis. If the modal value selected comprises less than 30%
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
        minimum_significant_count = 0.3 * data.shape[0]
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
        cube = self._prepare_input_cubes(
            cubes, self.record_run_attr, self.model_id_attr
        )

        self._unify_day_and_night(cube)

        # Handle case in which a single time is provided.
        if len(cube.coord("time").points) == 1:
            result = cube
        else:
            result = cube.collapsed("time", self.aggregator_instance)
        self._set_blended_times(result)

        result = self._prepare_result_cube(
            cube, cubes, result, self.record_run_attr, self.model_id_attr
        )

        # Handle any unset points where it was hard to determine a suitable mode
        if (result.data == UNSET_CODE_INDICATOR).any():
            self._group_codes(result, cube)

        return result


class ModalFromGroupings(BaseModalCategory):
    """Plugin that creates a modal weather code over a period using a grouping
    approach. Firstly, a wet and dry grouping is computed. Secondly, for the
    wet grouping, groupings can be provided, such as, "extreme", "frozen" and "liquid",
    so that wet weather codes can be grouped further. These groupings can be controlled
    as follows. Firstly, a day weighting functionality is provided so that daytime
    hours can be weighted more heavily. A wet bias can also be provided, so that
    wet symbols are given a larger weight as they are considered more impactful. The
    intensity of the codes can also be ignored. This is most useful when e.g. a period
    is best represented using a variety of frozen precipitation weather symbols.
    Grouping the codes, ignoring the intensities, helps to ensure that the most
    significant weather is highlighted e.g. snow, rather than sleet.

    The ordering of the codes within the category dictionaries provided guides which
    category is selected in the event of the tie with preference given to the lowest
    index. Incrementing the codes within the category dictionaries from most significant
    code to least significant code helps to ensure that the most significant code is
    returned in the event of a tie, if desired.

    Where there are different categories available for night and day, the
    modal code returned is always a day code, regardless of the times
    covered by the input files.
    """

    def __init__(
        self,
        broad_categories: Dict[str, int],
        wet_categories: Dict[str, int],
        intensity_categories: Dict[str, int],
        day_weighting: Optional[int] = 1,
        day_start: Optional[int] = 6,
        day_end: Optional[int] = 18,
        wet_bias: Optional[int] = 1,
        ignore_intensity: Optional[bool] = False,
        model_id_attr: Optional[str] = None,
        record_run_attr: Optional[str] = None,
    ):
        """
        Set up plugin.

        Args:
            broad_categories:
                Dictionary defining the broad categories for grouping the weather
                symbol codes. This is expected to have the keys: "dry" and "wet".
            wet_categories:
                Dictionary defining groupings for the wet categories. No specific
                names for the keys are required. Values within the dictionary should
                be ordered in terms of descending priority.
            intensity_categories:
                Dictionary defining intensity groupings. Values should be ordered in
                terms of descending priority.
            day_weighting:
                Weighting to provide day time weather codes. A weighting of 1 indicates
                the default weighting. A weighting of 2 indicates that the weather codes
                during the day time period will be duplicated, so that they count twice
                as much when computing a representative weather code.
            day_start:
                Hour defining the start of the daytime period for the time coordinate.
            day_end:
                Hour defining the end of the daytime period for the time coordinate.
            wet_bias:
                Weighting to provide wet weather codes. A weighting of 1 indicates the
                default weighting. A weighting of 2 indicates that the wet weather
                codes will be duplicated, so that they count twice as much when
                computing a representative weather code.
            ignore_intensity:
                Boolean indicating whether weather codes of different intensities
                should be grouped together when establishing the most representative
                weather code. The most common weather code from the options available
                representing different intensities will be used as the representative
                weather code.
            model_id_attr:
                Name of attribute recording source models that should be
                inherited by the output cube. The source models are expected as
                a space-separated string.
            record_run_attr:
                Name of attribute used to record models and cycles used in
                constructing the categories.
        """
        self.broad_categories = broad_categories
        self.wet_categories = wet_categories
        self.intensity_categories = intensity_categories
        self.day_weighting = day_weighting
        self.day_start = day_start
        self.day_end = day_end
        self.wet_bias = wet_bias
        self.ignore_intensity = ignore_intensity
        self.model_id_attr = model_id_attr
        self.record_run_attr = record_run_attr

    def _consolidate_intensity_categories(self, cube: Cube) -> Cube:
        """Consolidate weather codes representing different intensities of
        precipitation. This can help with computing a representative weather code.

        Args:
            cube: Weather codes cube.

        Returns:
            Weather codes cube with intensity categories consolidated, if the
            ignore_intensity option is enabled.
        """
        if self.ignore_intensity:
            # Ignore intensities, so that weather codes representing different
            # intensities can be grouped.
            for values in self.intensity_categories.values():
                primary_value = values[0]
                for secondary_value in values[1:]:
                    cube.data[cube.data == secondary_value] = primary_value
        return cube

    @staticmethod
    def _promote_time_coords(cube: Cube, template_cube: Cube) -> Cube:
        """Promote the time coordinate, so that cubes can be concatenated along the
        time coordinate. Concatenation, rather than merging, helps to ensure
        consistent output, as merging can lead to other coordinates e.g.
        forecast_reference_time and forecast_period being made the dimension coordinate.

        Args:
            cube: Cube with time coordinates.

        Returns:
            A cube with a time dimension coordinate and other time-related coordinates
            are associated with the time dimension coordinate.
        """
        cube = iris.util.new_axis(cube, "time")
        time_dim = cube.coord_dims("time")

        associated_coords = [
            c.name()
            for c in template_cube.coords(dimensions=time_dim, dim_coords=False)
        ]

        for coord in associated_coords:
            if cube.coords(coord):
                coord = cube.coord(coord).copy()
                # The blend_record coordinate needs to be set to a consistent dtype
                # to facilitate concatenation later.
                coord.points = coord.points.astype(template_cube.coord(coord).dtype)
                cube.remove_coord(coord)
                cube.add_aux_coord(coord, data_dims=time_dim)
        return cube

    def _emphasise_day_period(self, cube: Cube) -> Cube:
        """Use a day weighting, plus the hour of the day defining the day start and
        day end, so the daytime hours are weighted more heavily when computing the
        weather symbol. The time and forecast_period coordinates are incremented
        by the the minimum arbitrary amount (1 second) to ensure non-duplicate
        coordinates.

        Args:
            cube: Weather codes cube.

        Returns:
            Cube with more times during the daytime period, so that daytime hours
            are emphasised, depending upon the day_weighting chosen.
        """
        day_start_pdt = iris.time.PartialDateTime(hour=self.day_start)
        day_end_pdt = iris.time.PartialDateTime(hour=self.day_end)
        constr = iris.Constraint(
            time=lambda cell: day_start_pdt <= cell.point <= day_end_pdt
        )
        day_cube = cube.extract(constr)

        day_cubes = iris.cube.CubeList()
        for cube_slice in cube.slices_over("time"):
            cube_slice = self._promote_time_coords(cube_slice, cube)
            day_cubes.append(cube_slice)
        for increment in range(1, self.day_weighting):
            for day_slice in day_cube.slices_over("time"):
                for coord in ["time", "forecast_period"]:
                    if len(cube.coord_dims(coord)) > 0:
                        day_slice.coord(coord).points = (
                            day_slice.coord(coord).points + increment
                        )
                        bounds = day_slice.coord(coord).bounds.copy()
                        bounds[0] = bounds[0] + increment
                        day_slice.coord(coord).bounds = bounds
                day_slice = self._promote_time_coords(day_slice, cube)
                day_cubes.append(day_slice)

        cube = day_cubes.concatenate_cube()
        return cube

    def _find_dry_indices(self, cube: Cube, time_axis: int) -> np.ndarray:
        """Find the indices indicating dry weather codes. This can include a wet bias
        if supplied.

        Args:
            cube: Weather codes cube.
            time_axis: The time coordinate dimension.

        Returns:
            Boolean array that is true if the weather codes are dry or False otherwise.
        """
        # Find indices corresponding to dry weather codes inclusive of a wet bias.
        dry_counts = np.sum(
            np.isin(cube.data, self.broad_categories["dry"]), axis=time_axis
        )
        wet_counts = np.sum(
            np.isin(cube.data, self.broad_categories["wet"]), axis=time_axis
        )
        return dry_counts > self.wet_bias * wet_counts

    def _find_most_significant_dry_code(
        self, cube: Cube, result: Cube, dry_indices: np.ndarray, time_axis: int
    ) -> Cube:
        """Find the most significant dry weather code at each point.

        Args:
            cube: Weather code cube.
            result: Cube into which to put the result.
            dry_indices: Boolean, which is true if the weather codes at that point,
                are dry.
            time_axis: The time coordinate dimension.

        Returns:
            Cube where points that are dry are filled with the most common dry
            code present at that point. If there is a tie, the most significant dry
            weather code is used, assuming higher values for the weather code indicate
            more significant weather.
        """
        # Clip the weather codes to be within the range given by the dry weather codes.
        cube_min = np.min(cube.data, axis=time_axis)
        cube_max = np.max(cube.data, axis=time_axis)
        min_clip_value = np.max(
            [
                np.broadcast_to(np.min(self.broad_categories["dry"]), cube_min.shape),
                cube_min,
            ],
            axis=time_axis,
        )
        max_clip_value = np.min(
            [
                np.broadcast_to(np.max(self.broad_categories["dry"]), cube_max.shape),
                cube_max,
            ],
            axis=time_axis,
        )
        uniques, counts = np.unique(
            np.clip(cube.data, min_clip_value, max_clip_value),
            return_counts=True,
            axis=time_axis,
        )
        # Flip the unique values and the counts to be in descending order, so that
        # the argmax will use the weather code with the lowest index in the event of
        # a tie.
        uniques = np.flip(uniques, axis=time_axis)
        counts = np.flip(counts, axis=time_axis)
        result.data[dry_indices] = uniques[np.argmax(counts)][dry_indices]
        return result

    def _find_non_intensity_indices(self, cube: Cube, time_axis: int) -> np.ndarray:
        """Find which points have predictions for weather codes from any of the
        intensity categories.

        Args:
            cube: Weather code cube.
            time_axis: The time coordinate dimension.

        Returns:
            Boolean that is True is any weather code from the intensity categories
            are found at a given point, otherwise False.
        """
        return ~np.sum(
            np.isin(cube.data, self.intensity_categories.values()), axis=time_axis
        )

    def _get_most_likely_following_grouping(
        self,
        cube: Cube,
        result: Cube,
        categories: Dict,
        indices_to_ignore: np.ndarray,
        time_axis: int,
        categorise_using_modal: bool,
    ):
        """Determine the most likely category and subcategory using a dictionary
        defining the categorisation. The category could be a group of weather codes
        representing frozen precipitation, where the subcategory would be the individual
        weather codes.

        Args:
            cube: Weather codes cube.
            result: Cube in which to put the result.
            categories: Dictionary defining the categories (keys) and
                subcategories (values). The most likely category and then the most
                likely value for the subcategory is put into the result cube.
            indices_to_ignore: Boolean indicating which indices within the result cube
                to fill.
            time_axis: The time coordinate dimension.
            categorise_using_modal: Boolean defining whether the top level
                categorisation should use the input cube or the processed result time.
                The input cube will have a time dimension, whereas the result cube
                will not have a time dimension.

        Returns:
            A result cube containing the most appropriate weather code following
            categorisation.
        """
        # Identify the most likely weather code within each of the subcategory.
        category_counter = []
        most_likely_subcategory = {}
        for key in categories.keys():
            if categorise_using_modal:
                category_counter.append(np.isin(result.data, categories[key]))
            else:
                category_counter.append(
                    np.sum(np.isin(cube.data, categories[key]), axis=time_axis)
                )

            subcategory_counter = []
            for value in categories[key]:
                subcategory_counter.append(np.sum(cube.data == value, axis=time_axis))
            most_likely_subcategory[key] = np.array(categories[key])[
                np.argmax(subcategory_counter, axis=time_axis)
            ]

        # Identify which category is most likely.
        most_likely_category = np.argmax(category_counter, axis=time_axis)

        # Use the most likely subcategory from the most likely category.
        for index, key in enumerate(categories.keys()):
            category_index = np.logical_and(
                ~indices_to_ignore, most_likely_category == index
            )
            result.data[category_index] = most_likely_subcategory[key][category_index]
        return result

    @staticmethod
    def _set_blended_times(cube: Cube, result: Cube) -> None:
        """Updates time coordinates so that time point is at the end of the time bounds,
        blend_time and forecast_reference_time (if present) are set to the end of the
        bound period and bounds are removed, and forecast_period is updated to match."""
        result.coord("time").points = cube.coord("time").points[-1]
        result.coord("time").bounds = [
            cube.coord("time").bounds[0][0],
            cube.coord("time").bounds[-1][-1],
        ]

        for coord_name in ["blend_time", "forecast_reference_time"]:
            if coord_name in [c.name() for c in result.coords()] and coord_name in [
                c.name() for c in cube.coords()
            ]:
                coord = cube.coord(coord_name)
                coord = coord.copy(coord.points[-1])
                result.replace_coord(coord)

        if "forecast_period" in [c.name() for c in result.coords()]:
            calculated_coord = forecast_period_coord(
                result, force_lead_time_calculation=True
            )
            new_coord = result.coord("forecast_period").copy(
                points=calculated_coord.points, bounds=calculated_coord.bounds
            )
            result.replace_coord(new_coord)

    def process(self, cubes: CubeList) -> Cube:
        """Calculate the modal categorical code by grouping weather codes.

        Args:
            cubes:
                A list of categorical cubes at different times. A modal
                code will be calculated over the time coordinate to return
                the most common code, which is taken to be the best
                representation of the whole period.

        Returns:
            A single categorical cube with time bounds that span those of
            the input categorical cubes.
        """
        cube = self._prepare_input_cubes(
            cubes, self.record_run_attr, self.model_id_attr
        )

        self._unify_day_and_night(cube)

        if len(cube.coord("time").points) == 1:
            result = cube
        else:
            original_cube = cube.copy()
            cube = self._consolidate_intensity_categories(cube)
            cube = self._emphasise_day_period(cube)

            result = cube[0].copy()
            (time_axis,) = cube.coord_dims("time")

            dry_indices = self._find_dry_indices(cube, time_axis)
            result = self._find_most_significant_dry_code(
                cube, result, dry_indices, time_axis
            )

            result = self._get_most_likely_following_grouping(
                cube,
                result,
                self.wet_categories,
                dry_indices,
                time_axis,
                categorise_using_modal=False,
            )

            non_intensity_indices = self._find_non_intensity_indices(cube, time_axis)
            if self.ignore_intensity:
                result = self._get_most_likely_following_grouping(
                    original_cube,
                    result,
                    self.intensity_categories,
                    non_intensity_indices,
                    time_axis,
                    categorise_using_modal=True,
                )

        self._set_blended_times(cube, result)

        result = self._prepare_result_cube(
            cube, cubes, result, self.record_run_attr, self.model_id_attr
        )

        return result
