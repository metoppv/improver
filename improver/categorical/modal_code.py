# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing a plugin to calculate the modal category in a period."""

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
from .utilities import day_night_map


class BaseModalCategory(BasePlugin):

    def _unify_day_and_night(self, cube: Cube):
        """Remove distinction between day and night codes so they can each
        contribute when calculating the modal code. The cube of categorical data
        is modified in place with all night codes made into their
        daytime equivalents.

        Args:
            A cube of categorical data
        """
        for day, night in self.day_night_map.items():
            cube.data[cube.data == night] = day

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


class ModalCategory(BasePlugin):
    """Plugin that returns the modal category over the period spanned by the
    input data. In cases of a tie in the mode values, scipy returns the smaller
    value. The opposite is desirable in this case as the significance /
    importance of the weather code categories generally increases with the value. To
    achieve this the categories are subtracted from an arbitrarily larger
    number prior to calculating the mode, and this operation is reversed before the
    final output is returned.

    If there are many different categories for a single point over the time
    spanned by the input cubes it may be that the returned mode is not robust.
    Given the preference to return more significant categories explained above,
    a 12 hour period with 12 different categories, one of which is severe, will
    return that severe category to describe the whole period. This is likely not a
    good representation. In these cases grouping is used to try and select
    a suitable category (e.g. a rain shower if the codes include a mix of
    rain showers and dynamic rain) by providing a more robust mode. The lowest
    number (least significant) member of the group is returned as the code.
    Use of the least significant member reflects the lower certainty in the
    forecasts.

    Where there are different categories available for night and day, the
    modal code returned is always a day code, regardless of the times
    covered by the input files.
    """

    def __init__(
        self,
        decision_tree: Dict,
        day_weighting: Optional[int] = 2,
        day_start: Optional[int] = 6,
        day_end: Optional[int] = 17,
        wet_bias: Optional[int] = 1,
        model_id_attr: Optional[str] = None,
        record_run_attr: Optional[str] = None,
    ):
        """
        Set up plugin and create an aggregator instance for reuse

        Args:
            decision_tree:
                The decision tree used to generate the categories and which contains the
                mapping of day and night categories and of category groupings.
            model_id_attr:
                Name of attribute recording source models that should be
                inherited by the output cube. The source models are expected as
                a space-separated string.
            record_run_attr:
                Name of attribute used to record models and cycles used in
                constructing the categories.
        """
        self.aggregator_instance = Aggregator("mode", self.mode_aggregator)
        self.decision_tree = decision_tree
        self.day_weighting = day_weighting
        self.day_start = day_start
        self.day_end = day_end
        self.wet_bias = wet_bias
        self.model_id_attr = model_id_attr
        self.record_run_attr = record_run_attr
        self.day_night_map = day_night_map(self.decision_tree)

        codes = [
            node["leaf"]
            for node in self.decision_tree.values()
            if "leaf" in node.keys()
        ]
        self.code_max = max(codes) + 1
        self.unset_code_indicator = min(codes) - 100
        self.code_groups = self._code_groups()

    def _unify_day_and_night(self, cube: Cube):
        """Remove distinction between day and night codes so they can each
        contribute when calculating the modal code. The cube of categorical data
        is modified in place with all night codes made into their
        daytime equivalents.

        Args:
            A cube of categorical data
        """
        for day, night in self.day_night_map.items():
            cube.data[cube.data == night] = day

    def _code_groups(self) -> Dict:
        """Determines code groupings from the decision tree"""
        groups = {}
        for key, node in self.decision_tree.items():
            if "group" not in node.keys():
                continue
            groups[node["group"]] = groups.get(node["group"], []) + [node["leaf"]]
        return groups

    def _group_codes(self, modal: Cube, cube: Cube):
        """In instances where the mode returned is not significant, i.e. the
        category chosen occurs infrequently in the period, the codes can be
        grouped to yield a more definitive period code. Given the uncertainty,
        the least significant category (lowest number in a group that is
        found in the data) is used to replace the other data values that belong
        to that group prior to recalculating the modal code.

        The modal cube is modified in place.

        Args:
            modal:
                The modal categorical cube which contains UNSET_CODE_INDICATOR
                values that need to be replaced with a more definitive period
                code.
            cube:
                The original input data. Data relating to unset points will be
                grouped and the mode recalculated."""

        undecided_points = np.argwhere(modal.data == self.unset_code_indicator)

        for point in undecided_points:
            data = cube.data[(..., *point)].copy()

            for _, codes in self.code_groups.items():
                default_code = sorted([code for code in data if code in codes])
                if default_code:
                    data[np.isin(data, codes)] = default_code[0]
            mode_result, counts = stats.mode(self.code_max - data)
            modal.data[tuple(point)] = self.code_max - mode_result

    def mode_aggregator(self, data: ndarray, axis: int) -> ndarray:
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
        mode_result, counts = stats.mode(self.code_max - data, axis=0)
        mode_result[counts < minimum_significant_count] = (
            self.code_max - self.unset_code_indicator
        )
        return self.code_max - np.squeeze(mode_result)

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
        """Calculate the modal categorical code, with handling for edge cases.

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
        # Store the information for the record_run attribute on the cubes.
        if self.record_run_attr and self.model_id_attr:
            store_record_run_as_coord(cubes, self.record_run_attr, self.model_id_attr)

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

        # if self.day_weighting or self.wet_bias:

        #     day_start_pdt = iris.time.PartialDateTime(hour=self.day_start)
        #     day_end_pdt = iris.time.PartialDateTime(hour=self.day_end)
        #     constr = iris.Constraint(time= lambda cell: day_start_pdt <= cell.point <= day_end_pdt)
        #     day_cube = cube.extract(constr)

        #     day_cubes = iris.cube.CubeList()
        #     for cube_slice in cube.slices_over("time"):
        #         day_cubes.append(cube_slice)
        #     for increment in range(1, self.day_weighting):
        #         for day_slice in day_cube.slices_over("time"):
        #             for coord in ["time", "forecast_period"]:
        #                 day_slice.coord(coord).points = day_slice.coord(coord).points + increment
        #                 bounds = day_slice.coord(coord).bounds.copy()
        #                 bounds[0] = bounds[0] + increment
        #                 day_slice.coord(coord).bounds = bounds
        #             day_cubes.append(day_slice)
        #     cube = day_cubes.merge_cube()

        # import pdb
        # pdb.set_trace()

        # dry_values = [value for value in symbols if value in broad_categories["dry"]]
        # wet_values = [value for value in symbols if value in broad_categories["wet"]]

        # if len(dry_values) > wet_bias * len(wet_values):
        #     values = dry_values
        # else:
        #     count = 0
        #     for key, ref_values in wet_categories.items():
        #         matching_values = [value for value in wet_values if value in ref_values]
        #         n_count = len(matching_values)
        #         if n_count > count:
        #             count = n_count
        #             values = matching_values


        # Handle case in which a single time is provided.
        if len(cube.coord("time").points) == 1:
            result = cube
        else:
            result = cube.collapsed("time", self.aggregator_instance)

        # Handle any unset points where it was hard to determine a suitable mode
        if (result.data == self.unset_code_indicator).any():
            self._group_codes(result, cube)

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
            record_run_coord_to_attr(
                result, cube, self.record_run_attr, discard_weights=True
            )
            result.remove_coord(RECORD_COORD)

        return result


class ModalFromGroupings(BaseModalCategory):
    """Plugin that returns the modal category over the period spanned by the
    input data. In cases of a tie in the mode values, scipy returns the smaller
    value. The opposite is desirable in this case as the significance /
    importance of the weather code categories generally increases with the value. To
    achieve this the categories are subtracted from an arbitrarily larger
    number prior to calculating the mode, and this operation is reversed before the
    final output is returned.

    If there are many different categories for a single point over the time
    spanned by the input cubes it may be that the returned mode is not robust.
    Given the preference to return more significant categories explained above,
    a 12 hour period with 12 different categories, one of which is severe, will
    return that severe category to describe the whole period. This is likely not a
    good representation. In these cases grouping is used to try and select
    a suitable category (e.g. a rain shower if the codes include a mix of
    rain showers and dynamic rain) by providing a more robust mode. The lowest
    number (least significant) member of the group is returned as the code.
    Use of the least significant member reflects the lower certainty in the
    forecasts.

    Where there are different categories available for night and day, the
    modal code returned is always a day code, regardless of the times
    covered by the input files.
    """

    def __init__(
        self,
        decision_tree: Dict,
        day_weighting: Optional[int] = 2,
        day_start: Optional[int] = 6,
        day_end: Optional[int] = 17,
        wet_bias: Optional[int] = 1,
        ignore_intensity: Optional[bool] = True,
        model_id_attr: Optional[str] = None,
        record_run_attr: Optional[str] = None,
    ):
        """
        Set up plugin and create an aggregator instance for reuse

        Args:
            decision_tree:
                The decision tree used to generate the categories and which contains the
                mapping of day and night categories and of category groupings.
            model_id_attr:
                Name of attribute recording source models that should be
                inherited by the output cube. The source models are expected as
                a space-separated string.
            record_run_attr:
                Name of attribute used to record models and cycles used in
                constructing the categories.
        """
        self.decision_tree = decision_tree
        self.day_weighting = day_weighting
        self.day_start = day_start
        self.day_end = day_end
        self.wet_bias = wet_bias
        self.ignore_intensity = ignore_intensity
        self.model_id_attr = model_id_attr
        self.record_run_attr = record_run_attr
        self.day_night_map = day_night_map(self.decision_tree)

        self.broad_categories = {
            "wet": np.arange(9, 31),
            "dry": np.arange(0, 9),
        }
        # Priority ordered categories (keys) in case of ties
        self.wet_categories = {
            "extreme_convection": [30, 29, 28, 21, 20, 19],
            "frozen": [27, 26, 25, 24, 23, 22, 18, 17, 16],
            "liquid": [15, 14, 13, 12, 11, 10, 9],
        }
        self.intensity_categories = {
            "rain_shower": [10, 14],
            "rain": [12, 15],
            "snow_shower": [23, 26],
            "snow": [24, 27],
            "thunder": [29, 30],

        }

    def _consolidate_intensity_categories(self, cube):
        if self.ignore_intensity:
            # Ignore intensities, so that weather codes representing different
            # intensities can be grouped.
            for values in self.intensity_categories.values():
                primary_value = values[0]
                for secondary_value in values[1:]:
                    cube.data[cube.data==secondary_value] = primary_value
        return cube

    def _select_day_period(self, cube):
        day_start_pdt = iris.time.PartialDateTime(hour=self.day_start)
        day_end_pdt = iris.time.PartialDateTime(hour=self.day_end)
        constr = iris.Constraint(time= lambda cell: day_start_pdt <= cell.point <= day_end_pdt)
        day_cube = cube.extract(constr)

        day_cubes = iris.cube.CubeList()
        for cube_slice in cube.slices_over("time"):
            day_cubes.append(cube_slice)
        for increment in range(1, self.day_weighting):
            for day_slice in day_cube.slices_over("time"):
                for coord in ["time", "forecast_period"]:
                    day_slice.coord(coord).points = day_slice.coord(coord).points + increment
                    bounds = day_slice.coord(coord).bounds.copy()
                    bounds[0] = bounds[0] + increment
                    day_slice.coord(coord).bounds = bounds
                day_cubes.append(day_slice)
        cube = day_cubes.merge_cube()
        return cube

    def _find_dry_indices(self, cube, time_axis):
        # Find indices corresponding to dry weather codes inclusive of a wet bias.
        dry_counts = np.sum(np.isin(cube.data, self.broad_categories["dry"]), axis=time_axis)
        wet_counts = np.sum(np.isin(cube.data, self.broad_categories["wet"]), axis=time_axis)
        return dry_counts > self.wet_bias * wet_counts

    def _get_most_likely_following_grouping(self, cube, result, categories, dry_indices, time_axis, categorise_using_modal):
        # Identify the most likely weather code within each of the wet categories.
        category_counter = []
        most_likely_subcategory = {}
        for key in categories.keys():
            if categorise_using_modal:
                category_counter.append(np.isin(result.data, categories[key]))
            else:
                category_counter.append(np.sum(np.isin(cube.data, categories[key]), axis=time_axis))

            subcategory_counter = []
            for value in categories[key]:
                subcategory_counter.append(np.sum(cube.data == value, axis=time_axis))
            most_likely_subcategory[key] = np.array(categories[key])[np.argmax(subcategory_counter, axis=time_axis)]

        # Find the most likely wet category.
        most_likely_category = np.argmax(category_counter, axis=time_axis)

        # For each wet category, if that wet category is the most likely, assign
        # the most likely weather code from that specific wet category as the result.
        for index, key in enumerate(categories.keys()):
            category_index = np.logical_and(~dry_indices, most_likely_category == index)
            result.data[category_index] = most_likely_subcategory[key][category_index]
        return result

    def process(self, cubes: CubeList) -> Cube:
        """Calculate the modal categorical code, with handling for edge cases.

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
        # Store the information for the record_run attribute on the cubes.
        if self.record_run_attr and self.model_id_attr:
            store_record_run_as_coord(cubes, self.record_run_attr, self.model_id_attr)

        cube = MergeCubes()(cubes)
        result = cube.copy()

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

        original_cube = cube.copy()
        cube = self._consolidate_intensity_categories(cube)
        cube = self._select_day_period(cube)

        result = cube[0].copy()
        time_axis, = cube.coord_dims("time")

        dry_indices = self._find_dry_indices(cube, time_axis)
        # Ensure that the dry indices are filled with the most common symbols
        # without considering the wet categories.
        result.data[dry_indices] = stats.mode(cube.data, axis=time_axis).mode[0, dry_indices]

        result = self._get_most_likely_following_grouping(cube, result, self.wet_categories, dry_indices, time_axis, categorise_using_modal=False)

        if self.ignore_intensity:
            result = self._get_most_likely_following_grouping(original_cube, result, self.intensity_categories, dry_indices, time_axis, categorise_using_modal=True)

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
            record_run_coord_to_attr(
                result, cube, self.record_run_attr, discard_weights=True
            )
            result.remove_coord(RECORD_COORD)

        return result
