# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
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
from improver.constants import HOURS_IN_DAY
from improver.utilities.cube_manipulation import MergeCubes

from ..metadata.forecast_times import forecast_period_coord
from .utilities import day_night_map, dry_map


class BaseModalCategory(BasePlugin):
    """Base plugin for modal weather symbol plugins."""

    def __init__(
        self,
        decision_tree: Dict,
    ):
        """
        Set up base plugin.

        Args:
            decision_tree:
                The decision tree used to generate the categories and which contains the
                mapping of day and night categories and of category groupings.
        """
        self.decision_tree = decision_tree
        self.day_night_map = day_night_map(self.decision_tree)

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

    def _prepare_input_cubes(
        self,
        cubes: CubeList,
        record_run_attr: Optional[str] = None,
        model_id_attr: Optional[str] = None,
    ) -> Cube:
        """Prepare the input cubes by adding supplementary coordinates as required
        and merging the input cubes.

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


class ModalCategory(BaseModalCategory):
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
        super().__init__(decision_tree)
        self.aggregator_instance = Aggregator("mode", self.mode_aggregator)
        self.model_id_attr = model_id_attr
        self.record_run_attr = record_run_attr

        codes = [
            node["leaf"]
            for node in self.decision_tree.values()
            if "leaf" in node.keys()
        ]
        self.code_max = max(codes) + 1
        self.unset_code_indicator = min(codes) - 100
        self.code_groups = self._code_groups()

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
        cube = self._prepare_input_cubes(
            cubes, self.record_run_attr, self.model_id_attr
        )

        self._unify_day_and_night(cube)

        # Handle case in which a single time is provided.
        if len(cube.coord("time").points) == 1:
            result = cube
        else:
            result = cube.collapsed("time", self.aggregator_instance)
            result.data = result.data.astype(cube.data.dtype)
        self._set_blended_times(result)

        result = self._prepare_result_cube(
            cube, cubes, result, self.record_run_attr, self.model_id_attr
        )

        # Handle any unset points where it was hard to determine a suitable mode
        if (result.data == self.unset_code_indicator).any():
            self._group_codes(result, cube)

        return result


class ModalFromGroupings(BaseModalCategory):
    """Plugin that creates a modal weather code over a period using a grouping
    approach. Firstly, a wet and dry grouping is computed. Secondly, for the
    wet grouping, groupings can be provided, such as, "extreme", "frozen" and "liquid",
    so that wet weather codes can be grouped further. These groupings can be controlled
    as follows. Firstly, a day weighting functionality is provided so that daytime
    hours can be weighted more heavily. A wet bias can also be provided, so that
    wet codes are given a larger weight as they are considered more impactful.
    A second categorisation is then available for the wet codes. This is useful when
    e.g. a period is represented using a variety of frozen precipitation weather codes,
    so that a frozen precipitation weather code can be diagnosed as an appropriate
    summary. The ignore intensity option allows light and heavy weather types to be
    considered together when ascertaining the most common weather type. The final
    daily symbol will be the most common of the light and heavy input codes of
    the chosen type.

    The ordering of the codes within the category dictionaries guides which
    category is selected in the event of the tie with preference given to the lowest
    index. Incrementing the codes within the category dictionaries from most significant
    code to least significant code helps to ensure that the most significant code is
    returned in the event of a tie, if desired.

    Where there are different categories available for night and day, the
    modal code returned is always a day code, regardless of the times
    covered by the input files.

    If a location is to return a dry code after consideration of the various
    weightings, the wet codes for that location are converted into the best
    matching dry cloud code and these are included in determining the resulting
    dry code. The wet bias has no impact on the weight of these converted wet
    codes, but the day weighting still applies.
    """

    # Day length set to aid testing.
    DAY_LENGTH = HOURS_IN_DAY

    def __init__(
        self,
        decision_tree: Dict,
        broad_categories: Dict[str, int],
        wet_categories: Dict[str, int],
        intensity_categories: Optional[Dict[str, int]] = None,
        day_weighting: int = 1,
        day_start: int = 6,
        day_end: int = 18,
        wet_bias: int = 1,
        model_id_attr: Optional[str] = None,
        record_run_attr: Optional[str] = None,
    ):
        """
        Set up plugin.

        Args:
            decision_tree:
                The decision tree used to generate the categories and which contains the
                mapping of day and night categories and of category groupings.
            broad_categories:
                Dictionary defining the broad categories for grouping the weather
                symbol codes. This is expected to have the keys: "dry" and "wet".
            wet_categories:
                Dictionary defining groupings for the wet categories. No specific
                names for the keys are required. Key and values within the dictionary
                should both be ordered in terms of descending priority.
            intensity_categories:
                Dictionary defining intensity groupings. Values should be ordered in
                terms of descending priority. The most common weather code from the
                options available representing different intensities will be used as the
                representative weather code.
            day_weighting:
                Weighting to provide day time weather codes. A weighting of 1 indicates
                the default weighting. A weighting of 2 indicates that the weather codes
                during the day time period will be duplicated, so that they count twice
                as much when computing a representative weather code.
            day_start:
                Hour defining the start of the daytime period.
            day_end:
                Hour defining the end of the daytime period.
            wet_bias:
                Bias to provide wet weather codes. A bias of 1 indicates the
                default, where half of the codes need to be a wet code,
                in order to generate a wet code. A bias of 3 indicates that
                only a quarter of codes are required to be wet, in order to generate
                a wet symbol. To generate a wet symbol, the fraction of wet symbols
                therefore need to be greater than or equal to 1 / (1 + wet_bias).
            model_id_attr:
                Name of attribute recording source models that should be
                inherited by the output cube. The source models are expected as
                a space-separated string.
            record_run_attr:
                Name of attribute used to record models and cycles used in
                constructing the categories.
        """
        super().__init__(decision_tree)
        self.dry_map = dry_map(self.decision_tree)

        self.broad_categories = broad_categories
        self.wet_categories = wet_categories
        self.intensity_categories = intensity_categories
        self.day_weighting = day_weighting
        self.day_start = day_start
        self.day_end = day_end
        self.wet_bias = wet_bias
        self.model_id_attr = model_id_attr
        self.record_run_attr = record_run_attr

    def _consolidate_intensity_categories(self, cube: Cube) -> Cube:
        """Consolidate weather codes representing different intensities of
        precipitation. This can help with computing a representative weather code.

        Args:
            cube: Weather codes cube.

        Returns:
            Weather codes cube with intensity categories consolidated,
            if intensity categories are provided.
        """
        if self.intensity_categories:
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
            template_cube: Cube to provide coordinates associated with the time
                coordinate that will be added to the output cube.

        Returns:
            A cube with a time dimension coordinate and other time-related coordinates
            are associated with the time dimension coordinate.
        """
        cube = iris.util.new_axis(cube, "time")
        time_dim = cube.coord_dims("time")

        associated_coords = [
            c.name() for c in template_cube.coords(dimensions=time_dim)
        ]

        for coord in associated_coords:
            if cube.coords(coord):
                coord = cube.coord(coord).copy()
                # The blend_record coordinate needs to be set to a consistent dtype
                # to facilitate concatenation later.
                coord.points = coord.points.astype(template_cube.coord(coord).dtype)
                cube.remove_coord(coord)
                cube.add_aux_coord(coord, data_dims=time_dim)
        iris.util.promote_aux_coord_to_dim_coord(cube, "time")
        return cube

    def _emphasise_day_period(self, cube: Cube) -> Cube:
        """A day weighting can be set which biases the forecasts towards the hours of
        e.g. 6am-6pm. This is achieved by counting the number of input times available
        e.g. hourly and taking those that are 18 times from the end up to those
        that are 6 from the end and duplicating these symbols by the integer weighting.
        This approach is taken to accommodate different timezones without the need for
        any timezone awareness. Inputs are always provided from midnight to midnight, or
        ending at midnight if a partial day is provided. The middle of the set of input
        times therefore corresponds to the local middle of the day. The count back
        from the end of the period is done to accommodate partial periods (same day
        updates). The index counted backwards is clipped to 0, meaning if there are
        only 12 files being passed in (because we're around midday when we perform
        the update), the first index will be 0, rather than -6, and only symbols from
        6 periods will be multiplied up by the day_weighting.

        Metadata is not used to select the day period as the times recorded
        within the cubes are all UTC, rather than local time, so the local day period
        can not be identified. The time and forecast_period coordinates are
        incremented by the the minimum arbitrary amount (1 second) to ensure
        non-duplicate coordinates.

        Args:
            cube: Weather codes cube.

        Returns:
            Cube with more times during the daytime period, so that daytime hours
            are emphasised, depending upon the day_weighting chosen.
        """
        day_cubes = iris.cube.CubeList()
        for cube_slice in cube.slices_over("time"):
            cube_slice = self._promote_time_coords(cube_slice, cube)
            day_cubes.append(cube_slice)

        time_coord = cube.coord("time").copy()
        time_coord.convert_units("hours since 1970-01-01 00:00:00")
        interval = time_coord.bounds[0][1] - time_coord.bounds[0][0]

        n_times = len(day_cubes)
        start_file = np.clip(
            (n_times - int((self.DAY_LENGTH - self.day_start) / interval)), 0, None
        )
        end_file = np.clip(
            (n_times - int((self.DAY_LENGTH - self.day_end) / interval)), 0, None
        )

        for increment in range(1, self.day_weighting):
            for day_slice in day_cubes[start_file:end_file]:
                day_slice = day_slice.copy()
                for coord in ["time", "forecast_period"]:
                    if len(cube.coord_dims(coord)) > 0:
                        day_slice.coord(coord).points = (
                            day_slice.coord(coord).points + increment
                        )
                        bounds = day_slice.coord(coord).bounds.copy()
                        bounds[0] = bounds[0] + increment
                        day_slice.coord(coord).bounds = bounds
                day_cubes.append(day_slice)
        return day_cubes.concatenate_cube()

    def _find_wet_indices(self, cube: Cube, time_axis: int) -> np.ndarray:
        """Identify the points at which a wet weather code should be selected.
        This can include a wet bias if supplied.

        Args:
            cube: Weather codes cube.
            time_axis: The time coordinate dimension.

        Returns:
            Boolean array that is true if the weather codes are wet or False otherwise.
        """
        # Find indices corresponding to dry weather codes inclusive of a wet bias.
        dry_counts = np.sum(
            np.isin(cube.data, self.broad_categories["dry"]), axis=time_axis
        )
        wet_counts = np.sum(
            np.isin(cube.data, self.broad_categories["wet"]), axis=time_axis
        )
        return (self.wet_bias * wet_counts) >= dry_counts

    @staticmethod
    def counts_per_category(data: np.ndarray, bin_max: int) -> np.ndarray:
        """Implemented following https://stackoverflow.com/questions/46256279/bin-elements-per-row-vectorized-2d-bincount-for-numpy/46256361#46256361  # noqa: E501
        Use np.bincount to count the number of occurrences within each category, so that
        the most common occurrence can then be found.

        Args:
            data: Array where occurrences of each possible integer value between 0
                and data.max() will be counted.
            bin_max: Integer defining the number of categories expected.

        Returns:
            An array of counts for the occurrence of each category within each row.
        """
        n_cat = bin_max + 1
        a_offs = data + np.arange(data.shape[0])[:, None] * n_cat
        # Use compressed() to avoid counting masked values.
        return np.bincount(
            a_offs.ravel().compressed(), minlength=data.shape[0] * n_cat
        ).reshape(-1, n_cat)

    def _find_most_significant_dry_code(
        self, cube: Cube, result: Cube, dry_indices: np.ndarray
    ) -> Cube:
        """Find the most significant dry weather code at each point.

        Args:
            cube: Weather code cube.
            result: Cube into which to put the result.
            dry_indices: Boolean, which is true if the weather codes at that point,
                are dry.

        Returns:
            Cube where points that are dry are filled with the most common dry
            code present at that point. If there is a tie, the most significant dry
            weather code is used, assuming higher values for the weather code indicates
            more significant weather.
        """
        data = cube.data.copy()
        data = np.ma.masked_where(
            ~np.isin(cube.data.copy(), self.broad_categories["dry"]), data
        )

        # If required, reshape a 3D dataset e.g. a gridded dataset to 2D.
        # 2D datasets e.g. site datasets will be unchanged.
        reshaped_data = data.T.reshape(-1, len(cube.coord("time").points), order="F")
        bins = [i for v in self.broad_categories.values() for i in v]
        bin_max = np.amax(bins)

        counts = self.counts_per_category(reshaped_data, bin_max=bin_max)

        # Flip counts with the aim that the counts for the higher index weather codes
        # are on the left, and will therefore be selected by argmax.
        counts = np.fliplr(counts)

        # Reshape the 2D counts back to a 3D dataset, if required. If the input
        # data is 2D, this reshape will have no impact.
        reshaped_counts = counts.reshape(*data.shape[1:], -1)

        result.data[dry_indices] = (bin_max - np.argmax(reshaped_counts, axis=-1))[
            dry_indices
        ]
        return result

    def _find_intensity_indices(self, cube: Cube) -> np.ndarray:
        """Find which points / sites include any weather code predictions that fall
        within the intensity categories.

        Args:
            cube: Weather code cube.

        Returns:
            Boolean that is True if any weather code from the intensity categories
            are found at a given point, otherwise False.
        """
        values = np.isin(
            cube.data, [x for v in self.intensity_categories.values() for x in v]
        )
        return values.astype(bool)

    def _get_most_likely_following_grouping(
        self,
        cube: Cube,
        result: Cube,
        categories: Dict,
        required_indices: np.ndarray,
        time_axis: int,
        categorise_using_modal: bool,
    ):
        """Determine the most common category and subcategory using a dictionary
        defining the categorisation. The category could be a group of weather codes
        representing frozen precipitation, where the subcategory would be the individual
        weather codes, so that this method is able to identify the most likely weather
        code within the most likely weather code category. If a category or subcategory
        is tied, then the first as defined within the categories dictionary is taken.
        As the categories and subcategories within the dictionary are expected to be
        in descending priority order, this will ensure that the highest priority item
        is chosen in the event of a tie.

        Args:
            cube: Weather codes cube.
            result: Cube in which to put the result.
            categories: Dictionary defining the categories (keys) and
                subcategories (values). The most likely category and then the most
                likely value for the subcategory is put into the result cube.
            required_indices: Boolean indicating which indices within the result cube
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
                required_indices, most_likely_category == index
            )
            result.data[category_index] = most_likely_subcategory[key][category_index]
        return result

    @staticmethod
    def _set_blended_times(cube: Cube, result: Cube) -> None:
        """Updates time coordinates so that time point is at the end of the time bounds,
        blend_time and forecast_reference_time (if present) are set to the end of the
        bound period and bounds are removed, and forecast_period is updated to match.
        The result cube is modified in-place.

        Args:
            cube: Cube containing metadata on the temporal coordinates that will be
                used to add the relevant metadata to the result cube.
            result: Cube containing the computed modal weather code. This cube will be
                updated in-place.
        """
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

    def _get_dry_equivalents(
        self, cube: Cube, dry_indices: np.ndarray, time_axis: int
    ) -> Cube:
        """
        Returns a cube with only dry codes in which all wet codes have
        been replaced by their nearest dry cloud equivalent. For example a
        shower code is replaced with a partly cloudy code, a light rain code
        is replaced with a cloud code, and a heavy rain code is replaced with
        an overcast cloud code.

        Args:
            cube: Weather code cube.
            dry_indices: An array of bools which are true for locations where
                         the summary weather code will be dry.
            time_axis: The time coordinate dimension.

        Returns:
            cube: Wet codes converted to their dry equivalent for those points
                  that will receive a dry summary weather code.
        """
        dry_cube = cube.copy()
        for value, target in self.dry_map.items():
            dry_cube.data = np.where(cube.data == value, target, dry_cube.data)

        # Note that np.rollaxis returns a new view of the input data. As such
        # changes to `original` here are also changes to cube.data.
        original = np.rollaxis(cube.data, time_axis)
        dried = np.rollaxis(dry_cube.data, time_axis)
        original[..., dry_indices] = dried[..., dry_indices]

        return cube

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
            cube = self._emphasise_day_period(cube)

            result = cube[0].copy()
            (time_axis,) = cube.coord_dims("time")

            wet_indices = self._find_wet_indices(cube, time_axis)

            # For dry locations convert the wet codes to their equivalent dry
            # codes for use in determining the summary symbol.
            cube = self._get_dry_equivalents(cube, ~wet_indices, time_axis)

            original_cube = cube.copy()
            cube = self._consolidate_intensity_categories(cube)
            result = self._find_most_significant_dry_code(cube, result, ~wet_indices)

            result = self._get_most_likely_following_grouping(
                cube,
                result,
                self.wet_categories,
                wet_indices,
                time_axis,
                categorise_using_modal=False,
            )

            if self.intensity_categories:
                intensity_indices = self._find_intensity_indices(result)
                result = self._get_most_likely_following_grouping(
                    original_cube,
                    result,
                    self.intensity_categories,
                    intensity_indices,
                    time_axis,
                    categorise_using_modal=True,
                )

        self._set_blended_times(cube, result)

        result = self._prepare_result_cube(
            cube, cubes, result, self.record_run_attr, self.model_id_attr
        )

        return result
