# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing thresholding classes."""

import numbers
from collections.abc import Iterable
from typing import Dict, List, Optional, Tuple, Union

import iris
import numpy as np
from cf_units import Unit
from iris.cube import Cube

from improver import PostProcessingPlugin
from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    RebadgePercentilesAsRealizations,
)
from improver.metadata.constants import FLOAT_DTYPE
from improver.metadata.probabilistic import (
    format_cell_methods_for_probability,
    probability_is_above_or_below,
)
from improver.utilities.cube_manipulation import enforce_coordinate_ordering
from improver.utilities.probability_manipulation import comparison_operator_dict
from improver.utilities.rescale import rescale
from improver.utilities.spatial import (
    create_vicinity_coord,
    distance_to_number_of_grid_cells,
    maximum_within_vicinity,
    rename_vicinity_cube,
)


class Threshold(PostProcessingPlugin):
    """Apply a threshold truth criterion to a cube.

    Calculate the threshold truth values based on a linear membership function
    around the threshold values provided. A cube will be returned with a new
    threshold dimension coordinate.

    Can operate on multiple time sequences within a cube.
    """

    def __init__(
        self,
        threshold_values: Optional[Union[float, List[float]]] = None,
        threshold_config: Optional[Dict[str, Union[List[float], str]]] = None,
        fuzzy_factor: Optional[float] = None,
        threshold_units: Optional[str] = None,
        comparison_operator: str = ">",
        collapse_coord: str = None,
        vicinity: Optional[Union[float, List[float]]] = None,
        fill_masked: Optional[float] = None,
    ) -> None:
        """
        Set up for processing an in-or-out of threshold field, including the
        generation of fuzzy_bounds which are required to threshold an input
        cube (through self.process(cube)).  If fuzzy_factor is not None, fuzzy
        bounds are calculated using the threshold value in the units in which
        it is provided.

        The usage of fuzzy_factor is exemplified as follows:

        For a 6 mm/hr threshold with a 0.75 fuzzy factor, a range of 25%
        around this threshold (between (6*0.75=) 4.5 and (6*(2-0.75)=) 7.5)
        would be generated. The probabilities of exceeding values within this
        range are scaled linearly, so that 4.5 mm/hr yields a thresholded value
        of 0 and 7.5 mm/hr yields a thresholded value of 1. Therefore, in this
        case, the thresholded exceedance probabilities between 4.5 mm/hr and
        7.5 mm/hr would follow the pattern:

        ::

            Data value  | Probability
            ------------|-------------
                4.5     |   0
                5.0     |   0.167
                5.5     |   0.333
                6.0     |   0.5
                6.5     |   0.667
                7.0     |   0.833
                7.5     |   1.0

        Args:
            threshold_values:
                Threshold value or values (e.g. 270K, 300K) to use when calculating
                the probability of the input relative to the threshold value(s).
                The units of these values, e.g. K in the example can be defined
                using the threshold_units argument or are otherwise assumed to
                match the units of the diagnostic being thresholded.
                threshold_values and and threshold_config are mutually exclusive
                arguments, defining both will lead to an exception.
            threshold_config (dict):
                Threshold configuration containing threshold values and
                (optionally) fuzzy bounds. Best used in combination with
                'threshold_units'. It should contain a dictionary of strings that
                can be interpreted as floats with the structure:
                "THRESHOLD_VALUE": [LOWER_BOUND, UPPER_BOUND]
                e.g: {"280.0": [278.0, 282.0], "290.0": [288.0, 292.0]},
                or with structure "THRESHOLD_VALUE": "None" (no fuzzy bounds).
                Repeated thresholds with different bounds are ignored; only the
                last duplicate will be used.
                threshold_values and and threshold_config are mutually exclusive
                arguments, defining both will lead to an exception.
            fuzzy_factor:
                Optional: specifies lower bound for fuzzy membership value when
                multiplied by each threshold. Upper bound is equivalent linear
                distance above threshold.
            threshold_units:
                Units of the threshold values. If not provided the units are
                assumed to be the same as those of the input cube.
            comparison_operator:
                Indicates the comparison_operator to use with the threshold.
                e.g. 'ge' or '>=' to evaluate 'data >= threshold' or '<' to
                evaluate 'data < threshold'. When using fuzzy thresholds, there
                is no difference between < and <= or > and >=.
                Valid choices: > >= < <= gt ge lt le.
            collapse_coord:
                A coordinate over which an average is calculated, collapsing
                this coordinate. The only supported options are "realization" or
                "percentile". If "percentile" is requested, the percentile
                coordinate will be rebadged as a realization coordinate prior to
                collapse. The percentile coordinate needs to be evenly spaced
                around the 50th percentile to allow successful conversion from
                percentiles to realizations and subsequent collapsing over the
                realization coordinate.
            fill_masked:
                If provided all masked points in cube will be replaced with the
                provided value.
            vicinity:
                A list of vicinity radii to use to calculate maximum in vicinity
                thresholded values. This must be done prior to realization
                collapse.

        Raises:
            ValueError: If threshold_config and threshold_values are both set
            ValueError: If neither threshold_config or threshold_values are set
            ValueError: If both fuzzy_factor and bounds within the threshold_config are set.
            ValueError: If the fuzzy_factor is not strictly between 0 and 1.
            ValueError: If using a fuzzy factor with a threshold of 0.0.
            ValueError: Can only collapse over a realization coordinate or a percentile
                        coordinate that has been rebadged as a realization coordinate.
        """
        if threshold_config and threshold_values:
            raise ValueError(
                "threshold_config and threshold_values are mutually exclusive "
                "arguments - please provide one or the other, not both"
            )
        if threshold_config is None and threshold_values is None:
            raise ValueError(
                "One of threshold_config or threshold_values must be provided."
            )
        thresholds, fuzzy_bounds = self._set_thresholds(
            threshold_values, threshold_config
        )
        self.thresholds = [thresholds] if np.isscalar(thresholds) else thresholds
        self.threshold_units = (
            None if threshold_units is None else Unit(threshold_units)
        )
        self.threshold_coord_name = None

        # read fuzzy factor or set to 1 (basic thresholding)
        fuzzy_factor_loc = 1.0
        if fuzzy_factor is not None:
            if fuzzy_bounds is not None:
                raise ValueError(
                    "Invalid combination of keywords. Cannot specify "
                    "both a fuzzy_factor and use a threshold_config that "
                    "specifies bounds."
                )
            if not 0 < fuzzy_factor < 1:
                raise ValueError(
                    "Invalid fuzzy_factor: must be >0 and <1: {}".format(fuzzy_factor)
                )
            if 0 in self.thresholds:
                raise ValueError(
                    "Invalid threshold with fuzzy factor: cannot use a "
                    "multiplicative fuzzy factor with threshold == 0, use "
                    "the threshold_config approach instead."
                )
            fuzzy_factor_loc = fuzzy_factor

        if fuzzy_bounds is None:
            self.fuzzy_bounds = self._generate_fuzzy_bounds(fuzzy_factor_loc)
        else:
            self.fuzzy_bounds = (
                [fuzzy_bounds] if isinstance(fuzzy_bounds, tuple) else fuzzy_bounds
            )
            self._check_fuzzy_bounds()

        self.original_units = None
        self.comparison_operator_dict = comparison_operator_dict()
        self.comparison_operator_string = comparison_operator
        self._decode_comparison_operator_string()

        if collapse_coord and collapse_coord not in ["percentile", "realization"]:
            raise ValueError(
                "Can only collapse over a realization coordinate or a percentile "
                "coordinate that has been rebadged as a realization coordinate."
            )
        self.collapse_coord = collapse_coord

        self.vicinity = None
        if vicinity:
            if isinstance(vicinity, Iterable):
                self.vicinity = [float(x) for x in vicinity]
            else:
                self.vicinity = [float(vicinity)]

        if fill_masked is not None:
            fill_masked = float(fill_masked)

        self.fill_masked = fill_masked

    @staticmethod
    def _set_thresholds(
        threshold_values: Optional[Union[float, List[float]]],
        threshold_config: Optional[dict],
    ) -> Tuple[List[float], Optional[List[float]]]:
        """
        Interprets a threshold_config dictionary if provided, or ensures that
        a list of thresholds has suitable precision.

        Args:
            threshold_values:
                A list of threshold values or a single threshold value.
            threshold_config:
                A dictionary defining threshold values and optionally upper
                and lower bounds for those values to apply fuzzy thresholding.

        Returns:
            A tuple containing:
                thresholds:
                    A list of threshold values as float64 type.
                fuzzy_bounds:
                    A list of tuples that define the upper and lower bounds associated
                    with each threshold value, these also as float64 type. If these
                    are not set, None is returned instead.
        """
        if threshold_config:
            thresholds = []
            fuzzy_bounds = []
            for key in threshold_config.keys():
                # Ensure thresholds are float64 to avoid rounding errors during
                # possible unit conversion.
                thresholds.append(float(key))
                # If the first threshold has no bounds, fuzzy_bounds is
                # set to None and subsequent bounds checks are skipped
                if threshold_config[key] == "None":
                    fuzzy_bounds = None
                    continue
                fuzzy_bounds.append(tuple(threshold_config[key]))
        else:
            # Ensure thresholds are float64 to avoid rounding errors during possible
            # unit conversion.
            if isinstance(threshold_values, numbers.Number):
                threshold_values = [threshold_values]
            thresholds = [float(x) for x in threshold_values]
            fuzzy_bounds = None
        return thresholds, fuzzy_bounds

    def _generate_fuzzy_bounds(
        self, fuzzy_factor_loc: float
    ) -> List[Tuple[float, float]]:
        """Construct fuzzy bounds from a fuzzy factor.  If the fuzzy factor is 1,
        the fuzzy bounds match the threshold values for basic thresholding.
        """
        fuzzy_bounds = []
        for thr in self.thresholds:
            lower_thr = thr * fuzzy_factor_loc
            upper_thr = thr * (2.0 - fuzzy_factor_loc)
            if thr < 0:
                lower_thr, upper_thr = upper_thr, lower_thr
            fuzzy_bounds.append((lower_thr, upper_thr))
        return fuzzy_bounds

    def _check_fuzzy_bounds(self) -> None:
        """If fuzzy bounds have been set from the command line, check they
        are consistent with the required thresholds
        """
        for thr, bounds in zip(self.thresholds, self.fuzzy_bounds):
            if len(bounds) != 2:
                raise ValueError(
                    "Invalid bounds for one threshold: {}."
                    " Expected 2 floats.".format(bounds)
                )
            if bounds[0] > thr or bounds[1] < thr:
                bounds_msg = (
                    "Threshold must be within bounds: " "!( {} <= {} <= {} )".format(
                        bounds[0], thr, bounds[1]
                    )
                )
                raise ValueError(bounds_msg)

    def _add_threshold_coord(self, cube: Cube, threshold: float) -> None:
        """
        Add a scalar threshold-type coordinate with correct name and units
        to a 2D slice containing thresholded data.

        The 'threshold' coordinate will be float64 to avoid rounding errors
        during possible unit conversion.

        Args:
            cube:
                Cube containing thresholded data (1s and 0s)
            threshold:
                Value at which the data has been thresholded
        """
        coord = iris.coords.DimCoord(
            np.array([threshold], dtype="float64"), units=cube.units
        )
        coord.rename(self.threshold_coord_name)
        coord.var_name = "threshold"
        cube.add_aux_coord(coord)

    def _decode_comparison_operator_string(self) -> None:
        """Sets self.comparison_operator based on
        self.comparison_operator_string. This is a dict containing the keys
        'function' and 'spp_string'.
        Raises errors if invalid options are found.

        Raises:
            ValueError: If self.comparison_operator_string does not match a
                        defined method.
        """
        try:
            self.comparison_operator = self.comparison_operator_dict[
                self.comparison_operator_string
            ]
        except KeyError:
            msg = (
                f'String "{self.comparison_operator_string}" '
                "does not match any known comparison_operator method"
            )
            raise ValueError(msg)

    def _update_metadata(self, cube: Cube) -> None:
        """Rename the cube and add attributes to the threshold coordinate
        after merging

        Args:
            cube:
                Cube containing thresholded data
        """
        threshold_coord = cube.coord(self.threshold_coord_name)
        threshold_coord.attributes.update(
            {"spp__relative_to_threshold": self.comparison_operator.spp_string}
        )
        if cube.cell_methods:
            format_cell_methods_for_probability(cube, self.threshold_coord_name)

        cube.rename(
            "probability_of_{parameter}_{relative_to}_threshold".format(
                parameter=self.threshold_coord_name,
                relative_to=probability_is_above_or_below(cube),
            )
        )
        cube.units = Unit(1)

    def _calculate_truth_value(
        self, cube: Cube, threshold: float, bounds: Tuple[float, float]
    ) -> np.ndarray:
        """
        Compares the diagnostic values to the threshold value, converting units
        and applying fuzzy bounds as required. Returns the truth value.

        Args:
            cube:
                A cube containing the diagnostic values. The cube rather than array
                is passed in to allow for unit conversion.
            threshold:
                A single threshold value against which to compare the diagnostic
                values.
            bounds:
                The fuzzy bounds used for applying fuzzy thresholding.

        Returns:
            truth_value:
                An array of truth values given by the comparison of the diagnostic
                data to the threshold value. This is returned at the default float
                precision.
        """
        if self.threshold_units is not None:
            cube.convert_units(self.threshold_units)
        # if upper and lower bounds are equal, set a deterministic 0/1
        # probability based on exceedance of the threshold
        if bounds[0] == bounds[1]:
            truth_value = self.comparison_operator.function(cube.data, threshold)
        # otherwise, scale exceedance probabilities linearly between 0/1
        # at the min/max fuzzy bounds and 0.5 at the threshold value
        else:
            truth_value = np.where(
                cube.data < threshold,
                rescale(
                    cube.data,
                    data_range=(bounds[0], threshold),
                    scale_range=(0.0, 0.5),
                    clip=True,
                ),
                rescale(
                    cube.data,
                    data_range=(threshold, bounds[1]),
                    scale_range=(0.5, 1.0),
                    clip=True,
                ),
            )
            # if requirement is for probabilities less_than or
            # less_than_or_equal_to the threshold (rather than
            # greater_than or greater_than_or_equal_to), invert
            # the exceedance probability
            if "less_than" in self.comparison_operator.spp_string:
                truth_value = 1.0 - truth_value

        return truth_value.astype(FLOAT_DTYPE)

    def _vicinity_processing(
        self,
        thresholded_cube: Cube,
        truth_value: np.ndarray,
        unmasked: np.ndarray,
        landmask: np.ndarray,
        grid_point_radii: List[int],
        index: int,
    ):
        """
        Apply max in vicinity processing to the thresholded values. The
        resulting modified threshold values are changed in place in
        thresholded_cube.

        Args:
            thresholded_cube:
                The cube into which the resulting values are added.
            truth_value:
                An array of thresholded values prior to the application of
                vicinity processing.
            unmasked:
                Array identifying unmasked data points that should be updated.
            landmask:
                A binary grid of the same size as truth_value that
                differentiates between land and sea points to allow the
                different surface types to be processed independently.
            grid_point_radii:
                The vicinity radius to apply expressed as a number of grid
                cells.
            index:
                Index corresponding to the threshold coordinate to identify
                which array we are summing the contribution into.
        """
        for ivic, vicinity in enumerate(grid_point_radii):
            if truth_value.ndim > 2:
                maxes = np.zeros(truth_value.shape, dtype=FLOAT_DTYPE)
                for yxindex in np.ndindex(*truth_value.shape[:-2]):
                    slice_max = maximum_within_vicinity(
                        truth_value[yxindex + (slice(None), slice(None))],
                        vicinity,
                        landmask,
                    )
                    maxes[yxindex] = slice_max
            else:
                maxes = maximum_within_vicinity(truth_value, vicinity, landmask)
            thresholded_cube.data[ivic][index][unmasked] += maxes[unmasked]

    def _create_threshold_cube(self, cube: Cube) -> Cube:
        """
        Create a cube with suitable metadata and zeroed data array for
        storing the thresholded diagnostic data.

        Args:
            cube:
                A template cube from which to take the data shape.

        Returns:
            thresholded_cube:
                A cube of a suitable form for storing and describing
                the thresholded data.
        """
        template = cube.copy(data=np.zeros(cube.shape, dtype=FLOAT_DTYPE))

        if self.threshold_units is not None:
            template.units = self.threshold_units

        thresholded_cube = iris.cube.CubeList()
        for threshold in self.thresholds:
            thresholded = template.copy()
            self._add_threshold_coord(thresholded, threshold)
            thresholded.units = 1
            thresholded_cube.append(thresholded)
        thresholded_cube = thresholded_cube.merge_cube()

        # Promote the threshold coordinate to be dimensional if it is not already.
        if not thresholded_cube.coord_dims(self.threshold_coord_name):
            thresholded_cube = iris.util.new_axis(
                thresholded_cube, self.threshold_coord_name
            )

        if self.vicinity is not None:
            vicinity_coord = create_vicinity_coord(self.vicinity)
            vicinity_expanded = iris.cube.CubeList()
            for vicinity_coord_slice in vicinity_coord:
                thresholded_copy = thresholded_cube.copy()
                thresholded_copy.add_aux_coord(vicinity_coord_slice)
                vicinity_expanded.append(thresholded_copy)
            del thresholded_copy
            thresholded_cube = vicinity_expanded.merge_cube()
            if not thresholded_cube.coords(vicinity_coord.name(), dim_coords=True):
                thresholded_cube = iris.util.new_axis(
                    thresholded_cube, vicinity_coord.name()
                )

        # Ensure the threshold cube has suitable metadata for a probabilistic output.
        self._update_metadata(thresholded_cube)
        return thresholded_cube

    def process(self, input_cube: Cube, landmask: Cube = None) -> Cube:
        """Convert each point to a truth value based on provided threshold
        values. The truth value may or may not be fuzzy depending upon if
        fuzzy_bounds are supplied.  If the plugin has a "threshold_units"
        member, this is used to convert both thresholds and fuzzy bounds into
        the units of the input cube.

        Args:
            input_cube:
                Cube to threshold. The code is dimension-agnostic.
            landmask:
                Cube containing a landmask. Used with vicinity processing
                only.

        Returns:
            Cube after a threshold has been applied, possibly using fuzzy
            bounds, and / or with vicinity processing applied to return a
            maximum in vicinity value. The data within this cube will
            contain values between 0 and 1 to indicate whether a given
            threshold has been exceeded or not.

                The cube meta-data will contain:
                * Input_cube name prepended with
                probability_of_X_above(or below)_threshold (where X is
                the diagnostic under consideration)
                * The cube name will be suffixed with _in_vicinity if
                vicinity processing has been applied.
                * Threshold dimension coordinate with same units as input_cube
                * Threshold attribute ("greater_than",
                "greater_than_or_equal_to", "less_than", or
                less_than_or_equal_to" depending on the operator)
                * Cube units set to (1).

        Raises:
            ValueError: Cannot apply land-mask cube without in-vicinity processing.
            ValueError: if a np.nan value is detected within the input cube.
        """
        if self.vicinity is None and landmask is not None:
            raise ValueError(
                "Cannot apply land-mask cube without in-vicinity processing"
            )

        if self.fill_masked is not None:
            input_cube.data = np.ma.filled(input_cube.data, self.fill_masked)

        if self.collapse_coord == "percentile":
            input_cube = RebadgePercentilesAsRealizations()(input_cube)
            self.collapse_coord = "realization"

        self.original_units = input_cube.units
        self.threshold_coord_name = input_cube.name()
        # Retain only the landmask array as bools.
        if landmask is not None:
            landmask = landmask.data.astype(bool)
        if self.vicinity is not None:
            grid_point_radii = [
                distance_to_number_of_grid_cells(input_cube, radius)
                for radius in self.vicinity
            ]

        # Slice over realizations if required and create an empty threshold
        # cube to store the resulting thresholded data.
        if self.collapse_coord is not None:
            input_slices = input_cube.slices_over(self.collapse_coord)
            thresholded_cube = self._create_threshold_cube(
                next(input_cube.slices_over(self.collapse_coord))
            )
        else:
            input_slices = [input_cube]
            thresholded_cube = self._create_threshold_cube(input_cube)

        # Create a zeroed array for storing contributions (i.e. number of
        # unmasked realization values contributing to calculation).
        contribution_total = np.zeros(
            next(thresholded_cube.slices_over(self.threshold_coord_name)).shape,
            dtype=int,
        )

        for cube in input_slices:
            # Tests performed on each slice rather than whole cube to avoid
            # realising all of the data.
            if np.isnan(cube.data).any():
                raise ValueError("Error: NaN detected in input cube data")

            if np.ma.is_masked(cube.data):
                mask = cube.data.mask
                unmasked = ~mask
            else:
                unmasked = np.ones(cube.shape, dtype=bool)

            # All unmasked points contribute 1 to the numerator for calculating
            # a realization collapsed truth value. Note that if input_slices
            # above includes the realization coordinate (i.e. we are not collapsing
            # that coordinate) then contribution_total will include this extra
            # dimension and our denominator will be 1 at all unmasked points.
            contribution_total += unmasked

            for index, (threshold, bounds) in enumerate(
                zip(self.thresholds, self.fuzzy_bounds)
            ):
                truth_value = self._calculate_truth_value(cube, threshold, bounds)
                if self.vicinity is not None:
                    self._vicinity_processing(
                        thresholded_cube,
                        truth_value,
                        unmasked,
                        landmask,
                        grid_point_radii,
                        index,
                    )
                else:
                    thresholded_cube.data[index][unmasked] += truth_value[unmasked]

        # Any x-y position for which there are no valid contributions must be
        # a masked point in every realization, so we can use this array to
        # modify only unmasked points and reapply a mask to the final result.
        valid = contribution_total.astype(bool)

        # Slice over the array to avoid ballooning the memory required for the
        # denominators through broadcasting.
        enforce_coordinate_ordering(thresholded_cube, self.threshold_coord_name)
        for i, dslice in enumerate(thresholded_cube.data):
            result = np.divide(dslice[valid], contribution_total[valid])
            thresholded_cube.data[i, valid] = result

        if not valid.all():
            thresholded_cube.data = np.ma.masked_array(
                thresholded_cube.data,
                mask=np.broadcast_to(~valid, thresholded_cube.shape),
            )

        # Revert threshold coordinate units to those of the input cube.
        thresholded_cube.coord(self.threshold_coord_name).convert_units(
            self.original_units
        )
        # Squeeze any single value dimension coordinates to make them scalar.
        thresholded_cube = iris.util.squeeze(thresholded_cube)

        if self.collapse_coord is not None and thresholded_cube.coords(
            self.collapse_coord
        ):
            thresholded_cube.remove_coord(self.collapse_coord)

        # Re-cast to 32bit now that any unit conversion has already taken place.
        thresholded_cube.coord(var_name="threshold").points = thresholded_cube.coord(
            var_name="threshold"
        ).points.astype(FLOAT_DTYPE)

        if self.vicinity is not None:
            rename_vicinity_cube(thresholded_cube)

        enforce_coordinate_ordering(
            thresholded_cube,
            [
                "realization",
                "percentile",
                self.threshold_coord_name,
                "radius_of_vicinity",
            ],
        )

        return thresholded_cube


class LatitudeDependentThreshold(Threshold):
    """Apply a latitude-dependent threshold truth criterion to a cube.

    Calculates the threshold truth values based on the threshold function provided.
    A cube will be returned with a new threshold dimension auxillary coordinate on
    the latitude axis.

    Can operate on multiple time sequences within a cube.
    """

    def __init__(
        self,
        threshold_function: callable,
        threshold_units: Optional[str] = None,
        comparison_operator: str = ">",
    ) -> None:
        """
        Sets up latitude-dependent threshold class

        Args:
            threshold_function:
                A function which takes a latitude value (in degrees) and returns
                the desired threshold.
            threshold_units:
                Units of the threshold values. If not provided the units are
                assumed to be the same as those of the input cube.
            comparison_operator:
                Indicates the comparison_operator to use with the threshold.
                e.g. 'ge' or '>=' to evaluate 'data >= threshold' or '<' to
                evaluate 'data < threshold'. When using fuzzy thresholds, there
                is no difference between < and <= or > and >=.
                Valid choices: > >= < <= gt ge lt le.
        """
        super().__init__(
            threshold_values=[1],
            threshold_units=threshold_units,
            comparison_operator=comparison_operator,
        )
        if not callable(threshold_function):
            raise TypeError("Threshold must be callable")
        self.threshold_function = threshold_function

    def _add_latitude_threshold_coord(self, cube: Cube, threshold: np.ndarray) -> None:
        """
        Add a 1D threshold-type coordinate with correct name and units
        to a 2D slice containing thresholded data.
        Assumes latitude coordinate is always the penultimate one (which standardise
        will have enforced)

        Args:
            cube:
                Cube containing thresholded data (1s and 0s)
            threshold:
                Values at which the data has been thresholded (matches cube's y-axis)
        """
        coord = iris.coords.AuxCoord(threshold.astype(FLOAT_DTYPE), units=cube.units)
        coord.rename(self.threshold_coord_name)
        coord.var_name = "threshold"
        cube.add_aux_coord(coord, data_dims=len(cube.shape) - 2)

    def process(self, input_cube: Cube) -> Cube:
        """Convert each point to a truth value based on provided threshold,
        fuzzy bound, and vicinity values. If the plugin has a "threshold_units"
        member, this is used to convert a copy of the input_cube into
        the units specified.

        Args:
            input_cube:
                Cube to threshold. Must have a latitude coordinate.

        Returns:
            Cube after a threshold has been applied. The data within this
            cube will contain values between 0 and 1 to indicate whether
            a given threshold has been exceeded or not.

                The cube meta-data will contain:
                * Input_cube name prepended with
                probability_of_X_above(or below)_threshold (where X is
                the diagnostic under consideration)
                * Threshold dimension coordinate with same units as input_cube
                * Threshold attribute ("greater_than",
                "greater_than_or_equal_to", "less_than", or
                less_than_or_equal_to" depending on the operator)
                * Cube units set to (1).

        Raises:
            ValueError: if a np.nan value is detected within the input cube.
        """
        if np.isnan(input_cube.data).any():
            raise ValueError("Error: NaN detected in input cube data")

        self.threshold_coord_name = input_cube.name()

        cube = input_cube.copy()
        if self.threshold_units is not None:
            cube.convert_units(self.threshold_units)

        cube.coord("latitude").convert_units("degrees")
        threshold_variant = cube.coord("latitude").points
        threshold_over_latitude = np.array(self.threshold_function(threshold_variant))

        # Add a scalar axis for the longitude axis so that numpy's array-
        # broadcasting knows what we want to do
        truth_value = self.comparison_operator.function(
            cube.data, np.expand_dims(threshold_over_latitude, 1)
        )

        truth_value = truth_value.astype(FLOAT_DTYPE)

        if np.ma.is_masked(cube.data):
            # update unmasked points only
            cube.data[~input_cube.data.mask] = truth_value[~input_cube.data.mask]
        else:
            cube.data = truth_value

        self._add_latitude_threshold_coord(cube, threshold_over_latitude)
        cube.coord(var_name="threshold").convert_units(input_cube.units)

        self._update_metadata(cube)
        enforce_coordinate_ordering(cube, ["realization", "percentile"])

        return cube
