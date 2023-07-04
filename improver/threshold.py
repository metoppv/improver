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
"""Module containing thresholding classes."""

from typing import Callable, List, Optional, Tuple, Union

import iris
import numpy as np
from cf_units import Unit
from iris.cube import Cube

from improver import PostProcessingPlugin
from improver.metadata.constants import FLOAT_DTYPE
from improver.metadata.probabilistic import (
    format_cell_methods_for_probability,
    probability_is_above_or_below,
)
from improver.utilities.cube_manipulation import enforce_coordinate_ordering
from improver.utilities.probability_manipulation import comparison_operator_dict
from improver.utilities.rescale import rescale


class BasicThreshold(PostProcessingPlugin):

    """Apply a threshold truth criterion to a cube.

    Calculate the threshold truth values based on a linear membership function
    around the threshold values provided. A cube will be returned with a new
    threshold dimension coordinate.

    Can operate on multiple time sequences within a cube.
    """

    def __init__(
        self,
        thresholds: Union[float, List[float]],
        fuzzy_factor: Optional[float] = None,
        fuzzy_bounds: Optional[
            Union[Tuple[float, float], List[Tuple[float, float]]]
        ] = None,
        threshold_units: Optional[str] = None,
        comparison_operator: str = ">",
        each_threshold_func: Union[Callable, List[Callable]] = (),
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
            thresholds:
                Values at which to evaluate the input data using the specified
                comparison operator.
            fuzzy_factor:
                Optional: specifies lower bound for fuzzy membership value when
                multiplied by each threshold. Upper bound is equivalent linear
                distance above threshold.
            fuzzy_bounds:
                Optional: lower and upper bounds for fuzziness. Each entry in list
                should be a tuple of two floats representing the lower and upper
                bounds respectively. Tuple or list should match length of (or scalar)
                'thresholds' argument. Should not be set if fuzzy_factor is set.
            threshold_units:
                Units of the threshold values. If not provided the units are
                assumed to be the same as those of the input cube.
            comparison_operator:
                Indicates the comparison_operator to use with the threshold.
                e.g. 'ge' or '>=' to evaluate 'data >= threshold' or '<' to
                evaluate 'data < threshold'. When using fuzzy thresholds, there
                is no difference between < and <= or > and >=.
                Valid choices: > >= < <= gt ge lt le.
            each_threshold_func:
                Callable or sequence of callables to apply after thresholding.
                Eg vicinity processing or collapse over ensemble realizations.
            fill_masked:
                If provided all masked points in cube will be replaced with the
                provided value.

        Raises:
            ValueError: If using a fuzzy factor with a threshold of 0.0.
            ValueError: If the fuzzy_factor is not strictly between 0 and 1.
            ValueError: If both fuzzy_factor and fuzzy_bounds are set.
        """
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
                    "fuzzy_factor and fuzzy_bounds together"
                )
            if not 0 < fuzzy_factor < 1:
                raise ValueError(
                    "Invalid fuzzy_factor: must be >0 and <1: {}".format(fuzzy_factor)
                )
            if 0 in self.thresholds:
                raise ValueError(
                    "Invalid threshold with fuzzy factor: cannot use a "
                    "multiplicative fuzzy factor with threshold == 0"
                )
            fuzzy_factor_loc = fuzzy_factor

        if fuzzy_bounds is None:
            self.fuzzy_bounds = self._generate_fuzzy_bounds(fuzzy_factor_loc)
        else:
            self.fuzzy_bounds = (
                [fuzzy_bounds] if isinstance(fuzzy_bounds, tuple) else fuzzy_bounds
            )
            self._check_fuzzy_bounds()

        self.comparison_operator_dict = comparison_operator_dict()
        self.comparison_operator_string = comparison_operator
        self._decode_comparison_operator_string()

        if callable(each_threshold_func):
            each_threshold_func = (each_threshold_func,)
        self.each_threshold_func = each_threshold_func

        self.fill_masked = fill_masked

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
                    "Threshold must be within bounds: "
                    "!( {} <= {} <= {} )".format(bounds[0], thr, bounds[1])
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

    def process(self, input_cube: Cube) -> Cube:
        """Convert each point to a truth value based on provided threshold
        values. The truth value may or may not be fuzzy depending upon if
        fuzzy_bounds are supplied.  If the plugin has a "threshold_units"
        member, this is used to convert both thresholds and fuzzy bounds into
        the units of the input cube.

        Args:
            input_cube:
                Cube to threshold. The code is dimension-agnostic.

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

        if self.fill_masked is not None:
            input_cube.data = np.ma.filled(input_cube.data, self.fill_masked)

        self.threshold_coord_name = input_cube.name()

        thresholded_cubes = iris.cube.CubeList()
        for threshold, bounds in zip(self.thresholds, self.fuzzy_bounds):
            cube = input_cube.copy()
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

            truth_value = truth_value.astype(FLOAT_DTYPE)

            if np.ma.is_masked(cube.data):
                # update unmasked points only
                cube.data[~input_cube.data.mask] = truth_value[~input_cube.data.mask]
            else:
                cube.data = truth_value

            self._add_threshold_coord(cube, threshold)
            cube.coord(var_name="threshold").convert_units(input_cube.units)
            self._update_metadata(cube)

            for func in self.each_threshold_func:
                cube = func(cube)

            thresholded_cubes.append(cube)

        (cube,) = thresholded_cubes.merge()
        # Re-cast to 32bit now that any unit conversion has already taken place.
        cube.coord(var_name="threshold").points = cube.coord(
            var_name="threshold"
        ).points.astype(FLOAT_DTYPE)

        enforce_coordinate_ordering(cube, ["realization", "percentile"])
        return cube


class LatitudeDependentThreshold(BasicThreshold):

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
            thresholds=[1],
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
        """Convert each point to a truth value based on provided threshold
        function. If the plugin has a "threshold_units"
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
            cube.data, np.expand_dims(threshold_over_latitude, 1),
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
