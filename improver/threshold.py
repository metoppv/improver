# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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


import operator

import iris
import numpy as np
from cf_units import Unit

from improver import PostProcessingPlugin
from improver.utilities.cube_manipulation import enforce_coordinate_ordering
from improver.utilities.rescale import rescale


class BasicThreshold(PostProcessingPlugin):

    """Apply a threshold truth criterion to a cube.

    Calculate the threshold truth values based on a linear membership function
    around the threshold values provided. A cube will be returned with a new
    threshold dimension coordinate.

    Can operate on multiple time sequences within a cube.
    """

    def __init__(self, thresholds, fuzzy_factor=None,
                 fuzzy_bounds=None, threshold_units=None,
                 comparison_operator='>'):
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
            thresholds (list of float or float):
                The threshold points for 'significant' datapoints.
            fuzzy_factor (float):
                Specifies lower bound for fuzzy membership value when
                multiplied by each threshold. Upper bound is equivalent linear
                distance above threshold. If None, no fuzzy_factor is applied.
            fuzzy_bounds (list of tuple):
                Lower and upper bounds for fuzziness.
                List should be of same length as thresholds.
                Each entry in list should be a tuple of two floats
                representing the lower and upper bounds respectively.
                If None, no fuzzy_bounds are applied.
            threshold_units (str):
                Units of the threshold values. If not provided the units are
                assumed to be the same as those of the input cube.
            comparison_operator (str):
                Indicates the comparison_operator to use with the threshold.
                e.g. 'ge' or '>=' to evaluate data >= threshold or '<' to
                evaluate data < threshold. When using fuzzy thresholds, there
                is no difference between < and <= or > and >=.
                Valid choices: > >= < <= gt ge lt le.

        Raises:
            ValueError: If a threshold of 0.0 is requested when using a fuzzy
                        factor.
            ValueError: If the fuzzy_factor is not greater than 0 and less
                        than 1.
            ValueError: If both fuzzy_factor and fuzzy_bounds are set
                        as this is ambiguous.
        """
        # ensure threshold is a list, even if only a single value is provided
        self.thresholds = thresholds
        if np.isscalar(thresholds):
            self.thresholds = [thresholds]

        # if necessary, set threshold units
        if threshold_units is None:
            self.threshold_units = None
        else:
            self.threshold_units = Unit(threshold_units)

        # initialise threshold coordinate name as None
        self.threshold_coord_name = None

        # read fuzzy factor or set (default) to 1 (no smoothing)
        fuzzy_factor_loc = 1.
        if fuzzy_factor is not None:
            if fuzzy_bounds is not None:
                raise ValueError(
                    "Invalid combination of keywords. Cannot specify "
                    "fuzzy_factor and fuzzy_bounds together")
            if not 0 < fuzzy_factor < 1:
                raise ValueError(
                    "Invalid fuzzy_factor: must be >0 and <1: {}".format(
                        fuzzy_factor))
            if 0 in self.thresholds:
                raise ValueError(
                    "Invalid threshold with fuzzy factor: cannot use a "
                    "multiplicative fuzzy factor with threshold == 0")
            fuzzy_factor_loc = fuzzy_factor

        # Set fuzzy-bounds.  If neither fuzzy_factor nor fuzzy_bounds is set,
        # both lower_thr and upper_thr default to the threshold value.  A test
        # of this equality is used later to determine whether to process with
        # a sharp threshold or fuzzy bounds.
        if fuzzy_bounds is None:
            self.fuzzy_bounds = []
            for thr in self.thresholds:
                lower_thr = thr * fuzzy_factor_loc
                upper_thr = thr * (2. - fuzzy_factor_loc)
                if thr < 0:
                    lower_thr, upper_thr = upper_thr, lower_thr
                self.fuzzy_bounds.append((lower_thr, upper_thr))
        else:
            self.fuzzy_bounds = fuzzy_bounds

        # ensure fuzzy_bounds is a list of tuples
        if isinstance(fuzzy_bounds, tuple):
            self.fuzzy_bounds = [fuzzy_bounds]

        # check that thresholds and fuzzy_bounds are self-consistent
        for thr, bounds in zip(self.thresholds, self.fuzzy_bounds):
            if len(bounds) != 2:
                raise ValueError("Invalid bounds for one threshold: {}."
                                 " Expected 2 floats.".format(bounds))
            if bounds[0] > thr or bounds[1] < thr:
                bounds_msg = ("Threshold must be within bounds: "
                              "!( {} <= {} <= {} )".format(bounds[0],
                                                           thr, bounds[1]))
                raise ValueError(bounds_msg)

        # Dict of known logical comparisons. Each key contains a dict of
        # {'function': The operator function for this comparison_operator,
        #  'spp_string': Comparison_Operator string for use in CF-convention
        #                meta-data}
        self.comparison_operator_dict = {}
        self.comparison_operator_dict.update(dict.fromkeys(
            ['ge', 'GE', '>='], {'function': operator.ge,
                                 'spp_string': 'above'}))
        self.comparison_operator_dict.update(dict.fromkeys(
            ['gt', 'GT', '>'], {'function': operator.gt,
                                'spp_string': 'above'}))
        self.comparison_operator_dict.update(dict.fromkeys(
            ['le', 'LE', '<='], {'function': operator.le,
                                 'spp_string': 'below'}))
        self.comparison_operator_dict.update(dict.fromkeys(
            ['lt', 'LT', '<'], {'function': operator.lt,
                                'spp_string': 'below'}))
        self.comparison_operator_string = comparison_operator
        self._decode_comparison_operator_string()

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        return (
            '<BasicThreshold: thresholds {}, ' +
            'fuzzy_bounds {}, ' +
            'method: data {} threshold>'
        ).format(self.thresholds, self.fuzzy_bounds,
                 self.comparison_operator_string)

    def _add_threshold_coord(self, cube, threshold):
        """
        Add a scalar threshold-type coordinate to a cube containing
        thresholded data and promote the new coordinate to be the
        leading dimension of the cube.

        Args:
            cube (iris.cube.Cube):
                Cube containing thresholded data (1s and 0s)
            threshold (float):
                Value at which the data has been thresholded

        Returns:
            iris.cube.Cube:
                With new "threshold" axis
        """
        coord = iris.coords.DimCoord(np.array([threshold], dtype=np.float32),
                                     units=cube.units)
        coord.rename(self.threshold_coord_name)
        coord.var_name = "threshold"

        # Use an spp__relative_to_threshold attribute, as an extension to the
        # CF-conventions.
        coord.attributes.update({'spp__relative_to_threshold':
                                 self.comparison_operator['spp_string']})

        cube.add_aux_coord(coord)
        return iris.util.new_axis(cube, coord)

    def _decode_comparison_operator_string(self):
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
                self.comparison_operator_string]
        except KeyError:
            msg = (f'String "{self.comparison_operator_string}" '
                   'does not match any known comparison_operator method')
            raise ValueError(msg)

    def process(self, input_cube):
        """Convert each point to a truth value based on provided threshold
        values. The truth value may or may not be fuzzy depending upon if
        fuzzy_bounds are supplied.  If the plugin has a "threshold_units"
        member, this is used to convert both thresholds and fuzzy bounds into
        the units of the input cube.

        Args:
            input_cube (iris.cube.Cube):
                Cube to threshold. The code is dimension-agnostic.

        Returns:
            iris.cube.Cube:
                Cube after a threshold has been applied. The data within this
                cube will contain values between 0 and 1 to indicate whether
                a given threshold has been exceeded or not.

                The cube meta-data will contain:
                * Input_cube name prepended with
                probability_of_X_above(or below)_threshold (where X is
                the diagnostic under consideration)
                * Threshold dimension coordinate with same units as input_cube
                * Threshold attribute (above or below threshold)
                * Cube units set to (1).

        Raises:
            ValueError: if a np.nan value is detected within the input cube.

        """
        # Record input cube data type to ensure consistent output, though
        # integer data must become float to enable fuzzy thresholding.
        input_cube_dtype = input_cube.dtype
        if input_cube.dtype.kind == 'i':
            input_cube_dtype = np.float32

        thresholded_cubes = iris.cube.CubeList()
        if np.isnan(input_cube.data).any():
            raise ValueError("Error: NaN detected in input cube data")

        # if necessary, convert thresholds and fuzzy bounds into cube units
        if self.threshold_units is not None:
            self.thresholds = [self.threshold_units.convert(threshold,
                                                            input_cube.units)
                               for threshold in self.thresholds]
            self.fuzzy_bounds = [tuple([
                self.threshold_units.convert(threshold, input_cube.units)
                for threshold in bounds]) for bounds in self.fuzzy_bounds]

        # set name of threshold coordinate to match input diagnostic
        self.threshold_coord_name = input_cube.name()

        # apply fuzzy thresholding
        for threshold, bounds in zip(self.thresholds, self.fuzzy_bounds):
            cube = input_cube.copy()
            # if upper and lower bounds are equal, set a deterministic 0/1
            # probability based on exceedance of the threshold
            if bounds[0] == bounds[1]:
                truth_value = self.comparison_operator['function'](
                    cube.data, threshold)
            # otherwise, scale exceedance probabilities linearly between 0/1
            # at the min/max fuzzy bounds and 0.5 at the threshold value
            else:
                truth_value = np.where(
                    cube.data < threshold,
                    rescale(cube.data,
                            data_range=(bounds[0], threshold),
                            scale_range=(0., 0.5),
                            clip=True),
                    rescale(cube.data,
                            data_range=(threshold, bounds[1]),
                            scale_range=(0.5, 1.),
                            clip=True),
                )
                # if requirement is for probabilities below threshold (rather
                # than above), invert the exceedance probability
                if 'below' in self.comparison_operator['spp_string']:
                    truth_value = 1. - truth_value
            truth_value = np.ma.masked_where(
                np.ma.getmask(cube.data), truth_value)
            truth_value = truth_value.astype(input_cube_dtype)

            cube.data = truth_value
            # Overwrite masked values that have been thresholded
            # with the un-thresholded values from the input cube.
            if np.ma.is_masked(cube.data):
                cube.data[input_cube.data.mask] = (
                    input_cube.data[input_cube.data.mask])
            cube = self._add_threshold_coord(cube, threshold)
            thresholded_cubes.append(cube)

        cube, = thresholded_cubes.concatenate()

        cube.rename(
            "probability_of_{}_{}_threshold".format(
                cube.name(),
                self.comparison_operator['spp_string']))
        cube.units = Unit(1)

        enforce_coordinate_ordering(
            cube, ["realization", "percentile"])

        return cube
