# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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


import numpy as np
import iris
from cf_units import Unit
from improver.spotdata.extract_data import ExtractData


class BasicThreshold(object):

    """Apply a threshold truth criterion to a cube.

    Calculate the threshold truth values based on a linear membership function
    around the threshold values provided. A cube will be returned with a new
    threshold dimension coordinate.

    Can operate on multiple time sequences within a cube.

    """

    def __init__(self, thresholds, fuzzy_factor=None,
                 below_thresh_ok=False):
        """Set up for processing an in-or-out of threshold binary field.

        Args:
            thresholds : list of floats or float
                The threshold points for 'significant' datapoints.
            fuzzy_factor : float
                Percentage above or below threshold for fuzzy membership value.
                If None, no fuzzy_factor is applied.
            below_thresh_ok : boolean
                True to count points as significant if *below* the threshold,
                False to count points as significant if *above* the threshold.

        Raises:
            ValueError: If a threshold of 0.0 is requested when using a fuzzy
                        factor.
            ValueError: If the fuzzy_factor is not greater than 0 and less
                        than 1.

        """
        # Ensure iterable threshold list provided, even if it's a single value.
        self.thresholds = thresholds
        if np.isscalar(thresholds):
            self.thresholds = [thresholds]

        if fuzzy_factor is not None:
            if not 0 < fuzzy_factor < 1:
                raise ValueError(
                    "Invalid fuzzy_factor: must be >0 and <1: {}".format(
                        fuzzy_factor))
            if 0 in self.thresholds:
                raise ValueError(
                    "Invalid threshold with fuzzy factor: cannot use a "
                    "multiplicative fuzzy factor with threshold == 0")

        self.fuzzy_factor = fuzzy_factor
        self.below_thresh_ok = below_thresh_ok

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        return (
            '<BasicThreshold: thresholds {}, fuzzy factor {}, ' +
            'below_thresh_ok: {}>'
        ).format(self.thresholds, self.fuzzy_factor, self.below_thresh_ok)

    def process(self, input_cube):
        """Convert each point to a truth value based on provided threshold
        values. The truth value may or may not be fuzzy depending upon if a
        fuzzy_factor is supplied.

        Args:
            input_cube : iris.cube.Cube
                Cube to threshold. The code is dimension-agnostic.

        Returns:
            cube : iris.cube.Cube
                Cube after a threshold has been applied. The data within this
                cube will contain values between 0 and 1 to indicate whether
                a given threshold has been exceeded or not.
                The cube meta-data will contain:
                 * input_cube name prepended with "probability_of_"
                 * threshold dimension coordinate with same units as input_cube
                 * threshold attribute (above or below threshold)
                 * cube units set to (1).

        Raises:
            ValueError: if a np.nan value is detected within the input cube.

        """
        thresholded_cubes = iris.cube.CubeList()
        if np.isnan(input_cube.data).any():
            raise ValueError("Error: NaN detected in input cube data")

        for threshold in self.thresholds:
            cube = input_cube.copy()
            if self.fuzzy_factor is None:
                truth_value = cube.data > threshold
            else:
                lower_threshold = threshold * self.fuzzy_factor
                truth_value = (
                    (cube.data - lower_threshold) /
                    ((threshold * (2. - self.fuzzy_factor)) - lower_threshold)
                )
            truth_value = np.clip(truth_value, 0., 1.).astype(np.float64)
            if self.below_thresh_ok:
                truth_value = 1. - truth_value
            cube.data = truth_value

            cube.rename("probability_of_{}".format(cube.name()))
            coord = iris.coords.DimCoord(threshold,
                                         long_name="threshold",
                                         units=cube.units)
            cube.add_aux_coord(coord)
            cube = iris.util.new_axis(cube, 'threshold')
            thresholded_cubes.append(cube)

        cube, = thresholded_cubes.concatenate()

        # TODO: Correct when formal cf-standards exists
        # Force the metadata to temporary conventions
        if self.below_thresh_ok:
            cube.attributes.update({'relative_to_threshold': 'below'})
        else:
            cube.attributes.update({'relative_to_threshold': 'above'})
        cube.units = Unit(1)

        cube = ExtractData.make_stat_coordinate_first(cube)

        return cube
