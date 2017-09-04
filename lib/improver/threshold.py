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


class BasicThreshold(object):

    """Apply a threshold truth criterion to a cube.

    Calculate the threshold truth value based on a linear membership function
    around the threshold.

    Can operate on multiple time sequences within a cube.

    """

    def __init__(self, threshold, fuzzy_factor=None,
                 below_thresh_ok=False):
        """Set up for processing an in-or-out of threshold binary field.

        Args:
            threshold : float
                The threshold point for 'significant' datapoints.
            fuzzy_factor : float
                Percentage above or below threshold for fuzzy membership value.
                If None, no fuzzy_factor is applied.
            below_thresh_ok : boolean
                True to count points as significant if *below* the threshold,
                False to count points as significant if *above* the threshold.

        Raises:
            ValueError: If a threshold of 0.0 is requested.
            ValueError: If the fuzzy_factor is not greater than 0 and less
                        than 1.

        """
        if threshold == 0.0:
            raise ValueError(
                "Invalid threshold: zero not allowed")
        self.threshold = threshold
        if fuzzy_factor is not None:
            if not 0 < fuzzy_factor < 1:
                raise ValueError(
                    "Invalid fuzzy_factor: must be >0 and <1: {}".format(
                        fuzzy_factor))
        self.fuzzy_factor = fuzzy_factor
        self.below_thresh_ok = below_thresh_ok

    def __str__(self):
        """Represent the configured plugin instance as a string."""
        return (
            '<BasicThreshold: threshold {}, fuzzy factor {}' +
            'below_thresh_ok: {}>'
        ).format(self.threshold, self.fuzzy_factor, self.below_thresh_ok)

    def process(self, cube):
        """Convert each point to a truth value based on threshold. The truth
        value may or may not be fuzzy depending upon if a fuzzy_factor is
        supplied.

        Args:
            cube : iris.cube.Cube
                Cube to threshold. The code is dimension-agnostic.

        Returns:
            cube : iris.cube.Cube
                Cube after a threshold has been applied. The data within this
                cube will contain values between 0 and 1 to indicate whether
                a given threshold has been exceeded or not.

        Raises:
            ValueError: if a np.nan value is detected within the input cube.

        """
        if np.isnan(cube.data).any():
            raise ValueError("Error: NaN detected in input cube data")
        if self.fuzzy_factor is None:
            truth_value = cube.data > self.threshold
        else:
            lower_threshold = self.threshold * self.fuzzy_factor
            truth_value = (
                (cube.data - lower_threshold) /
                ((self.threshold * (2. - self.fuzzy_factor)) - lower_threshold)
            )
        truth_value = np.clip(truth_value, 0., 1.)
        if self.below_thresh_ok:
            truth_value = 1. - truth_value
        cube.data = truth_value

        # Force the metadata to temporary conventions
        if self.below_thresh_ok:
            cube.attributes.update({'relative_to_threshold': 'below'})
        else:
            cube.attributes.update({'relative_to_threshold': 'above'})
        cube.rename("probability_of_{}".format(cube.name()))
        coord = iris.coords.DimCoord(self.threshold,
                                     long_name="threshold",
                                     units=cube.units)
        cube.add_aux_coord(coord)
        cube.units = Unit(1)
        return cube
