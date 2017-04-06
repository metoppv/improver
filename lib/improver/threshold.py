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


class BasicThreshold(object):

    """Apply a threshold truth criterion to a cube.

    Calculate the threshold truth value based on a linear membership function
    around the threshold.

    Can operate on multiple time sequences within a cube.

    """

    def __init__(self, threshold, fuzzy_factor,
                 below_thresh_ok=False):
        """Set up processing for an in-or-out of threshold binary field.

        Parameters
        ----------

        threshold : float
            The threshold point for 'significant' datapoints.

        fuzzy_factor : float
            Percentage above or below threshold for fuzzy membership value.

        below_thresh_ok : boolean
            True to count points as significant if *below* the threshold,
            False to count points as significant if *above* the threshold.

        """
        if threshold == 0.0:
            raise ValueError(
                "Invalid threshold: zero not allowed")
        self.threshold = threshold
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
        """Convert each point to a fuzzy truth value based on threshold.

        Parameters
        ----------

        cube - iris.cube.Cube
            Cube to threshold. The code is dimension-agnostic, so any
            setup should work.

        """
        lower_threshold = self.threshold * self.fuzzy_factor
        if np.isnan(cube.data).any():
            raise ValueError("Error: NaN detected in input cube data")
        truth_value = (
            (cube.data - lower_threshold) /
            ((self.threshold * (2. - self.fuzzy_factor)) - lower_threshold)
        )
        truth_value = np.clip(truth_value, 0., 1.)
        if self.below_thresh_ok:
            truth_value = 1. - truth_value
        cube.data = truth_value
        return cube
