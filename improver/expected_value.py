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
"""Calculation of expected value from a probability distribution."""

import numpy as np
from iris.coords import CellMethod
from iris.cube import Cube

from improver import PostProcessingPlugin
from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    RebadgePercentilesAsRealizations,
    get_bounds_of_distribution,
)
from improver.metadata.probabilistic import (
    find_threshold_coordinate,
    is_percentile,
    is_probability,
)
from improver.utilities.cube_manipulation import collapse_realizations
from improver.utilities.probability_manipulation import to_threshold_inequality


class ExpectedValue(PostProcessingPlugin):
    """Calculation of expected value from a probability distribution"""

    @staticmethod
    def integrate_over_thresholds(cube: Cube) -> Cube:
        """
        Calculation of expected value for threshold data by converting from
        cumulative distribution (CDF) to probability density (PDF), then
        integrating over the threshold dimension.

        Args:
            cube:
                Probabilistic data with a threshold coordinate.

        Returns:
            Expected value of probability distribution. Same shape as input cube
            but with threshold coordinate removed.
        """
        # make sure that the threshold direction is "below"
        cube = to_threshold_inequality(cube, above=False)
        # set up threshold values, these will be needed as a multiplier during integration
        threshold_coord = find_threshold_coordinate(cube)
        threshold_coord_idx = cube.dim_coords.index(threshold_coord)
        thresholds = threshold_coord.points
        # add an extra threshold below/above with zero/one probability
        # ensure that the PDF integral covers the full CDF probability range
        try:
            ecc_bounds = get_bounds_of_distribution(
                threshold_coord.name(), threshold_coord.units
            )
        except KeyError:
            # no bound available, this will be skipped below via floating point rules
            # eg. min(infinty, a) = a and max(-infinity, b) = b
            ecc_bounds = np.array([np.inf, -np.inf])
        # expand to the widest of ECC bounds or +/- mean threshold spacing
        # this will always expand, even if the original data covered the full ECC bounds range
        threshold_spacing = np.mean(np.diff(thresholds))
        thresholds_expanded = np.array(
            [
                min(ecc_bounds[0], thresholds[0] - threshold_spacing),
                *thresholds,
                max(ecc_bounds[1], thresholds[-1] + threshold_spacing),
            ]
        )
        # expand the data to match the newly added thresholds
        extra_data_shape = list(cube.shape)
        extra_data_shape[threshold_coord_idx] = 1
        data_expanded = np.concatenate(
            [np.zeros(extra_data_shape), cube.data, np.ones(extra_data_shape)],
            axis=threshold_coord_idx,
            dtype=np.float32,
        )
        data_pdf = np.diff(data_expanded, axis=threshold_coord_idx)
        del data_expanded
        # the PDF should always be positive
        if np.any(data_pdf < 0.0):
            raise ValueError(
                "PDF contains negative values - CDF was likely not monotonic increasing"
            )
        # use the midpoint of each threshold to weight the PDF
        threshold_midpoints = thresholds_expanded[:-1] + 0.5 * np.diff(
            thresholds_expanded
        )
        # expand the shape with additional length-1 dimensions so it broadcasts with data_pdf
        thresh_mid_shape = [
            len(threshold_midpoints) if i == threshold_coord_idx else 1
            for i in range(data_pdf.ndim)
        ]
        threshold_midpoints_bcast = threshold_midpoints.reshape(thresh_mid_shape)
        # apply threshold weightings to produce a weighed PDF suitable for integration
        weighted_pdf = data_pdf * threshold_midpoints_bcast
        del data_pdf
        # sum of weighted_pdf is equivalent to midpoint rule integration over the CDF
        ev_data = np.sum(weighted_pdf, axis=threshold_coord_idx, dtype=np.float32)
        del weighted_pdf
        # set up output cube based on input, with the threshold dimension removed
        ev_cube = next(cube.slices_over([threshold_coord])).copy()
        ev_cube.remove_coord(threshold_coord)
        # name and units come from the threshold coordinate
        ev_cube.rename(threshold_coord.name())
        ev_cube.units = threshold_coord.units
        # replace data with calculated values
        ev_cube.data = ev_data
        return ev_cube

    def process(self, cube: Cube) -> Cube:
        """Expected value calculation and metadata updates.

        Args:
            cube:
                Probabilistic data with a realization, threshold or percentile
                representation.

        Returns:
            Expected value of probability distribution. Same shape as input cube
            but with realization/threshold/percentile coordinate removed.
        """
        if is_probability(cube):
            ev_cube = self.integrate_over_thresholds(cube)
        elif is_percentile(cube):
            ev_cube = collapse_realizations(
                RebadgePercentilesAsRealizations().process(cube)
            )
        else:
            ev_cube = collapse_realizations(cube)
        ev_cube.add_cell_method(CellMethod("mean", coords="realization"))
        return ev_cube
