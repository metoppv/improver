# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Calculation of expected value from a probability distribution."""

import numpy as np
from iris.coords import CellMethod
from iris.cube import Cube

from improver import PostProcessingPlugin
from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    RebadgePercentilesAsRealizations,
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
        # check the thresholds are in increasing order
        if np.any(np.diff(thresholds) <= 0.0):
            raise ValueError("threshold coordinate in decreasing order")
        # add an extra threshold below/above with zero/one probability
        # ensure that the PDF integral covers the full CDF probability range
        # thresholds are usually float32 with epsilon of ~= 1.1e-7
        eps = np.finfo(thresholds.dtype).eps
        # for small values (especially exactly zero), at least epsilon
        # for larger values, the next representable float number
        thresholds_expanded = np.array(
            [
                thresholds[0] - max(eps, abs(thresholds[0] * eps)),
                *thresholds,
                thresholds[-1] + max(eps, abs(thresholds[-1] * eps)),
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
