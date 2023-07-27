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

"""Applies height adjustment for height above ground level spot forecasts."""

from itertools import product

import iris
import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError
from scipy.interpolate import LinearNDInterpolator

from improver import BasePlugin
from improver.metadata.probabilistic import find_threshold_coordinate, is_probability
from improver.spotdata.apply_lapse_rate import extract_vertical_displacements
from improver.utilities.cube_manipulation import enforce_coordinate_ordering


class SpotHeightAdjustment(BasePlugin):
    """
    Class to adjust spot extracted "height above ground level" forecasts to account
    for differences between site height and orography grid square height. The spot
    forecast contains information representative of the associated grid point and as
    such this needs to be adjusted to reflect the true site altitude.

    For realization or percentile data the vertical displacement is added on to
    each realization or percentile.

    For probability data the data is interpolated between thresholds for each site and
    the equivalent set of thresholds relative to the site altitude extracted. Any
    new threshold that is above or below the original set of thresholds uses the highest
    or lowest threshold's probability from the original cube respectively for that spot.
    """

    def __init__(self, neighbour_selection_method: str = "nearest",) -> None:
        """
        Args:
            neighbour_selection_method:
                The neighbour cube may contain one or several sets of grid
                coordinates that match a spot site. These are determined by
                the neighbour finding method employed. This keyword is used to
                extract the desired set of coordinates from the neighbour cube.
        """
        self.neighbour_selection_method = neighbour_selection_method
        self.threshold_coord = DimCoord
        self.units = None

    def adjust_prob_cube(self, spot_cube: Cube, vertical_displacement: Cube) -> Cube:
        """
        Adjust probability spot forecasts based on the vertical displacement of sites
        in relation to orography.

        Args:
            spot_cube:
                A cube of spot forecasts. If this is a cube of probabilities
                then the units of the threshold coordinate must be convertible to
                metres as this is expected to represent a vertical coordinate. There
                must be at least two thresholds on this cube.
                If this is a cube of percentiles or realizations then the
                units of the cube must be convertible to metres as the cube is
                expected to represent a vertical profile.
            vertical_displacement:
                A cube containing information about the difference between spot
                data site height and the orography grid square height.

        Returns:
            A cube with the same metadata and shape as spot_cube but with probabilities
            adjusted to be relative to the site altitude rather than grid square altitude.
        """

        enforce_coordinate_ordering(
            spot_cube, ["spot_index", self.threshold_coord.name()]
        )
        n_sites = len(spot_cube.coord("spot_index").points)
        n_thresholds = len(self.threshold_coord.points)

        thresholds = self.threshold_coord.points
        spot_index = spot_cube.coord("spot_index").points
        shape = spot_cube.shape

        # Maximum probability over all thresholds for each site.
        broadcast_max = np.transpose(
            np.broadcast_to(np.amax(spot_cube.data, axis=1), (n_thresholds, n_sites))
        )
        # Minimum probability over all thresholds for each site.
        broadcast_min = np.transpose(
            np.broadcast_to(np.amin(spot_cube.data, axis=1), (n_thresholds, n_sites))
        )

        broadcast_thresholds = np.broadcast_to(thresholds, (n_sites, n_thresholds))
        broadcast_vertical_displacement = np.transpose(
            np.broadcast_to(vertical_displacement.data, (n_thresholds, n_sites),)
        )
        desired_thresholds = broadcast_thresholds + broadcast_vertical_displacement.data

        # creates a list of pairs of values of spot index with the thresholds that need to
        # be calculated for the spot index
        coord = list(product(spot_index, thresholds))
        needed_pair = []
        for index, threshold in zip(spot_index, desired_thresholds):
            needed_pair.extend(list(product([index], threshold)))

        # interpolate across the cube and request needed thresholds
        interp = LinearNDInterpolator(coord, spot_cube.data.flatten())
        spot_data = np.reshape(interp(needed_pair), shape)

        # Points outside the range of the original data return NAN. These points are replaced
        # with the highest or lowest along the axis depending on the whether the vertical
        # displacement was positive or negative
        indices = np.where(np.isnan(spot_data))
        spot_data[indices] = np.where(
            broadcast_vertical_displacement[indices] > 0,
            broadcast_max[indices],
            broadcast_min[indices],
        )
        spot_cube.data = spot_data
        return spot_cube

    def process(self, spot_cube: Cube, neighbour: Cube,) -> Cube:
        """
        Adjusts spot forecast data to be relative to site height rather than
        grid square orography height.

        Args:
            spot_cube:
                A cube of spot forecasts. If this is a cube of probabilities
                then the units of the threshold coordinate must be convertible to
                metres as this is expected to represent a vertical coordinate. There
                must be at least two thresholds on this cube.
                If this is a cube of percentiles or realizations then the
                units of the cube must be convertible to metres as the cube is
                expected to represent a vertical profile.
            neighbour:
                A cube containing information about the spot data sites and
                their grid point neighbours.
        Returns:
            A cube of the same shape as spot_data but with data adjusted to account for
            the difference between site height and orography height.

        Raises:
            ValueError:
                If spot_cube is a probability cube and there are fewer than two thresholds.
        """
        vertical_displacement = extract_vertical_displacements(
            neighbour_cube=neighbour,
            neighbour_selection_method_name=self.neighbour_selection_method,
        )

        if is_probability(spot_cube):
            threshold_coord = find_threshold_coordinate(spot_cube)
            self.threshold_coord = threshold_coord.copy()

            if len(self.threshold_coord.points) < 2:
                raise ValueError(
                    f"There are fewer than 2 thresholds present in this cube, this is "
                    f"{spot_cube.coord(threshold_coord).points}. At least two thresholds "
                    "are needed for interpolation"
                )

            self.units = self.threshold_coord.units
            self.threshold_coord.convert_units("m")

            try:
                cube_slices = [x for x in spot_cube.slices_over("realization")]
            except CoordinateNotFoundError:
                cube_slices = [spot_cube]
            coord_list = [c.name() for c in spot_cube.dim_coords]
            spot_data = iris.cube.CubeList()
            for cube_slice in cube_slices:
                spot_data.append(
                    self.adjust_prob_cube(cube_slice, vertical_displacement)
                )
            spot_cube = spot_data.merge_cube()
            enforce_coordinate_ordering(spot_cube, coord_list)

        else:
            self.units = spot_cube.units
            spot_cube.convert_units("m")

            spot_cube.data = spot_cube.data + vertical_displacement.data
            spot_cube.convert_units(self.units)
        spot_cube.data = spot_cube.data.astype(np.float32)
        return spot_cube
