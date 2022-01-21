# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""Module containing neighbourhood processing utilities."""

from typing import List, Optional, Union

import iris
import numpy as np
from iris.cube import Cube
from numpy import ndarray

from improver import PostProcessingPlugin
from improver.metadata.forecast_times import forecast_period_coord

# from improver.nbhood.square_kernel import Neighbourhood
from improver.utilities.cube_checker import (
    check_cube_coordinates,
    find_dimension_coordinate_mismatch,
)
from improver.utilities.cube_manipulation import MergeCubes


class BaseNeighbourhoodProcessing(PostProcessingPlugin):
    """
    Apply a neighbourhood processing method to a thresholded cube. This is a
    base class for usage with a subclass that will inherit the functionality
    within this base class.

    When applied to a thresholded probabilistic cube, it acts like a
    low-pass filter which reduces noisiness in the probabilities.

    The neighbourhood methods will presently only work with projections in
    which the x grid point spacing and y grid point spacing are constant
    over the entire domain, such as the UK national grid projection

    """

    def __init__(
        self, radii: Union[float, List[float]], lead_times: Optional[List] = None,
    ) -> None:
        """
        Create a neighbourhood processing plugin that applies a smoothing
        to points in a cube.

            radii:
                The radii in metres of the neighbourhood to apply.
                Rounded up to convert into integer number of grid
                points east and north, based on the characteristic spacing
                at the zero indices of the cube projection-x and y coords.
            lead_times:
                List of lead times or forecast periods, at which the radii
                within 'radii' are defined. The lead times are expected
                in hours.
        """
        if isinstance(radii, list):
            self.radii = [float(x) for x in radii]
        else:
            self.radius = float(radii)
        self.lead_times = lead_times
        if self.lead_times is not None:
            if len(radii) != len(lead_times):
                msg = (
                    "There is a mismatch in the number of radii "
                    "and the number of lead times. "
                    "Unable to continue due to mismatch."
                )
                raise ValueError(msg)

    def _find_radii(
        self, cube_lead_times: Optional[ndarray] = None
    ) -> Union[float, ndarray]:
        """Revise radius or radii for found lead times.
        If cube_lead_times is None, no automatic adjustment
        of the radii will take place.
        Otherwise it will interpolate to find the radius at
        each cube lead time as required.

        Args:
            cube_lead_times:
                Array of forecast times found in cube.

        Returns:
            Required neighbourhood sizes.
        """
        radii = np.interp(cube_lead_times, self.lead_times, self.radii)
        return radii

    def run(self, cube_slice, mask_cube: Optional[Cube] = None):
        return cube_slice

    def process(self, cube: Cube, mask_cube: Optional[Cube] = None) -> Cube:
        """
        Supply neighbourhood processing method, in order to smooth the
        input cube.

        Args:
            cube:
                Cube to apply a neighbourhood processing method to, in order to
                generate a smoother field.
            mask_cube:
                Cube containing the array to be used as a mask.

        Returns:
            Cube after applying a neighbourhood processing method, so that
            the resulting field is smoothed.
        """
        # Check if a dimensional realization coordinate exists. If so, the
        # cube is sliced, so that it becomes a scalar coordinate.
        try:
            cube.coord("realization", dim_coords=True)
        except iris.exceptions.CoordinateNotFoundError:
            slices_over_realization = [cube]
        else:
            slices_over_realization = cube.slices_over(["realization", "time"])

        if np.isnan(cube.data).any():
            raise ValueError("Error: NaN detected in input cube data")

        cubes_real = []
        for cube_realization in slices_over_realization:
            if self.lead_times:
                # Interpolate to find the radius at each required lead time.
                fp_coord = forecast_period_coord(cube_realization)
                fp_coord.convert_units("hours")
                self.radius = self._find_radii(cube_lead_times=fp_coord.points)

            cube_new = self.run(cube_realization, mask_cube=mask_cube)

            cubes_real.append(cube_new)

        if len(cubes_real) > 1:
            combined_cube = MergeCubes()(cubes_real, slice_over_realization=True)
        else:
            combined_cube = cubes_real[0]

        # Promote dimensional coordinates that used to be present.
        exception_coordinates = find_dimension_coordinate_mismatch(
            cube, combined_cube, two_way_mismatch=False
        )
        combined_cube = check_cube_coordinates(
            cube, combined_cube, exception_coordinates=exception_coordinates
        )

        return combined_cube
