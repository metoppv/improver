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

"""Apply temperature lapse rate adjustments to a spot data cube."""
from typing import Optional

import iris
import numpy as np
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError

from improver import PostProcessingPlugin
from improver.lapse_rate import compute_lapse_rate_adjustment
from improver.metadata.probabilistic import is_probability
from improver.spotdata.spot_extraction import SpotExtraction, check_grid_match


class SpotLapseRateAdjust(PostProcessingPlugin):
    """
    Adjusts spot data temperatures by a lapse rate to better represent the
    conditions at their altitude that may not be captured by the model
    orography.
    """

    def __init__(
        self,
        neighbour_selection_method: str = "nearest",
        fixed_lapse_rate: float = None,
    ) -> None:
        """
        Args:
            neighbour_selection_method:
                The neighbour cube may contain one or several sets of grid
                coordinates that match a spot site. These are determined by
                the neighbour finding method employed. This keyword is used to
                extract the desired set of coordinates from the neighbour cube.
                The methods available will be all, or a subset, of the
                following::

                  - nearest
                  - nearest_land
                  - nearest_land_minimum_dz
                  - nearest_minimum_dz

                The method available in a neighbour cube will depend on the
                options that were specified when it was created.

            fixed_lapse_rate:
                A float value that is the lapse rate to apply to all sites.
                This can be used instead of providing a gridded lapse rate
                cube. Value is given in Kelvin / metre of temperature change
                with ascent. For example a dry adiabatic lapse rate would be
                given as -0.0098.
        """
        self.neighbour_selection_method = neighbour_selection_method
        self.fixed_lapse_rate = fixed_lapse_rate
        self.use_fixed_lr = self.fixed_lapse_rate is not None

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string."""
        return "<SpotLapseRateAdjust: neighbour_selection_method: {}" ">".format(
            self.neighbour_selection_method
        )

    def _check_inputs(
        self, spot_data_cube: Cube, neighbour_cube: Cube, gridded_lapse_rate_cube: Cube
    ):
        """
        Checks cubes for compatibility, raising Exceptions as appropriate.

        Raises:
            ValueError:
                If the spot_data_cube are probabilities.
            ValueError:
                If the lapse rate cube was provided but the diagnostic being
                processed is not air temperature or feels-like temperature.
            ValueError:
                If neither or both a lapse rate cube and a fixed lapse rate
                are provided.
            ValueError:
                If the lapse rate cube provided does not have the name
                "air_temperature_lapse_rate"
            ValueError:
                If the lapse rate cube does not contain a single valued height
                coordinate or the heights do not match the spot_data_cube.
        """
        if is_probability(spot_data_cube):
            msg = (
                "Input cube has a probability coordinate which cannot be lapse "
                "rate adjusted. Input data should be in percentile or "
                "deterministic space only."
            )
            raise ValueError(msg)

        # Check that we are dealing with temperature data.
        if spot_data_cube.name() not in ["air_temperature", "feels_like_temperature"]:
            msg = (
                "The diagnostic being processed is not air temperature "
                "or feels like temperature and therefore cannot be adjusted."
            )
            raise ValueError(msg)

        if gridded_lapse_rate_cube is None:
            if not self.use_fixed_lr:
                msg = (
                    "No lapse rate cube has been provided, and no fixed lapse rate "
                    "has been set. Provide one or other."
                )
                raise ValueError(msg)

            cubes_to_check = [neighbour_cube, spot_data_cube]

        else:
            if self.use_fixed_lr:
                msg = (
                    "Both a lapse rate cube and a fixed lapse rate have been provided. "
                    "Provide only one source of lapse rate information."
                )
                raise ValueError(msg)

            cubes_to_check = [neighbour_cube, spot_data_cube, gridded_lapse_rate_cube]

            if not gridded_lapse_rate_cube.name() == "air_temperature_lapse_rate":
                msg = (
                    "A cube has been provided as a lapse rate cube but does "
                    "not have the expected name air_temperature_lapse_rate: "
                    "{}".format(gridded_lapse_rate_cube.name())
                )
                raise ValueError(msg)
            try:
                lapse_rate_height_coord = gridded_lapse_rate_cube.coord("height")
            except CoordinateNotFoundError:
                msg = (
                    "Lapse rate cube does not contain a single valued height "
                    "coordinate. This is required to ensure it is applied to "
                    "equivalent temperature data."
                )
                raise CoordinateNotFoundError(msg)

            # Check the height of the temperature data matches that used to
            # calculate the lapse rates. If so, adjust temperatures using the lapse
            # rate values.
            if not spot_data_cube.coord("height") == lapse_rate_height_coord:
                raise ValueError(
                    "A lapse rate cube was provided, but the height of the "
                    "temperature data does not match that of the data used "
                    "to calculate the lapse rates. As such the temperatures "
                    "were not adjusted with the lapse rates."
                )

        # Check the cubes are compatible.
        check_grid_match(cubes_to_check)

    def get_gridded_lapse_rate(
        self,
        spot_data_cube: Cube,
        neighbour_cube: Cube,
        gridded_lapse_rate_cube: Cube = None,
    ) -> np.ndarray:
        """
        Extract the lapse rates that correspond to the spot sites and broadcast to the
        spot data shape.
        """
        spot_lapse_rate = SpotExtraction(
            neighbour_selection_method=self.neighbour_selection_method
        )(neighbour_cube, gridded_lapse_rate_cube)
        return iris.util.broadcast_to_shape(
            spot_lapse_rate.data, spot_data_cube.shape, [-1]
        )

    def broadcast_fixed_lapse_rate(self, spot_data_cube: Cube) -> np.ndarray:
        """Create an array of fixed lapse rate values"""
        return np.full(spot_data_cube.shape, self.fixed_lapse_rate, dtype=np.float32)

    def extract_vertical_displacements(self, neighbour_cube: Cube) -> Cube:
        """Extract vertical displacements between the model orography and sites."""
        method_constraint = iris.Constraint(
            neighbour_selection_method_name=self.neighbour_selection_method
        )
        data_constraint = iris.Constraint(grid_attributes_key="vertical_displacement")
        vertical_displacement = neighbour_cube.extract(
            method_constraint & data_constraint
        )
        return vertical_displacement

    def process(
        self,
        spot_data_cube: Cube,
        neighbour_cube: Cube,
        gridded_lapse_rate_cube: Optional[Cube] = None,
    ) -> Cube:
        """
        Extract lapse rates from the appropriate grid points and apply them to
        the spot extracted temperatures.

        The calculation is::

         lapse_rate_adjusted_temperatures = temperatures + lapse_rate *
         vertical_displacement

        Args:
            spot_data_cube:
                A spot data cube of temperatures for the spot data sites,
                extracted from the gridded temperature field. These
                temperatures will have been extracted using the same
                neighbour_cube and neighbour_selection_method that are being
                used here.
            neighbour_cube:
                The neighbour_cube that contains the grid coordinates at which
                lapse rates should be extracted and the vertical displacement
                between those grid points on the model orography and the spot
                data sites actual altitudes. This cube is only updated when
                a new site is added.
            gridded_lapse_rate_cube:
                A cube of temperature lapse rates on the same grid as that from
                which the spot data temperatures were extracted. If not provided
                a fixed lapse rate must have been specified.

        Returns:
            A copy of the input spot_data_cube with the data modified by
            the lapse rates to give a better representation of the site's
            temperatures.
        """
        self._check_inputs(spot_data_cube, neighbour_cube, gridded_lapse_rate_cube)

        # Apply lapse rate adjustment to the temperature at each site.
        if self.use_fixed_lr:
            lapse_rate_values = self.broadcast_fixed_lapse_rate(spot_data_cube)
        else:
            lapse_rate_values = self.get_gridded_lapse_rate(
                spot_data_cube, neighbour_cube, gridded_lapse_rate_cube
            )

        vertical_displacement = self.extract_vertical_displacements(neighbour_cube)

        new_temperatures = (
            spot_data_cube.data
            + compute_lapse_rate_adjustment(
                lapse_rate_values, vertical_displacement.data
            )
        ).astype(np.float32)
        return spot_data_cube.copy(data=new_temperatures)
