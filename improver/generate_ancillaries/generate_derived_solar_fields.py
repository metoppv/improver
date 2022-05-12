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
"""Module for generating derived solar fields."""
from datetime import datetime
from typing import Tuple

import numpy as np
from iris.cube import Cube

from improver import BasePlugin
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.cube_checker import spatial_coords_match

DEFAULT_TEMPORAL_SPACING_IN_MINUTES = 30


class GenerateSolarTime(BasePlugin):
    """A plugin to evaluate local solar time."""

    def process(self, target_grid: Cube, time: datetime) -> Cube:
        """Calculate the local solar time over the specified grid.

        Args:
            target_grid:
                A cube containing the desired spatial grid.
            time:
                The valid time at which to evaluate the local solar time.

        Returns:
            A cube containing local solar time, on the same spatial grid as target_grid.
        """
        pass


class GenerateClearskySolarRadiation(BasePlugin):
    """A plugin to evaluate clearsky solar radiation."""

    def _initialise_input_cubes(
        self, target_grid: Cube, surface_altitude: Cube, linke_turbidity: Cube
    ) -> Tuple[Cube, Cube]:
        """Assign default values to input cubes where none have been passed, and ensure
        that all cubes are defined over consistent spatial grid.

        Args:
            target_grid:
                A cube containing the desired spatial grid.
            surface_altitude:
                Input surface altitude value.
            linke_turbidity:
                Input linke-turbidity value.

        Returns:
            - Cube containing surface altitude, defined on the same grid as target_grid.
            - Cube containing linke-turbidity, defined on the same grid as target_grid.

        Raises:
            ValueError:
                If surface_altitude or linke_turbidity have inconsistent spatial coords
                relative to target_grid.
        """
        if surface_altitude is None:
            # Create surface_altitude cube using target_grid as template.
            surface_altitude_data = np.zeros(shape=target_grid.shape, dtype=np.float32)
            surface_altitude = create_new_diagnostic_cube(
                name="surface_altitude",
                units="m",
                template_cube=target_grid,
                mandatory_attributes=generate_mandatory_attributes([target_grid]),
                optional_attributes=target_grid.attributes,
                data=surface_altitude_data,
            )
        else:
            if not spatial_coords_match([target_grid, surface_altitude]):
                raise ValueError(
                    "surface altitude spatial coordinates do not match target_grid"
                )

        if linke_turbidity is None:
            # Create linke_turbidity cube using target_grid as template.
            linke_turbidity_data = 3.0 * np.ones(
                shape=target_grid.shape, dtype=np.float32
            )
            linke_turbidity = create_new_diagnostic_cube(
                name="linke_turbidity",
                units="1",
                template_cube=target_grid,
                mandatory_attributes=generate_mandatory_attributes([target_grid]),
                optional_attributes=target_grid.attributes,
                data=linke_turbidity_data,
            )
        else:
            if not spatial_coords_match([target_grid, linke_turbidity]):
                raise ValueError(
                    "linke-turbidity spatial coordinates do not match target_grid"
                )

        return surface_altitude, linke_turbidity

    def process(
        self,
        target_grid: Cube,
        time: datetime,
        accumulation_period: int,
        surface_altitude: Cube = None,
        linke_turbidity: Cube = None,
        temporal_spacing: int = DEFAULT_TEMPORAL_SPACING_IN_MINUTES,
    ) -> Cube:
        """Calculate the gridded clearsky solar radiation by integrating clearsky solar irradiance
        values over the specified time-period, and on the specified grid.

        Args:
            target_grid:
                A cube containing the desired spatial grid.
            time:
                The valid time at which to evaluate the accumulated clearsky solar
                radiation. This time is taken to be the end of the accumulation period.
            accumulation_period:
                The number of hours over which the solar radiation accumulation is defined.
            surface_altitude:
                Surface altitude data, specified in metres, used in the evaluation of the clearsky
                solar irradiance values.
            linke_turbidity:
                Linke turbidity data used in the evaluation of the clearsky solar irradiance
                values. Linke turbidity is a dimensionless quantity that accounts for the
                atmospheric scattering of radiation due to aerosols and water vapour, relative
                to a dry atmosphere.
            temporal_spacing:
                The time stepping, specified in mins, used in the integration of solar irradiance
                to produce the accumulated solar radiation.

        Returns:
            A cube containing the clearsky solar radiation accumulated over the specified
            period, on the same spatial grid as target_grid.
        """
        surface_altitude, linke_turbidity = self._initialise_input_cubes(
            target_grid, surface_altitude, linke_turbidity
        )

        pass
