# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2022 Met Office.
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
from typing import Optional, Union

from iris.cube import Cube

from improver import BasePlugin

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

    def process(
        self,
        target_grid: Cube,
        time: datetime,
        accumulation_period: int,
        temporal_spacing: int = DEFAULT_TEMPORAL_SPACING_IN_MINUTES,
        altitude: Optional[Union[Cube, float]] = 0.0,
        linke_turbidity_climatology: Optional[Union[Cube, float]] = 3.0,
    ) -> Cube:
        """Calculate the gridded clear sky radiation data by integrating clear sky irradiance
        over the specified time-period.

        Args:
            target_grid:
                A cube containing the desired spatial grid.
            time:
                The valid time at which to evaluate the accumulated clearsky solar
                radiation. This time is taken to be the end of the accumulation period.
            accumulation_period:
                The number of hours over which the solar radiation accumulation is defined.
            temporal_spacing:
                The spacing between irradiance times used in the evaluation of the accumulated
                solar radiation, specified in mins.
            altitude:
                Altitude data used in the evaluation of the clearsky solar irradiance values,
                specified in metres.
            linke_turbidity_climatology:
                Linke turbidity data used in the evaluation of the clearsky solar irradiance
                values. Linke turbidity is a dimensionless quantity that accounts for the
                atmospheric scattering of radiation due to aerosols and water vapour, relative
                to a dry atmosphere. The linke turbidity climatology data is assumed to a time
                dimension that represents the day-of-year from which the associated climatological
                linke turbidity values can be interpolated to the specified valid-time.

        Returns:
            A cube containing the clearsky solar radiation accumulated over the specified
            period, on the same spatial grid as target_grid.
        """
        pass
