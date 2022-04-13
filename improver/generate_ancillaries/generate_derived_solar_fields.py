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
from datetime import datetime, timedelta
from typing import List, Tuple, Union

import cf_units
import numpy as np
from iris.coords import AuxCoord
from iris.cube import Cube
from numpy import ndarray

from improver import BasePlugin
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.cube_checker import spatial_coords_match

SECONDS_IN_MINUTE = 60
MINUTES_IN_HOUR = 60
DEFAULT_TEMPORAL_SPACING_IN_MINUTES = 30

CLEARSKY_SOLAR_RADIATION_CF_NAME = (
    "integral_of_surface_downwelling_shortwave_flux_in_air_assuming_clear_sky_wrt_time"
)
CLEARSKY_SOLAR_RADIATION_BRUCE_NAME = "clearsky_solar_radiation"


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

    def _get_irradiance_times(
        self, time: datetime, accumulation_period: int, temporal_spacing: int
    ) -> ndarray:
        """Get evenly-spaced times over the specied time period at which
        to evaluate irradiance values which will later be integrated to
        give accumulated solar-radiation values.

        Args:
            time:
                Datetime specifying the end of the accumulation period.
            accumulation_period:
                Time window over which solar radiation is to be accumulated,
                specified in hours.
            temporal_spacing:
                Spacing between irradiance times used in the evaluation of the
                accumulated solar radiation, specified in minutes.

        Returns:
            A list of datetimes.
        """
        if accumulation_period * MINUTES_IN_HOUR % temporal_spacing != 0:
            raise ValueError(
                (
                    f"accumulation_period in minutes ({accumulation_period} * 60) must be integer "
                    f"multiple of temporal_spacing ({temporal_spacing})."
                )
            )

        accumulation_start_time = time - timedelta(hours=accumulation_period)
        time_step = timedelta(minutes=temporal_spacing)
        n_time_steps = timedelta(hours=accumulation_period) // timedelta(
            minutes=temporal_spacing
        )

        return np.array(
            [
                accumulation_start_time + step * time_step
                for step in range(n_time_steps + 1)
            ]
        )

    def _calc_clearsky_irradiance_data(
        self,
        target_grid: Cube,
        irradiance_times: ndarray,
        altitude: Union[ndarray, float],
        linke_turbidity: Union[ndarray, float],
    ) -> ndarray:
        """Evaluated the gridded irradiance data over the specified period, calculated on
        the same spatial grid points as target_grid.

        Args:
            target_cube:
                Cube containing the target spatial grid on which to evaluate irradiance.
            irradiance_times:
                Datetimes at which to evaluate the irradiance data.
            altitude:
                Altitude data, specified in metres.
            linke_turbidity:
                Linke turbidity data.

        Returns:
            Gridded irradiance values evaluated over the specified times.
        """
        irradiance_data = np.zeros(
            shape=(
                irradiance_times.size,
                target_grid.coord(axis="Y").shape[0],
                target_grid.coord(axis="X").shape[0],
            ),
            dtype=np.float32,
        )

        return irradiance_data

    def _get_solar_radiation_cube(
        self,
        solar_radiation_data: ndarray,
        target_grid: Cube,
        time: datetime,
        accumulation_period: int,
        at_mean_sea_level: str,
    ) -> Cube:
        """Create a cube of accumulated clearsky solar radiation.

        Args:
            solar_radiation_data:
                Solar radiation data.
            target_grid:
                Cube containing spatial grid over which the solar radiation
                has been calculated.
            time:
                Time corresponding to the solar radiation accumulation.
            accumulation_period:
                Time window over which solar radiation has been accumulated,
                specified in hours.
            at_mean_sea_level:
                Flag denoting whether solar radiation is defined at mean-sea-level
                or at the Earth's surface. The appropriate vertical coordinate will
                be assigned accordingly.

        Returns:
            Cube containing clearsky solar radaition.
        """
        X_coord = target_grid.coord(axis="X")
        Y_coord = target_grid.coord(axis="Y")

        time_lower_bounds = np.array(
            (time - timedelta(hours=accumulation_period)).timestamp(), dtype=np.int64
        )
        time_upper_bounds = np.array(time.timestamp(), dtype=np.int64)

        time_coord = AuxCoord(
            time_upper_bounds,
            bounds=np.array([time_lower_bounds, time_upper_bounds]),
            standard_name="time",
            units=cf_units.Unit(
                "seconds since 1970-01-01 00:00:00 UTC",
                calendar=cf_units.CALENDAR_STANDARD,
            ),
        )

        # Add vertical coordinate to indicate whether solar radiation is evaluated at mean_sea_level
        # or at altitude.
        if at_mean_sea_level:
            vertical_coord = "altitude"
        else:
            vertical_coord = "height"
        Z_coord = AuxCoord(
            np.float32(0.0),
            standard_name=vertical_coord,
            units="m",
            attributes={"positive": "up"},
        )

        attrs = generate_mandatory_attributes([target_grid])

        solar_radiation_cube = Cube(
            solar_radiation_data,
            long_name=CLEARSKY_SOLAR_RADIATION_CF_NAME,
            units="W s m-2",
            dim_coords_and_dims=[(Y_coord, 0), (X_coord, 1)],
            aux_coords_and_dims=[(time_coord, None), (Z_coord, None)],
            attributes=attrs,
        )

        return solar_radiation_cube

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

        # Assuming the altitude data is on the same grid as target grid
        if isinstance(surface_altitude, Cube):
            if not spatial_coords_match([target_grid, surface_altitude]):
                raise ValueError(
                    "altitude spatial coordinates do not match target_grid"
                )
            # we will work with numpy array for calculating irradiance.
            altitude_data = surface_altitude.data
        else:
            altitude_data = surface_altitude
        # Altitude specifier is used for cf-like naming of output variable
        if np.allclose(altitude_data, 0.0):
            at_mean_sea_level = True
        else:
            at_mean_sea_level = False

        # Assuming the linke-turbidity data is on the same grid as target grid,
        if isinstance(linke_turbidity, Cube):
            if not spatial_coords_match([target_grid, linke_turbidity]):
                raise ValueError(
                    "linke_turbidity spatial coordinates do not match target_grid"
                )
            # we will work with numpy array for calculating irradiance.
            linke_turbidity_data = linke_turbidity.data
        else:
            linke_turbidity_data = linke_turbidity

        irradiance_times = self._get_irradiance_times(
            time, accumulation_period, temporal_spacing
        )

        irradiance_data = self._calc_clearsky_irradiance_data(
            target_grid, irradiance_times, altitude_data, linke_turbidity_data
        )

        # integrate the irradiance data along the time dimension to get the
        # accumulated solar irradiance.
        solar_radiation_data = np.trapz(
            irradiance_data, dx=SECONDS_IN_MINUTE * temporal_spacing, axis=0
        )

        solar_radiation_cube = self._get_solar_radiation_cube(
            solar_radiation_data,
            target_grid,
            time,
            accumulation_period,
            at_mean_sea_level,
        )

        return solar_radiation_cube
