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
"""Module containing the CloudTopTemperature plugin"""
import numpy as np
from iris.cube import Cube
from numpy import ndarray

from improver import BasePlugin
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.psychrometric_calculations.psychrometric_calculations import (
    adjust_for_latent_heat,
    dry_adiabatic_temperature,
    saturated_humidity,
)
from improver.utilities.cube_checker import spatial_coords_match


class CloudTopTemperature(BasePlugin):
    """Plugin to calculate the convective cloud top temperature from the
    cloud condensation level temperature and pressure, and temperature
    on pressure levels data using saturated ascent.
    The temperature is that of the parcel after saturated ascent at the last pressure level
    where the parcel is buoyant. The interpolation required to get closer is deemed expensive.
    If the cloud top temperature is less than 4K smaller than the cloud condensation level,
    the cloud top temperature is masked.
    """

    def __init__(self, model_id_attr: str = None):
        """
        Set up class

        Args:
            model_id_attr:
                Name of model ID attribute to be copied from source cubes to output cube
        """
        self.model_id_attr = model_id_attr
        self.ccl = Cube(None)
        self.temperature = Cube(None)
        self.minimum_t_diff = 4

    def _calculate_cct(self) -> ndarray:
        """
        Ascends through the pressure levels (decreasing pressure) calculating the saturated
        ascent from the cloud condensation level until a level is reached where the
        atmosphere profile is warmer than the ascending parcel, signifying that buoyancy
        is negative and cloud growth has ceased.
        CCT is the saturated adiabatic temperature at the last atmosphere pressure level where
        the profile is buoyant.
        Temperature data are in Kelvin, Pressure data are in pascals, humidity data are in kg/kg.
        """
        p_coord = self.temperature.coord("pressure")
        t_data = [t.data for t in self.temperature.slices_over("pressure")]
        cct = np.ma.masked_array(self.ccl.data.copy())
        q_at_ccl = saturated_humidity(
            self.ccl.data, self.ccl.coord("air_pressure").points
        )
        ccl_with_mask = np.ma.where(True, self.ccl.data, False)
        for p, t in zip(p_coord.points, t_data):
            t_dry = dry_adiabatic_temperature(
                self.ccl.data, self.ccl.coord("air_pressure").points, p
            )
            t_2, _ = adjust_for_latent_heat(t_dry, q_at_ccl, p)
            # Mask out points where parcel temperature, t_2, is less than atmosphere temperature, t,
            # but only after the parcel pressure, p, becomes lower than the cloud base pressure.
            ccl_with_mask = np.ma.masked_where(
                (t_2 < t) & (p < self.ccl.coord("air_pressure").points), ccl_with_mask,
            )
            cct[~ccl_with_mask.mask] = t_2[~ccl_with_mask.mask]

        cct = np.ma.masked_where(self.ccl.data - cct < self.minimum_t_diff, cct)

        return cct

    def _make_cct_cube(self, data: ndarray) -> Cube:
        """Puts the data array into a CF-compliant cube"""
        attributes = {}
        if self.model_id_attr:
            attributes[self.model_id_attr] = self.ccl.attributes[self.model_id_attr]
        cube = create_new_diagnostic_cube(
            "temperature_at_convective_cloud_top",
            "K",
            self.ccl,
            mandatory_attributes=generate_mandatory_attributes(
                [self.ccl, self.temperature]
            ),
            optional_attributes=attributes,
            data=data,
        )
        return cube

    def process(self, ccl: Cube, temperature: Cube) -> Cube:
        """

        Args:
            ccl:
                cloud condensation level
            temperature:
                temperature on pressure levels

        Returns:
            Cube of cloud top temperature
        """
        self.ccl = ccl
        self.temperature = temperature
        spatial_coords_match([self.ccl, self.temperature])
        self.temperature.convert_units("K")
        self.ccl.convert_units("K")
        self.ccl.coord("air_pressure").convert_units("Pa")
        cct = self._make_cct_cube(self._calculate_cct())
        return cct
