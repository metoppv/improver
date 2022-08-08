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

"""modules to calculate hail_size"""

import math
from operator import itemgetter
from typing import List

import iris
import numpy as np
from iris.cube import Cube

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
from improver.utilities.cube_manipulation import MergeCubes


class HailSize(BasePlugin):
    def __init__(self, model_id_attr: str = None):
        self.model_id_attr = model_id_attr
        self.ccl_temperature = None
        self.ccl_pressure = None

    def nomogram_values(self):

        """set-up dictionary of the table found in https://doi.org/10.1002/met.236 of hail sizes"""

        """formatting turned off for this graph as the formatting makes it unreadable. Might need to 
        reconsider how I'm formatting this"""

        # fmt: off
        lookup_nomogram={
            5:  [0 ,0 ,0 ,0 ,0 ,0 ,0 ,2 ,2 ,2 ,2 ,2 ,5 ,5 ,5 ,5 ,10,10,10,10 ,10 ,10 ,10 ,15 ,15 ,15 ],
            10: [0 ,0 ,0 ,0 ,0 ,2 ,2 ,2 ,5 ,5 ,5 ,10,10,15,15,15,15,15,15,20 ,20 ,20 ,20 ,25 ,25 ,25 ],
            15: [0 ,0 ,2 ,2 ,2 ,2 ,5 ,10,10,15,15,20,20,20,25,25,25,25,30,30 ,30 ,35 ,35 ,35 ,35 ,35 ],
            20: [2 ,2 ,2 ,2 ,2 ,5 ,10,15,20,20,20,25,25,30,30,35,35,40,40,40 ,40 ,45 ,45 ,45 ,45 ,50 ],
            25: [2 ,5 ,5 ,5 ,10,15,20,20,25,30,30,35,40,40,40,40,45,45,45,50 ,50 ,50 ,50 ,50 ,55 ,55 ],
            30: [5 ,5 ,5 ,10,15,20,20,25,30,35,40,40,40,45,45,50,50,55,55,60 ,60 ,60 ,60 ,65 ,65 ,65 ],
            35: [5 ,5 ,10,15,20,20,25,30,35,40,40,45,45,50,55,55,60,60,65,70 ,70 ,75 ,75 ,80 ,80 ,80 ],
            40: [5 ,5 ,10,15,20,20,25,30,35,40,40,45,50,55,60,60,65,70,75,80 ,80 ,85 ,90 ,100,100,100],
            45: [5 ,10,15,20,20,25,30,35,40,45,45,50,55,60,60,65,70,80,85,90 ,95 ,100,105,110,110,110],
            50: [5 ,10,15,20,20,25,30,35,40,45,50,50,55,60,65,75,80,85,90,100,105,110,115,120,120,120]  
        }
        # fmt: on

        return lookup_nomogram

    def temperature_of_atmosphere_at_CCL_pressure(
        self, temperature_on_pressure, pressure_ccl, temperature_ccl
    ):
        """calculates the temperature of the atmosphere at the same pressure level as the CCL"""

        temperature_slices_list = []

        for atmosphere_slices, pressure_ccl_slice in zip(
            temperature_on_pressure.slices_over(
                ["latitude", "longitude", "realization"]
            ),
            pressure_ccl.slices_over(["latitude", "longitude", "realization"]),
        ):
            index_of_closest_pressure = np.argmin(
                abs(
                    atmosphere_slices.coord("pressure").points - pressure_ccl_slice.data
                )
            )
            # cube units of temperature, pascals due to copying cube needs to be fixed
            temperature_slice_of_atmosphere_at_CCL = pressure_ccl_slice.copy(
                data=atmosphere_slices.data[index_of_closest_pressure]
            )
            temperature_slice_of_atmosphere_at_CCL.rename(
                "temperature_of_atmosphere_at_CCL"
            )

            temperature_slices_list.append(temperature_slice_of_atmosphere_at_CCL)
        atmosphere_temperature_at_CCL = MergeCubes()(temperature_slices_list)

        return atmosphere_temperature_at_CCL

    def pressure_of_atmosphere_at_268(self, temperature_on_pressure):
        """Extracts the pressure and temperature where the environment temperature drops below 
        -5 degrees (268.15K)"""

        temperature_storage = []
        pressure_storage = []

        for slice in temperature_on_pressure.slices_over(
            ["latitude", "longitude", "realization"]
        ):
            if any(slice.data < 268.15):
                for temperature in slice.slices_over("pressure"):
                    if temperature.data < 268.15:

                        pressure = temperature.copy(
                            data=temperature.coord("pressure").points[0]
                        )
                        pressure.remove_coord("pressure")
                        pressure_storage.append(pressure)

                        temperature.remove_coord("pressure")
                        temperature_storage.append(temperature)

                        break
            else:
                raise ValueError("no temperature below-5")

        temperature_at_268 = MergeCubes()(temperature_storage)
        pressure_at_268 = MergeCubes()(pressure_storage)

        return temperature_at_268, pressure_at_268

    def humidity_mixing_ratio_at_CCL(
        self, humidity_mixing_ratio_on_pressure, pressure_at_268
    ):

        """Extract the humidity mixing ratio at the CCL"""

        humidity_at_ccl = []

        for humidity, pressure in zip(
            humidity_mixing_ratio_on_pressure.slices_over(
                ["latitude", "longitude", "realization"]
            ),
            pressure_at_268.slices_over(["latitude", "longitude", "realization"]),
        ):
            index = np.argmin(abs(humidity.coord("pressure").points - pressure.data))
            pressure = humidity.coord("pressure").points[index]
            humidity_at_ccl.append(humidity.extract(iris.Constraint(pressure=pressure)))

        humidity_mixing_ratio_at_CCL = MergeCubes()(humidity_at_ccl)

        return humidity_mixing_ratio_at_CCL

    def potential_temperature_after_saturated_ascent_from_CCL(
        self,
        temperature_of_atmosphere_at_CCL,
        ccl_pressure,
        pressure_at_268,
        humidity_mixing_ratio_at_268_pressure,
        ccl_temperature,
    ):

        """Calculates point b. Raises up a saturated adiabat from point C (temperature of atmosphere at CCl) 
        to the pressure of the atmosphere at 268.15K"""

        t_dry = dry_adiabatic_temperature(
            temperature_of_atmosphere_at_CCL, ccl_pressure.data, pressure_at_268.data
        )

        t_2, _ = adjust_for_latent_heat(
            t_dry.data, humidity_mixing_ratio_at_268_pressure.data, pressure_at_268.data
        )

        temperature_after_saturated_ascent_from_ccl = ccl_temperature.copy(data=t_2)
        temperature_after_saturated_ascent_from_ccl.rename(
            "temperature_after_saturated_ascent_from_ccl"
        )

        return temperature_after_saturated_ascent_from_ccl

    def Dry_adiabatic_drop_to_ccl(
        self, ccl_pressure, temperature_at_268, pressure_at_268
    ):

        """Used to calulate c. Does a dry adiabatic descent from B to the ccl_pressure_level"""

        t_dry = dry_adiabatic_temperature(
            temperature_at_268.data, pressure_at_268.data, ccl_pressure.data
        )
        ccl_level_temperature = temperature_at_268.copy(data=t_dry)
        ccl_level_temperature.rename("potential temperature of 268K at ccl pressure")
        return ccl_level_temperature

    def get_hail_size(self, vertical, horizontal):

        """Uses the lookup_table and the vertical and horizontal indexes calculated to extract and store values from nomogram """

        lookup_table = self.nomogram_values()
        shape = np.shape(vertical)

        vertical = np.around(vertical.data / 5, decimals=0) * 5
        horizontal = np.around(horizontal.data * 2, decimals=0)

        vertical_flat = vertical.flatten(order="C")
        horizontal_flat = horizontal.flatten(order="C")

        hail_size_list = []

        for vertical, horizontal in zip(vertical_flat, horizontal_flat):

            if horizontal > len(lookup_table[vertical]):
                horizontal = len(lookup_table[vertical])

            hail_size_list.append(lookup_table[int(vertical)][int(horizontal - 1)])

        hail_size = np.reshape(hail_size_list, shape, order="C")

        return hail_size

    def make_hail_cube(self, hail_size, ccl_temperature, ccl_pressure):

        """makes a cube containing hail_data"""

        attributes = {}
        if self.model_id_attr:
            attributes[self.model_id_attr] = self.ccl_temperature.attributes[
                self.model_id_attr
            ]

        hail_size_cube = create_new_diagnostic_cube(
            name="size_of_hail_stones",
            units="mm",
            template_cube=ccl_temperature,
            data=hail_size,
            mandatory_attributes=generate_mandatory_attributes(
                [ccl_temperature, ccl_pressure]
            ),
            optional_attributes=attributes,
        )

        return hail_size_cube

    def process(
        self,
        ccl_temperature,
        ccl_pressure,
        temperature_on_pressure,
        humidity_mixing_ratio_on_pressure,
    ) -> Cube:

        atmosphere_temperature_at_CCL = self.temperature_of_atmosphere_at_CCL_pressure(
            temperature_on_pressure, ccl_pressure, ccl_temperature
        )
        temperature_at_268, pressure_at_268 = self.pressure_of_atmosphere_at_268(
            temperature_on_pressure
        )
        humidity_mixing_ratio_at_CCL_pressure = self.humidity_mixing_ratio_at_CCL(
            humidity_mixing_ratio_on_pressure, pressure_at_268
        )

        B = temperature_at_268
        c = self.Dry_adiabatic_drop_to_ccl(
            ccl_pressure, temperature_at_268, pressure_at_268
        )
        b = self.potential_temperature_after_saturated_ascent_from_CCL(
            atmosphere_temperature_at_CCL,
            ccl_pressure,
            pressure_at_268,
            humidity_mixing_ratio_at_CCL_pressure,
            ccl_temperature,
        )

        vertical = c - B
        horizontal = b - B

        hail_size = self.get_hail_size(vertical, horizontal)
        hail_cube = self.make_hail_cube(hail_size, ccl_temperature, ccl_pressure)

        return hail_cube
