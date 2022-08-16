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

"""module to calculate hail_size"""


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
)
from ..utilities.cube_checker import assert_spatial_coords_match
from improver.utilities.cube_manipulation import MergeCubes

class HailSize(BasePlugin):
    def __init__(self, model_id_attr: str = None):
        """Sets up Class
            Args:
                model_id_attr: 
                    Name of model ID attribute to be copied from source cubes to output cube
        
        """

        self.model_id_attr = model_id_attr

    def nomogram_values(self) -> np.ndarray:

        """set-up an array of a table containing possible hail sizes (mm) 
        as described in https://doi.org/10.1002/met.236."""


        lookup_nomogram = (
            [0, 0, 0, 2, 2, 5, 5, 5, 5, 5],
            [0, 0, 0, 2, 5, 5, 5, 5, 10, 10],
            [0, 0, 2, 2, 5, 5, 10, 10, 15, 15],
            [0, 0, 2, 2, 5, 10, 15, 15, 20, 20],
            [0, 0, 2, 2, 10, 15, 20, 20, 20, 20],
            [0, 2, 2, 5, 15, 20, 20, 20, 25, 25],
            [0, 2, 5, 10, 20, 20, 25, 25, 30, 30],
            [2, 2, 10, 15, 20, 25, 30, 30, 35, 35],
            [2, 5, 10, 20, 25, 30, 35, 35, 40, 40],
            [2, 5, 15, 20, 30, 35, 40, 40, 45, 45],
            [2, 5, 15, 20, 30, 40, 40, 40, 45, 50],
            [2, 10, 20, 25, 35, 40, 45, 45, 50, 50],
            [5, 10, 20, 25, 40, 40, 45, 50, 55, 55],
            [5, 15, 20, 30, 40, 45, 50, 55, 60, 60],
            [5, 15, 25, 30, 40, 45, 55, 60, 60, 65],
            [5, 15, 25, 35, 40, 50, 55, 60, 65, 75],
            [10, 15, 25, 35, 45, 50, 60, 65, 70, 80],
            [10, 15, 25, 40, 45, 55, 60, 70, 80, 85],
            [10, 15, 30, 40, 45, 55, 65, 75, 85, 90],
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            [10, 20, 30, 40, 50, 60, 70, 80, 95, 105],
            [10, 20, 35, 45, 50, 60, 75, 85, 100, 110],
            [10, 20, 35, 45, 50, 60, 75, 90, 105, 115],
            [15, 25, 35, 45, 50, 65, 80, 100, 110, 120],
            [15, 25, 35, 45, 55, 65, 80, 100, 110, 120],
            [15, 25, 35, 50, 55, 65, 80, 100, 110, 120],
        )

        return lookup_nomogram


    def variable_at_pressure(
        self, variable_on_pressure:Cube, pressure: Cube
    ) -> Cube:
        """extracts the values of a variable at a given pressure level described 
        in another cube"""

        pressure_grid=self.pressure_grid(variable_on_pressure)

        pressure_diff= abs(pressure_grid-pressure.data)
        indicies=np.nanargmin(pressure_diff,axis=0)
        
        rea,lat,long=indicies.shape
        rea,lat,long=np.ogrid[:rea,:lat,:long]

        variable=variable_on_pressure.data[rea,indicies,lat,long]
        return variable

    def pressure_grid(self,temperature_on_pressure):

        pressure_points=temperature_on_pressure.coord("pressure").points
        shape=np.shape(next(temperature_on_pressure.slices_over("pressure")))

        pressure_array=[]

        for points in pressure_points:
            pressure_array.append(np.full(shape,points))

        return pressure_array


    def humidity_mixing_ratio_at_268(
        self, humidity_mixing_ratio_on_pressure, pressure_at_268):

        """Extract the humidity mixing ratio at the 268K"""

        humidity=self.variable_at_pressure(humidity_mixing_ratio_on_pressure, pressure_at_268)

        humidity_at_ccl=pressure_at_268.copy(data=humidity)
        humidity_at_ccl.units="kg/kg"
        humidity_at_ccl.rename("humidity_mixing_ratio_at_268K")

        return humidity_at_ccl


    def pressure_at_268(self, temperature_on_pressure:Cube) -> Cube:
        """Extracts the pressure where the environment temperature drops below 
        -5 degrees (268.15K)"""

        pressure_template=next(temperature_on_pressure.slices_over(['pressure']))
        pressure_template.rename("pressure of atmosphere at 268.15K")
        pressure_template.units = temperature_on_pressure.coord("pressure").units
        pressure_template.remove_coord("pressure")

        temperature_template=next(temperature_on_pressure.slices_over(['pressure']))
        temperature_template.rename("tempeature_of_atmosphere_at_268.15K")
        temperature_template.remove_coord("pressure")

        shape=temperature_on_pressure.data.shape
        r,l,m=(shape[0],shape[2],shape[3]) #ignores pressure axis

        data = np.ma.masked_greater(temperature_on_pressure.data,268.15)
        data = np.ma.masked_invalid(data)

        indicies=np.ma.notmasked_edges(data,axis=1)[0][1].reshape(r,l,m)

        pressure_template.data=temperature_on_pressure.coord("pressure").points[indicies]

        temperature_template.data=self.variable_at_pressure(temperature_on_pressure,pressure_template)

        return pressure_template, temperature_template

    def potential_temperature_after_saturated_ascent_from_CCL(
        self,
        temperature_at_CCL,
        ccl_pressure,
        pressure_at_268,
        humidity_mixing_ratio_at_268,
        ccl_temperature,
    ):

        """Calculates point b. Raises up a saturated adiabat from point C (temperature of atmosphere at CCl) 
        to the pressure of the atmosphere at 268.15K"""

        t_dry = dry_adiabatic_temperature(
            temperature_at_CCL, ccl_pressure.data, pressure_at_268.data
        )

        t_2, _ = adjust_for_latent_heat(
            t_dry.data, humidity_mixing_ratio_at_268.data, pressure_at_268.data
        )

        temperature_after_saturated_ascent_from_ccl = ccl_temperature.copy(data=t_2)
        temperature_after_saturated_ascent_from_ccl.rename(
            "temperature_after_saturated_ascent_from_ccl"
        )

        return temperature_after_saturated_ascent_from_ccl

    def dry_adiabatic_drop_to_ccl(
        self, ccl_pressure, temperature_at_268, pressure_at_268
    ):

        """Used to calulate c. Does a dry adiabatic descent from B to the ccl_pressure"""

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

        horizontal = np.around(horizontal.data/5, decimals=0)-1  
        vertical = np.around(vertical.data*2, decimals=0)-1

        horizontal_flat = horizontal.flatten(order="C")
        vertical_flat = vertical.flatten(order="C")

        hail_size_list = []
        for vertical, horizontal in zip(vertical_flat, horizontal_flat):

            if min(horizontal,vertical)< 0:
                hail_size_list.append(0)
            else:


                if vertical > len(lookup_table)-1:
                    vertical = len(lookup_table)-1
                if horizontal > len(lookup_table[0])-1:
                    horizontal = len(lookup_table[0])-1

                hail_size_list.append(lookup_table[int(vertical)][int(horizontal)])

        hail_size = np.reshape(hail_size_list, shape, order="C")
        return hail_size

    def make_hail_cube(self, hail_size, ccl_temperature, ccl_pressure):

        """puts the data into a cube containing the hail_data"""

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

    def check_cubes(self,ccl_temperature,ccl_pressure,temperature_on_pressure,humidity_mixing_ratio_on_pressure):
        """function to check size and units of input cubes"""
        
        temp_slice=next(temperature_on_pressure.slices_over("pressure"))

        assert_spatial_coords_match([ccl_temperature,ccl_pressure,temp_slice])
        assert_spatial_coords_match([temperature_on_pressure,humidity_mixing_ratio_on_pressure])

        ccl_temperature.convert_units("K")
        ccl_pressure.convert_units("Pa")
        temperature_on_pressure.convert_units("K")
        humidity_mixing_ratio_on_pressure.convert_units("kg/kg")


    def process(
        self,
        ccl_temperature:Cube,
        ccl_pressure: Cube,
        temperature_on_pressure: Cube,
        humidity_mixing_ratio_on_pressure: Cube,
    ) -> Cube:
        """
        Args:
            ccl_temeprature:
                cube of the cloud condensation level temperature
            ccl_pressure:
                cube of the cloud condensation level pressure.
            temperature_on_pressure:
                cube of temperature on pressure levels 
            humidity_mixing_ratio_on_pressure:
                cube of humidity mixing ratio on pressure levels

        Returns:
            Cube of hail size
        """

        self.check_cubes(ccl_temperature,ccl_pressure,temperature_on_pressure,humidity_mixing_ratio_on_pressure)

        np.ma.masked_less(ccl_temperature.data,268.15)

        pressure_at_268 ,temperature_at_268= self.pressure_at_268(
            temperature_on_pressure
        )

        humidity_mixing_ratio_at_268 = self.humidity_mixing_ratio_at_268(
            humidity_mixing_ratio_on_pressure, pressure_at_268
        )

        B = temperature_at_268
        c = self.dry_adiabatic_drop_to_ccl(
            ccl_pressure, temperature_at_268, pressure_at_268
        )
        b = self.potential_temperature_after_saturated_ascent_from_CCL(
            ccl_temperature,
            ccl_pressure,
            pressure_at_268,
            humidity_mixing_ratio_at_268,
            ccl_temperature,
        )

        horizontal = c - B
        vertical = b - B

        hail_size = self.get_hail_size(vertical,horizontal)
        hail_cube = self.make_hail_cube(hail_size, ccl_temperature, ccl_pressure)

        return hail_cube
