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

from bisect import bisect_right
from typing import List, Optional, Tuple

import numpy as np
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError

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
from improver.utilities.cube_checker import assert_spatial_coords_match
from improver.utilities.cube_manipulation import enforce_coordinate_ordering


class ExtractPressureLevel(BasePlugin):
    """
    Class for extracting a pressure surface from data-on-pressure-levels.
    """

    def __init__(
        self,
        value_of_pressure_level: Optional[float] = None,
        model_id_attr: Optional[str] = None,
    ):
        """Sets up Class
            Args:
                value_of_pressure_level:
                    The value of the input cube for which the pressure level is required
                model_id_attr:
                    Name of model ID attribute to be copied from source cubes to output cube
        """

        self.model_id_attr = model_id_attr
        self.value_of_pressure_level = value_of_pressure_level

    def process(self, cube: Cube, pressure: Optional[Cube] = None) -> Cube:
        """
        Main entry point.

        Args:
            cube: Variable on pressure levels from which a pressure slice is required
            pressure: A pre-defined pressure slice. If provided, this is used to return
                the variable on this 2D surface, otherwise, a pressure surface is
                calculated for the point where the variable equals self.value_at_pressure_level

        Returns:
            iris.cube.Cube: Either a pressure surface where the variable equals a specific value
                or a variable on such a pressure surface
        """
        if pressure:
            result = self.extract_level_at_pressure(cube, pressure)
        else:
            result = self.extract_pressure_at_value(cube)
        return result

    def variable_at_pressure(
        self, variable_on_pressure: Cube, pressure: Cube
    ) -> np.ndarray:
        """Extracts the values from variable_on_pressure cube at pressure
        levels described by the pressure cube.

        Args:
            variable_on_pressure:
                Cube of some variable with pressure levels
            pressure:
                Cube of pressure values
        Returns:
            An n dimensional array, with the same dimensions as the pressure cube,
            of values for the variable extracted at the pressure levels described
            by the pressure cube.
        """

        pressure_grid = self.pressure_grid(variable_on_pressure)

        try:
            press_slices = pressure.slices_over("realization")
            var_slices = variable_on_pressure.slices_over("realization")
        except CoordinateNotFoundError:
            press_slices = [pressure]
            var_slices = [variable_on_pressure]
            grid_slices = [pressure_grid]
            variable = None
        else:
            grid_slices = pressure_grid[:]
            variable = []

        counter = 0
        for press, var, grid in zip(press_slices, var_slices, grid_slices):
            counter += 1
            pressure_diff = abs(grid - press.data)
            indices = np.nanargmin(pressure_diff, axis=0)
            lat, long = indices.shape
            lat, long = np.ogrid[:lat, :long]
            if variable is None:
                variable = var.data[indices, lat, long]
            else:
                variable.append(var.data[indices, lat, long])

        variable_cube = pressure.copy(data=variable)
        return variable_cube.data

    @staticmethod
    def pressure_grid(variable_on_pressure: Cube) -> np.ndarray:
        """Creates a pressure grid of the same shape as variable_on_pressure cube.
        It is populated at every grid square and for every realization with
        a column of all pressure levels taken from variable_on_pressure's pressure coordinate

        Args:
            variable_on_pressure:
                Cube of some variable with pressure levels
        Returns:
            An n dimensional array with the same dimensions as variable_on_pressure containing,
            at every grid square and for every realization, a column of all pressure levels
            taken from variable_on_pressure's pressure coordinate
        """

        required_shape = variable_on_pressure.shape
        pressure_points = variable_on_pressure.coord("pressure").points
        (pressure_axis,) = variable_on_pressure.coord_dims("pressure")
        pressure_shape = np.ones_like(required_shape)
        pressure_shape[pressure_axis] = required_shape[pressure_axis]
        pressure_array = np.broadcast_to(
            pressure_points.reshape(pressure_shape), required_shape
        )
        return pressure_array

    @staticmethod
    def fill_invalid(cube: Cube):
        """Populate any invalid values with the maximum in that column on the assumption
        that missing values fall below the model's surface."""
        (pressure_axis,) = cube.coord_dims("pressure")
        cube_shape = np.array(cube.shape)
        max_shape = cube_shape.copy()
        max_shape[pressure_axis] = 1
        max_values = np.nanmax(cube.data, axis=pressure_axis).reshape(max_shape)
        cube.data = np.where(np.isnan(cube.data), max_values, cube.data)

    def extract_pressure_at_value(self, temperature_on_pressure: Cube) -> Cube:
        """Extracts the pressure level where the environment
        temperature first drops below -5 Celsius (268.15K) starting at a pressure value
        near the surface and ascending in altitude from there. It also produces
        the environment temperature at that pressure value

        Args:
            temperature_on_pressure:
                A cube of temperature on pressure levels
        Returns:
            A cube of the environment pressure at 268.15K
        """
        from stratify import interpolate

        self.fill_invalid(temperature_on_pressure)

        try:
            slicer = temperature_on_pressure.slices_over((["realization"]))
        except CoordinateNotFoundError:
            slicer = [temperature_on_pressure]
            one_slice = temperature_on_pressure
            has_r_coord = False
        else:
            one_slice = temperature_on_pressure.slices_over((["realization"])).next()
            has_r_coord = True
        p_grid = self.pressure_grid(one_slice).astype(np.float32)
        (pressure_axis,) = one_slice.coord_dims("pressure")
        result_data = np.empty_like(
            temperature_on_pressure.slices_over((["pressure"])).next().data
        )
        for i, zyx_slice in enumerate(slicer):
            interp_data = interpolate(
                np.array([self.value_of_pressure_level], dtype=np.float32),
                zyx_slice.data.data,
                p_grid,
                axis=pressure_axis,
            ).squeeze(axis=pressure_axis)
            if has_r_coord:
                result_data[i] = interp_data
            else:
                result_data = interp_data
        result_data = np.ma.masked_invalid(result_data)

        pressure_cube = next(temperature_on_pressure.slices_over(["pressure"])).copy(
            result_data
        )
        pressure_cube.rename(
            "pressure_of_atmosphere_at_"
            f"{self.value_of_pressure_level}{temperature_on_pressure.units}"
        )
        pressure_cube.units = temperature_on_pressure.coord("pressure").units
        pressure_cube.remove_coord("pressure")
        return pressure_cube

    def extract_level_at_pressure(self, cube: Cube, pressure: Cube) -> Cube:
        """Extract relative humidity at pressure of the environment at 268.15K

        Args:
            cube:
                Cube of data values on pressure levels
            pressure:
                Cube of pressure values to extract data at
        Returns:
            A cube of data values at the pressure specified
        """

        data_on_pressure = self.variable_at_pressure(cube, pressure)
        cube_on_pressure = pressure.copy(data=data_on_pressure)
        divider = pressure.name().index("_at_")
        cube_on_pressure.rename(f"{cube.name()}{pressure.name()[divider:]}")
        cube_on_pressure.units = cube.units
        if self.model_id_attr:
            cube_on_pressure.attributes[self.model_id_attr] = cube.attributes[
                self.model_id_attr
            ]

        return cube_on_pressure


class HailSize(BasePlugin):
    """Plugin to calculate the diameter of the hail stones from input cubes
    cloud condensation level (ccl) temperature, cloud condensation level pressure,
    temperature on pressure levels, the height of the wet bulb freezing level
    above sea level and orography.

    From these, the values for three other cubes are calculated:
        - Temperature of the environment at 268.15K (-5 Celsius) and the
          pressure level where this occurs.
        - Temperature after a saturated ascent from ccl pressure to the
          pressure of the environment at 268.15K (-5 Celsius).
        - Temperature after a dry adiabatic descent from the pressure of the
          environment at 268.15K (-5 Celsius) to the ccl pressure.

    From these, two indexes are calculated as:
        - Temperature after a dry adiabatic descent - the temperature of the
          atmosphere at 268.15K
        - Temperature after a saturated ascent - the temperature of the
          atmosphere at 268.15K

    These indexes can then be used to extract values of hail size depending on
    the wet bulb freezing altitude. The wet bulb freezing altitude is calculated
    by subtracting the orography from the wet bulb freezing altitude above sea level.

    If the wet bulb freezing altitude is between 3350m and 4400m then the indexes are used
    to extract an initial hail value from the first table. A second table
    is then accessed to reduce the hail size. The second table is stored as a dictionary, with the
    key being the wet bulb freezing altitude and each column in the associated
    arrays referring to the previously calculated hail diameter being less than a pre-defined
    value. An updated hail_size is then extracted and stored.

    If the wet_bulb_freezing_altitude is greater than 4400m then the hail size is set to
    0 and if the wet bulb_freezing_altitude is less than 3350m then the originally calculated
    hail size is not altered.

    Both tables are taken from Hand and Cappelluti (2011) which are a tabular versions of the
    graphs from Fawbush and Miller(1953)


    References
        - Hand, W., and G. Cappelluti. 2011. “A global hail climatology using the UK
          Met Office convection diagnosis procedure (CDP) and model analyses.”
          Meteorological Applications 18: 446-458. doi:https://doi.org/10.1002/met.236
        - Fawbush, E.J., and R.C. Miller. 1953. “A method for forecasting hailstone size
          at the earth's surface.” Bulletin of the American Meteorological Society 34: 235-244.
          doi: https://doi.org/10.1175/1520-0477-34.6.235
    """

    def __init__(self, model_id_attr: str = None):
        """Sets up Class
            Args:
                model_id_attr:
                    Name of model ID attribute to be copied from source cubes to output cube
        """

        self.final_order = None
        self.model_id_attr = model_id_attr

        (
            self._wbzh_keys,
            self._hail_groups,
            self._updated_values,
        ) = self.updated_nomogram()

    @staticmethod
    def nomogram_values() -> np.ndarray:

        """Sets-up an array of a table containing possible diameter of hail stones(mm).
        It is a transposed version of the table in Hand and Cappelluti (2011).

        The axes of the table are as follows:

            - Horizontal axis is calculated from two values: the temperature after a
              dry adiabatic descent from the pressure of atmosphere at 268.15K to the
              cloud condensation level pressure and the temperature of the atmosphere
              at 268.15K. Each column represents a value calculated as the temperature
              after the dry adiabatic descent minus the temperature of the atmosphere
              at 268.15K rounded to the nearest 0.5K.
            - The vertical axis is also calculated from two values: the temperature after
              a saturated ascent from the ccl pressure to the pressure of environment at
              268.15K and the temperature of the atmosphere at 268.25K.
              Each row is represented by a value calculated as the temperature after
              the saturated ascent minus the temperature of the atmosphere at 268.15K
              rounded to the nearest 5K.
        """

        lookup_nomogram = np.array(
            [
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
            ],
            np.int8,
        )

        return lookup_nomogram

    @staticmethod
    def updated_nomogram() -> Tuple[List, List, np.array]:

        """Sets up a dictionary of updated hail diameter values (mm).

        The dictionary keys are the height of the wet bulb freezing level (m) where,
        when accessing at some height value, it should be rounded to the nearest lower value
        (e.g. 3549m should access 3350m key).

        Each key has an associated list in which each element is a new hail diameter based on
        the original hail size that was calculated from nomogram_values table.
        Specifically each column associated hail size (mm) is [<5,<10,<20,<25,<50,<75,<100,<125].
        The largest possible value where the equality still holds should be used.

        If the wet bulb freezing height is less than 3350m then the original hail size is used.
        If the wet bulb freezing height is greater than 4400m then all hail sizes are set to 0.
        """
        lookup_dict = {
            3350: [0, 5, 10, 15, 25, 50, 65, 75],
            3550: [0, 0, 5, 10, 20, 20, 25, 30],
            3750: [0, 0, 0, 5, 10, 15, 15, 15],
            3950: [0, 0, 0, 0, 5, 10, 10, 10],
            4150: [0, 0, 0, 0, 0, 0, 5, 5],
            4400: [0, 0, 0, 0, 0, 0, 0, 0],
        }
        hail_groups = [5, 10, 20, 25, 50, 75, 100, 125]

        return (
            list(lookup_dict.keys()),
            hail_groups,
            np.array(list(lookup_dict.values())),
        )

    def check_cubes(self, ccl_temperature: Cube, ccl_pressure: Cube, temperature_on_pressure: Cube,
                    wet_bulb_zero_asl: Cube, orography: Cube) -> None:
        """Checks the size and units of input cubes

            Args:
                ccl_temperature:
                    Cube of cloud condensation level temperature
                ccl_pressure:
                    Cube of cloud condensation level pressure
                temperature_on_pressure:
                    Cube of environment temperature on pressure levels
                wet_bulb_zero_asl:
                    Cube of the height of the wet bulb freezing level above sea level
                orography:
                    Cube of the orography height.
        """
        coord_order = ["realization", "pressure", "longitude", "latitude"]
        self.final_order = [c.name() for c in wet_bulb_zero_asl.dim_coords]
        for cube in [
            ccl_temperature,
            ccl_pressure,
            temperature_on_pressure,
            wet_bulb_zero_asl,
            orography,
        ]:
            enforce_coordinate_ordering(cube, coord_order)

        temp_slice = next(temperature_on_pressure.slices_over("pressure"))
        try:
            wb_slice = next(wet_bulb_zero_asl.slices_over("realization"))
        except CoordinateNotFoundError:
            wb_slice = wet_bulb_zero_asl
        assert_spatial_coords_match([wb_slice, orography])
        assert_spatial_coords_match(
            [ccl_temperature, ccl_pressure, temp_slice, wet_bulb_zero_asl]
        )

        ccl_temperature.convert_units("K")
        ccl_pressure.convert_units("Pa")
        temperature_on_pressure.convert_units("K")
        wet_bulb_zero_asl.convert_units("m")
        orography.convert_units("m")

    @staticmethod
    def temperature_after_saturated_ascent_from_ccl(
        ccl_temperature: Cube,
        ccl_pressure: Cube,
        pressure_at_268: Cube,
        humidity_mixing_ratio_at_ccl: np.array,
    ) -> np.ndarray:
        """Calculates the temperature after a saturated ascent
        from the cloud condensation level to the pressure of the atmosphere at 268.15K

        Args:
            ccl_temperature:
                Cube of cloud condensation level temperature
            ccl_pressure:
                Cube of cloud condensation level pressure
            pressure_at_268:
                Cube of the pressure of the environment at 268.15K
            humidity_mixing_ratio_at_ccl:
                Array of humidity mixing ratio at the pressure of the environment at the CCL
        Returns:
            Cube of temperature after the saturated ascent
        """

        t_dry = dry_adiabatic_temperature(
            ccl_temperature.data, ccl_pressure.data, pressure_at_268.data
        )
        t_2, _ = adjust_for_latent_heat(
            t_dry, humidity_mixing_ratio_at_ccl, pressure_at_268.data
        )
        return t_2

    @staticmethod
    def dry_adiabatic_descent_to_ccl(
        ccl_pressure: Cube, temperature_at_268: Cube, pressure_at_268: Cube
    ) -> np.ndarray:
        """Calculates the temperature due to a dry adiabatic descent from the
        pressure of the environment at 268.15K to the cloud condensation level
        pressure.

        Args:
            ccl_pressure:
                Cube of cloud condensation level pressure
            temperature_at_268:
                Cube of the temperature of the environment at 268.15K
            pressure_at_268:
                Cube of the pressure of the environment at 268.15K
        Returns:
            Cube of temperature after the dry adiabatic descent
        """

        t_dry = dry_adiabatic_temperature(
            temperature_at_268.data, pressure_at_268.data, ccl_pressure.data
        )
        return t_dry

    def get_hail_size(
        self, vertical: np.ndarray, horizontal: np.ndarray, wet_bulb_zero: np.ndarray
    ) -> np.ndarray:
        """Uses the lookup_table and the vertical and horizontal indexes calculated
        to extract and store values from the lookup nomogram.

        The hail size will be set to 0 if
            1) there are masked data points,
            2) vertical or horizontal values are negative,
            3) the wet bulb freezing altitude is greater that 4400m.

        If the wet bulb freezing altitude is greater that 3300m then the hail_size is reduced.

        Args:
            vertical:
                An n dimensional array containing the values used to calculate the vertical indexes
            horizontal:
                An n dimensional array containing the values used to calculate the horizontal
                indexes
            wet_bulb_zero:
                An n dimensional array containing the height of the wet bulb freezing level
        Returns:
            an n dimension array of values for the diameter of hail (mm)
        """

        lookup_table = self.nomogram_values()

        # Rounds the calculated horizontal value to the nearest 5 which is
        # then turned into a relevant index for accessing the appropriate column.
        # Rounds the calculated vertical values to the nearest 0.5 which is then
        # turned into a relevant index for accessing the appropriate row.
        horizontal_rounded = np.around(horizontal / 5, decimals=0) - 1
        vertical_rounded = np.around(vertical * 2, decimals=0)

        # clips index values to not be longer than the table
        vertical_clipped = np.clip(vertical_rounded, None, len(lookup_table) - 1)
        horizontal_clipped = np.clip(horizontal_rounded, None, len(lookup_table[0]) - 1)

        vertical_clipped = np.ma.where(
            (vertical_rounded >= 0) & (horizontal_rounded >= 0), vertical_clipped, 0
        ).filled(0)
        horizontal_clipped = np.ma.where(
            (vertical_rounded >= 0) & (horizontal_rounded >= 0), horizontal_clipped, 0
        ).filled(0)

        hail_size = lookup_table[
            vertical_clipped.astype(int), horizontal_clipped.astype(int)
        ]
        hail_size = np.where(
            wet_bulb_zero >= 3300,
            self.updated_hail_size(hail_size, wet_bulb_zero),
            hail_size,
        )
        return hail_size

    def updated_hail_size(
        self, hail_size: np.array, wet_bulb_height: np.array
    ) -> np.array:
        """Uses the updated_nomogram values dictionary to access an updated hail size
        based on the original predicted hail size and a wet bulb freezing height.

        Args:
            hail_size:
                Integers of hail diameter value taken from the original nomogram
            wet_bulb_height:
                Floats of the height of the wet bulb freezing level
        Returns:
            An updated value for the hail diameter (mm)
        """

        vectorised = np.vectorize(lambda n: bisect_right(self._wbzh_keys, n))
        height_index = np.array(vectorised(wet_bulb_height) - 1).astype(int)

        vectorised = np.vectorize(lambda n: bisect_right(self._hail_groups, n))
        hail_index = vectorised(hail_size)

        updated_hail_size = self._updated_values[height_index, hail_index]

        return np.int8(updated_hail_size)

    def hail_size_data(
        self,
        temperature_at_268: Cube,
        pressure_at_268: Cube,
        ccl_pressure: Cube,
        ccl_temperature: Cube,
        humidity_mixing_ratio_at_ccl: np.array,
        wet_bulb_zero: Cube,
    ) -> np.ndarray:
        """Gets temperature of environment at 268.15K, temperature after a dry adiabatic descent
        from the pressure of air at 268.15K to ccl pressure and the temperature
        after a saturated ascent from ccl pressure to the pressure of air at 268.15K.
        From these values it calculates vertical and horizontal indices. It also masks
        data where the ccl_temperature is below 268.15K.

        Args:
            temperature_at_268:
                Cube of the temperature of the environment at 268.15K
            pressure_at_268:
                Cube of the pressure of the environment at 268.15K
            ccl_pressure:
                Cube of cloud condensation level pressure
            ccl_temperature:
                Cube of cloud condensation level pressure
            humidity_mixing_ratio_at_ccl:
                Array of humidity mixing ratio at the pressure of the environment at the CCL
            wet_bulb_zero:
                Cube of the height of the wet-bulb freezing level
        Returns:
            An n dimensional array of diameter of hail stones (m)
        """

        # temperature_at_268 is big-B in Hand (2011).
        # ccl_temperature is big-C in Hand (2011).
        # temp_dry is little-c in Hand (2011).
        temp_dry = self.dry_adiabatic_descent_to_ccl(
            ccl_pressure, temperature_at_268, pressure_at_268
        )

        # temp_saturated_ascent is little-b in Hand (2011).
        temp_saturated_ascent = self.temperature_after_saturated_ascent_from_ccl(
            ccl_temperature,
            ccl_pressure,
            pressure_at_268,
            humidity_mixing_ratio_at_ccl,
        )

        # horizontal is c - B in Hand (2011).
        horizontal = temp_dry.data - temperature_at_268.data
        # vertical is b - B in Hand (2011).
        vertical = temp_saturated_ascent.data - temperature_at_268.data

        temperature_mask = np.ma.masked_less(ccl_temperature.data, 268.15)
        vertical_masked = np.ma.masked_where(np.ma.getmask(temperature_mask), vertical)
        horizontal_masked = np.ma.masked_where(
            np.ma.getmask(temperature_mask), horizontal
        )

        hail_size = self.get_hail_size(
            vertical_masked, horizontal_masked, wet_bulb_zero.data
        )
        hail_size = hail_size / 1000
        hail_size = hail_size.astype("float32")

        return hail_size

    @staticmethod
    def make_hail_cube(
        hail_size: np.ndarray,
        ccl_temperature: Cube,
        ccl_pressure: Cube,
        attributes: dict,
    ) -> Cube:
        """Puts the hail data into a cube with appropriate metadata

        Args:
            hail_size:
                An n dimensional array of the diameter of hail stones (m)
            ccl_temperature:
                Cube of cloud condensation level pressure
            ccl_pressure:
                Cube of cloud condensation level pressure
            attributes:
                Dictionary of attributes for the new cube

        Returns:
            A cube of the diameter of hail stones (m)
        """

        hail_size_cube = create_new_diagnostic_cube(
            name="diameter_of_hail_stones",
            units="m",
            template_cube=ccl_temperature,
            data=hail_size,
            mandatory_attributes=generate_mandatory_attributes(
                [ccl_temperature, ccl_pressure]
            ),
            optional_attributes=attributes,
        )
        return hail_size_cube

    def process(self, ccl_temperature: Cube, ccl_pressure: Cube, temperature_on_pressure: Cube,
                wet_bulb_zero_height_asl: Cube, orography: Cube) -> Cube:
        """
        Main entry point of this class

        Args:
            ccl_temperature:
                Cube of the cloud condensation level temperature
            ccl_pressure:
                Cube of the cloud condensation level pressure.
            temperature_on_pressure:
                Cube of temperature on pressure levels
            wet_bulb_zero_height_asl:
                Cube of the height of the wet-bulb freezing level above sea level
            orography:
                Cube of the orography height.
        Returns:
            Cube of hail diameter (m)
        """

        self.check_cubes(ccl_temperature, ccl_pressure, temperature_on_pressure,
                         wet_bulb_zero_height_asl, orography)
        extract_pressure = ExtractPressureLevel(value_of_pressure_level=268.15)
        pressure_at_268 = extract_pressure(temperature_on_pressure)

        temperature_at_268 = next(temperature_on_pressure.slices_over(["pressure"]))
        temperature_at_268.rename("temperature_of_atmosphere_at_268.15K")
        temperature_at_268.remove_coord("pressure")
        temperature = np.full_like(
            temperature_at_268.data,
            extract_pressure.value_of_pressure_level,
            dtype=np.float32,
        )
        temperature = np.ma.masked_where(np.ma.getmask(pressure_at_268), temperature)
        temperature_at_268.data = temperature

        attributes = {}
        if self.model_id_attr:
            attributes[self.model_id_attr] = temperature_on_pressure.attributes[
                self.model_id_attr
            ]
        del temperature_on_pressure

        humidity_mixing_ratio_at_ccl = saturated_humidity(ccl_temperature.data, ccl_pressure.data)

        wet_bulb_zero_height = wet_bulb_zero_height_asl - orography

        hail_size = self.hail_size_data(
            temperature_at_268,
            pressure_at_268,
            ccl_pressure,
            ccl_temperature,
            humidity_mixing_ratio_at_ccl,
            wet_bulb_zero_height,
        )

        hail_cube = self.make_hail_cube(
            hail_size, ccl_temperature, ccl_pressure, attributes
        )
        enforce_coordinate_ordering(hail_cube, self.final_order)
        return hail_cube