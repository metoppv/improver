# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""module to calculate hail_size"""

from bisect import bisect_right
from typing import List, Tuple, Union

import numpy as np
from iris.cube import Cube, CubeList
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
from improver.utilities.common_input_handle import as_cubelist
from improver.utilities.cube_checker import assert_spatial_coords_match
from improver.utilities.cube_extraction import ExtractLevel
from improver.utilities.cube_manipulation import enforce_coordinate_ordering


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
    graphs from Fawbush and Miller (1953)

    If the cloud condensation level is invalid, or masked, this indicates no convection and
    the hail size will be zero.


    References
        - Hand, W., and G. Cappelluti. 2011. “A global hail climatology using the UK
          Met Office convection diagnosis procedure (CDP) and model analyses.”
          Meteorological Applications 18: 446-458. https://doi.org/10.1002/met.236
        - Fawbush, E.J., and R.C. Miller. 1953. “A method for forecasting hailstone size
          at the earth's surface.” Bulletin of the American Meteorological Society 34: 235-244.
          https://doi.org/10.1175/1520-0477-34.6.235
    """

    def __init__(self, model_id_attr: str = None):
        """Sets up Class
        Args:
            model_id_attr:
                Name of model ID attribute to be copied from source cubes to output cube
        """

        self.final_order = None
        self.model_id_attr = model_id_attr

        (self._wbzh_keys, self._hail_groups, self._updated_values) = (
            self.updated_nomogram()
        )

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

    def check_cubes(
        self,
        ccl_temperature: Cube,
        ccl_pressure: Cube,
        temperature_on_pressure: Cube,
        wet_bulb_zero_asl: Cube,
        orography: Cube,
    ) -> None:
        """Checks the size and units of input cubes and enforces the standard coord order

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
        coord_order = ["realization", "pressure"] + [
            temperature_on_pressure.coord(axis=axis).name() for axis in "yx"
        ]
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
            ccl_temperature, ccl_pressure, pressure_at_268, humidity_mixing_ratio_at_ccl
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
                Cube of cloud condensation level temperature
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

    def process(self, *cubes: Union[Cube, CubeList]) -> Cube:
        """
        Main entry point of this class

        Args:
            cubes
                air_temperature:
                    Cube of the cloud condensation level temperature
                air_pressure_at_condensation_level:
                    Cube of the cloud condensation level pressure.
                air_temperature_at_condensation_level:
                    Cube of temperature on pressure levels
                wet_bulb_freezing_level_altitude:
                    Cube of the height of the wet-bulb freezing level above sea level
                surface_altitude:
                    Cube of the orography height.
        Returns:
            Cube of hail diameter (m)
        """
        cubes = as_cubelist(*cubes)
        (
            temperature_on_pressure,
            ccl_pressure,
            ccl_temperature,
            wet_bulb_zero_height_asl,
            orography,
        ) = cubes.extract(
            [
                "air_temperature",
                "air_pressure_at_condensation_level",
                "air_temperature_at_condensation_level",
                "wet_bulb_freezing_level_altitude",
                "surface_altitude",
            ]
        )

        self.check_cubes(
            ccl_temperature,
            ccl_pressure,
            temperature_on_pressure,
            wet_bulb_zero_height_asl,
            orography,
        )
        extract_pressure = ExtractLevel(
            value_of_level=268.15, positive_correlation=True
        )
        pressure_at_268 = extract_pressure(temperature_on_pressure)

        temperature_at_268 = next(temperature_on_pressure.slices_over(["pressure"]))
        temperature_at_268.rename("temperature_of_atmosphere_at_268.15K")
        temperature_at_268.remove_coord("pressure")
        temperature = np.full_like(
            temperature_at_268.data, extract_pressure.value_of_level, dtype=np.float32
        )
        temperature = np.ma.masked_where(np.ma.getmask(pressure_at_268), temperature)
        temperature_at_268.data = temperature

        attributes = {}
        if self.model_id_attr:
            attributes[self.model_id_attr] = temperature_on_pressure.attributes[
                self.model_id_attr
            ]
        del temperature_on_pressure

        humidity_mixing_ratio_at_ccl = saturated_humidity(
            ccl_temperature.data, ccl_pressure.data
        )

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
