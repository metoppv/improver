# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to contain Condensation trail formation calculations."""

from typing import Tuple, Union

import numpy as np
from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.constants import EARTH_REPSILON
from improver.generate_ancillaries.generate_svp_derivative_table import (
    SaturatedVapourPressureDerivativeTable,
    SaturatedVapourPressureTable,
)
from improver.psychrometric_calculations.psychrometric_calculations import (
    calculate_svp_in_air,
)
from improver.utilities.common_input_handle import as_cubelist


class CondensationTrailFormation(BasePlugin):
    """Plugin to calculate whether a condensation trail (contrail) will
    form based on a given set of atmospheric conditions.

    The calculations require the following data:

    - Temperature on pressure levels.
    - Relative Humidity on pressure levels.

    Alongside constants including the ratio of the molecular masses of
    water and air (EARTH_REPSILON), and defined values for the engine
    contrail factor.

    .. include:: extended_documentation/psychrometric_calculations/condensation_trails/appleman_diagram.rst

    References:
        Schrader, M.L., 1997. Calculations of aircraft contrail
        formation critical temperatures. Journal of Applied
        Meteorology, 36(12), pp.1725-1729.
    """

    temperature = None
    humidity = None
    pressure_levels = None
    engine_mixing_ratios = None
    critical_temperatures = None
    critical_intercepts = None

    def __init__(self, engine_contrail_factors: list = [3e-5, 3.4e-5, 3.9e-5]):
        """Initialises the Class

        Args:
            engine_contrail_factors (list, optional):
                List of engine contrail factors to use in the
                calculations. Defaults to [3e-5, 3.4e-5, 3.9e-5].
                These values are for Non-, Low-, and High-Bypass
                engines from Schrader (1997). The units are
                kg/kg/K.
        """

        self._engine_contrail_factors: np.ndarray = np.array(
            engine_contrail_factors, dtype=np.float32
        )

    def _calculate_engine_mixing_ratios(
        self, pressure_levels: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the mixing ratio of the atmosphere and aircraft
        exhaust (Schrader, 1997). This calculation uses
        EARTH_REPSILON, which is the ratio of the molecular
        weights of water and air on Earth.

        Returns:
            np.ndarray: The mixing ratio of the atmosphere and aircraft exhaust, provided in units: Pa/K.
        """
        return (
            pressure_levels[np.newaxis, :]
            * self._engine_contrail_factors[:, np.newaxis]
            / EARTH_REPSILON
        )

    def _find_local_vapour_pressure(self, pressure_levels: np.ndarray) -> np.ndarray:
        """
        Calculate the local vapour pressure with respect to water at the given
        pressure levels using the temperature and pressure data.

        Args:
            pressure_levels (np.ndarray): Pressure levels (Pa).

        Returns:
            np.ndarray: The localised vapour pressure at the given pressure levels (Pa).
        """
        # Pressure levels has to be reshaped to match the temperature and humidity dimensions
        pressure_levels_reshaped = np.reshape(
            pressure_levels,
            (len(pressure_levels),) + (1,) * (self.temperature.ndim - 1),
        )
        svp = calculate_svp_in_air(
            temperature=self.temperature,
            pressure=pressure_levels_reshaped,
            phase="water",
        )
        return self.relative_humidity * svp

    def _critical_temperatures_and_intercepts_for_given_contrail_factor(
        self,
        engine_mixing_ratio_for_contrail_factor: np.ndarray,
        svp_table: Cube,
        svp_derivative_table: Cube,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the critical temperatures and critical intercepts on pressure levels for a single engine contrail
        factor.

        These are calculated at each pressure level by drawing a tangent to the saturation vapour pressure curve with
        respect to water. The tangent gradient is equal to the engine mixing ratio.

        .. include:: extended_documentation/psychrometric_calculations/condensation_trails/critical_temperatures.rst

        Args:
            engine_mixing_ratio_for_contrail_factor (np.ndarray): Engine mixing ratios on pressure levels for a single
                contrail factor. Array axis is [pressure levels] (Pa/K).
            svp_table (iris.cube.Cube): Lookup table of saturation vapour pressure with respect to water (Pa).
            svp_derivative_table (iris.cube.Cube): Lookup table of the first derivative of saturation vapour pressure with
                respect to water (Pa/K).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Critical temperatures on pressure levels. Array axes are [pressure levels, latitude, longitude] (K).
                - Critical intercepts on pressure levels. Array axis is [pressure levels] (Pa).
        """
        temperature_from_svp_table = svp_table.coord("air_temperature").points

        # maximum critical temperature (at 100% relative humidity) is given by
        # the point on the SVP derivative curve that is equal to the engine mixing
        # ratio
        ind_max = np.abs(
            svp_derivative_table.data
            - engine_mixing_ratio_for_contrail_factor[:, np.newaxis]
        ).argmin(axis=1)
        tangent_gradient = svp_derivative_table.data[ind_max]
        critical_temperature_maximum = temperature_from_svp_table[ind_max]

        # tangent to SVP curve, with gradient equal to the engine mixing ratio
        critical_intercept = (
            svp_table.data[ind_max] - tangent_gradient * critical_temperature_maximum
        )
        tangent_vapour_pressure = (
            tangent_gradient[:, np.newaxis] * temperature_from_svp_table[np.newaxis, :]
            + critical_intercept[:, np.newaxis]
        )

        # minimum critical temperature (at 0% relative humidity) is given by
        # point at which the tangent crosses the line of zero vapour pressure
        ind_min = np.abs(tangent_vapour_pressure).argmin(axis=1)
        critical_temperature_minimum = temperature_from_svp_table[ind_min]

        # full range of critical temperatures for all relative humidites, 0% to 100%
        critical_temperature_all_relative_humidities = np.linspace(
            critical_temperature_minimum, critical_temperature_maximum, num=100, axis=1
        )

        # For each pressure level, construct the characteristic curve that shows
        # the variation of critical temperature with relative humidity. An example
        # curve is shown in the documentation for this method.
        #
        # The critical temperatures are then interpolated from the curve at the
        # relative humidities stored in the contrails class.

        # interpolate the local vapour pressure from the tangent
        e = np.array(
            [
                np.interp(
                    critical_temperature_all_relative_humidities[i],
                    temperature_from_svp_table,
                    tangent_vapour_pressure[i],
                )
                for i in range(critical_temperature_all_relative_humidities.shape[0])
            ]
        )

        # interpolate the saturation vapour pressure from the lookup table
        esat = np.array(
            [
                np.interp(
                    critical_temperature_all_relative_humidities[i],
                    temperature_from_svp_table,
                    svp_table.data,
                )
                for i in range(critical_temperature_all_relative_humidities.shape[0])
            ]
        )

        # interpolate the critical temperature at given relative humidities
        critical_temperature = np.array(
            [
                np.interp(
                    self.relative_humidity[i],
                    e[i] / esat[i],
                    critical_temperature_all_relative_humidities[i],
                )
                for i in range(critical_temperature_all_relative_humidities.shape[0])
            ]
        )
        return critical_temperature, critical_intercept

    def _calculate_critical_temperatures_and_intercepts(self):
        """Calculate the critical temperatures and intercepts on pressure levels for all engine contrail factors."""
        self.critical_temperatures = np.zeros(
            self.engine_mixing_ratios.shape[:2] + self.relative_humidity.shape[1:],
            dtype=np.float32,
        )
        self.critical_intercepts = np.zeros(
            self.engine_mixing_ratios.shape[:2], dtype=np.float32
        )

        svp_table = SaturatedVapourPressureTable(
            183.15, 253.15, water_only=True
        ).process()
        svp_derivative_table = SaturatedVapourPressureDerivativeTable(
            183.15, 253.15, water_only=True
        ).process()

        for i, engine_mixing_ratio_for_contrail_factor in enumerate(
            self.engine_mixing_ratios
        ):
            self.critical_temperatures[i], self.critical_intercepts[i] = (
                self._critical_temperatures_and_intercepts_for_given_contrail_factor(
                    engine_mixing_ratio_for_contrail_factor,
                    svp_table,
                    svp_derivative_table,
                )
            )

    def _calculate_contrail_persistency(
        self,
        saturated_vapour_pressure_ice: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply four conditions to determine whether non-persistent or persistent contrails will form.

        .. include:: extended_documentation/psychrometric_calculations/condensation_trails/formation_conditions.rst

        Args:
            saturated_vapour_pressure_ice: The saturated vapour pressure with respect to ice, on pressure
                levels. Pressure is the leading axis (Pa).

        Returns:
            Two boolean arrays that state whether 'non-persistent' or 'persistent' contrails will form, respectively.
            Array axes are [contrail factor, pressure level, latitude, longitude].
        """

        def reshape_and_broadcast(arr, target_shape):
            """Broadcast an input array to a target shape. Returns a view."""
            num_missing_dims = len(target_shape) - arr.ndim
            if num_missing_dims < 0:
                raise ValueError("Target shape has fewer dimensions than input array.")

            # reshape input array by adding trailing singleton dimensions
            reshaped = arr.reshape(arr.shape + (1,) * num_missing_dims)
            return np.broadcast_to(reshaped, target_shape)

        engine_mixing_ratios_reshaped = reshape_and_broadcast(
            self.engine_mixing_ratios, self.critical_temperatures.shape
        )
        critical_intercepts_reshaped = reshape_and_broadcast(
            self.critical_intercepts, self.critical_temperatures.shape
        )

        # Condition 1
        vapour_pressure_above_threshold = (
            self.local_vapour_pressure[np.newaxis]
            - engine_mixing_ratios_reshaped * self.temperature[np.newaxis]
            > critical_intercepts_reshaped
        )
        # Condition 2
        temperature_below_threshold = (
            self.temperature[np.newaxis] < self.critical_temperatures
        )
        # Condition 3
        air_is_saturated = self.local_vapour_pressure > saturated_vapour_pressure_ice
        # Condition 4
        temperature_below_freezing = self.temperature < 273.15

        nonpersistent_contrails = (
            vapour_pressure_above_threshold
            & temperature_below_threshold
            & ~(air_is_saturated & temperature_below_freezing)
        )
        persistent_contrails = (
            vapour_pressure_above_threshold
            & temperature_below_threshold
            & air_is_saturated
            & temperature_below_freezing
        )
        return nonpersistent_contrails, persistent_contrails

    def process_from_arrays(
        self,
        temperature: np.ndarray,
        relative_humidity: np.ndarray,
        pressure_levels: np.ndarray,
    ) -> np.ndarray:
        """
        Main entry point of this class for data as Numpy arrays

        Process the temperature, humidity and pressure data to calculate the
        contrails data.

        Args:
            temperature (np.ndarray): Temperature data on pressure levels where pressure is the leading axis (K).
            relative_humidity (np.ndarray): Relative humidity data on pressure levels where pressure is the leading axis (kg/kg).
            pressure_levels (np.ndarray): Pressure levels (Pa).

        Returns:
            np.ndarray: The calculated engine mixing ratios on pressure levels (Pa/K).
            This is a placeholder until the full contrail formation logic is implemented.
        """
        self.temperature = temperature
        self.relative_humidity = relative_humidity
        self.pressure_levels = pressure_levels
        self.engine_mixing_ratios = self._calculate_engine_mixing_ratios(
            self.pressure_levels
        )
        self.local_vapour_pressure = self._find_local_vapour_pressure(
            self.pressure_levels
        )
        self._calculate_critical_temperatures_and_intercepts()
        return self.engine_mixing_ratios

    def process(self, *cubes: Union[Cube, CubeList]) -> Cube:
        """
        Main entry point of this class for data as iris.Cubes

        Args:
            cubes
                air_temperature:
                    Cube of the temperature on pressure levels.
                relative_humidity:
                    Cube of the relative humidity on pressure levels.

        Returns:
            Cube of heights above sea level at which contrails will form.
        """
        cubes = as_cubelist(*cubes)
        (temperature_cube, humidity_cube) = CubeList(cubes).extract(
            ["air_temperature", "relative_humidity"]
        )
        temperature_cube.convert_units("K")
        humidity_cube.convert_units("kg kg-1")

        # Get the pressure levels from the first cube
        pressure_coord = temperature_cube.coord("pressure")
        pressure_coord.convert_units("Pa")

        # Calculate contrail formation using numpy arrays
        _ = self.process_from_arrays(
            temperature_cube.data, humidity_cube.data, pressure_coord.points
        )

        # Placeholder return to silence my type checker
        return_cube = Cube(
            self.engine_mixing_ratios,
            long_name="engine_mixing_ratios",
            units="Pa K-1",
        )

        return return_cube


class ContrailHeightExtractor(BasePlugin):
    """
    Plugin to extract contrail formation heights by category. It extracts the maximum or minimum
    height where contrail formation is Non-persistent or Persistent.
    """

    def __init__(self, use_max: bool = True):
        """
        Initialize the Class

        Args:
            use_max:
                If True, extract maximum heights; if False, extract minimum heights.
        """

        self.use_max = use_max

    def _define_max_min_height_contrail_cubes(
        self,
        formation_cube: Cube,
        height_cube: Cube,
        non_persistent_result: np.ndarray,
        persistent_result: np.ndarray,
        operation: str,
    ) -> Tuple[Cube, Cube]:
        """
        Create new cubes containing the max or min heights
        for persistent or non-persistent contrail formation.

        Args:
            formation_cube:
                Categorical cube of shape (contrail_factor, pressure_level, lat (optional), lon (optional))
            height_cube:
                Height cube of shape (pressure_level, lat (optional), lon (optional))
            non_persistent_result:
                Extracted height data for non-persistent contrails.
            persistent_result:
                Extracted height data for persistent contrails.
            operation:
                Either "max" or "min" indicating the type of extraction performed.

        Returns:
            - Cube of extracted heights for non-persistent contrails
            - Cube of extracted heights for persistent contrails
        """

        template_cube = formation_cube.slices_over("pressure").next()
        template_cube.remove_coord("pressure")
        template_cube.attributes.pop("contrail_type", None)
        template_cube.attributes.pop("contrail_type_meaning", None)

        non_persistent_cube = template_cube.copy(
            data=non_persistent_result,
        )
        non_persistent_cube.rename(f"{operation}_height_non_persistent_contrail")
        non_persistent_cube.units = height_cube.units

        persistent_cube = template_cube.copy(
            data=persistent_result,
        )
        persistent_cube.rename(f"{operation}_height_persistent_contrail")
        persistent_cube.units = height_cube.units

        return non_persistent_cube, persistent_cube

    def process(self, formation_cube: Cube, height_cube: Cube) -> Tuple[Cube, Cube]:
        """
        Main entry point for this class to extract the maximum or minimum height where contrail
        formation is categorized as Non-persistent or Persistent.

        Args:
            formation_cube:
                Categorical cube of shape (engine_contrail_factor, pressure_level, lat (optional), lon (optional))
            height_cube:
                Height cube of shape (pressure_level, lat (optional), lon (optional))

        Returns:
            - Cube of extracted height values for non-persistent contrails
            - Cube of extracted height values for persistent contrails

            Each cube has dimensions (engine_contrail_factor, lat (optional), lon (optional)).
        """

        try:
            broadcast_height = np.broadcast_to(height_cube.data, formation_cube.shape)
        except ValueError as broadcast_error:
            raise ValueError(
                f"Cannot broadcast height data of shape {height_cube.shape} to formation_cube shape {formation_cube.shape}"
            ) from broadcast_error

        if "contrail_type_meaning" not in formation_cube.attributes:
            raise ValueError(
                "formation_cube is missing the 'contrail_type_meaning' attribute."
            )
        if "contrail_type" not in formation_cube.attributes:
            raise ValueError("formation_cube is missing the 'contrail_type' attribute.")
        if len(formation_cube.attributes["contrail_type"]) != len(
            formation_cube.attributes["contrail_type_meaning"].split()
        ):
            raise ValueError(
                "The length of the 'contrail_type' and 'contrail_type_meaning' attributes do not match."
            )

        contrail_types = [
            ct.lower()
            for ct in formation_cube.attributes["contrail_type_meaning"].split()
        ]
        non_persistent_index = contrail_types.index("non-persistent")
        persistent_index = contrail_types.index("persistent")

        non_persistent_mask = (
            formation_cube.data
            == formation_cube.attributes["contrail_type"][non_persistent_index]
        )
        persistent_mask = (
            formation_cube.data
            == formation_cube.attributes["contrail_type"][persistent_index]
        )

        if self.use_max:
            non_persistent_result = np.nanmax(
                np.where(non_persistent_mask, broadcast_height, np.nan), axis=1
            )
            persistent_result = np.nanmax(
                np.where(persistent_mask, broadcast_height, np.nan), axis=1
            )
            operation = "max"
        else:
            non_persistent_result = np.nanmin(
                np.where(non_persistent_mask, broadcast_height, np.nan), axis=1
            )
            persistent_result = np.nanmin(
                np.where(persistent_mask, broadcast_height, np.nan), axis=1
            )
            operation = "min"

        non_persistent_cube, persistent_cube = (
            self._define_max_min_height_contrail_cubes(
                formation_cube,
                height_cube,
                non_persistent_result,
                persistent_result,
                operation,
            )
        )

        return non_persistent_cube, persistent_cube
