# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to contain Condensation trail formation calculations."""

from typing import Tuple, Union

import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.categorical.utilities import categorical_attributes
from improver.constants import EARTH_REPSILON
from improver.generate_ancillaries.generate_svp_derivative_table import (
    SaturatedVapourPressureDerivativeTable,
    SaturatedVapourPressureTable,
)
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.psychrometric_calculations.psychrometric_calculations import (
    calculate_svp_in_air,
)
from improver.utilities.common_input_handle import as_cubelist
from improver.utilities.cube_manipulation import (
    add_coordinate_to_cube,
    get_dim_coord_names,
)


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
    nonpersistent_contrails = None
    persistent_contrails = None

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

    def _calculate_critical_temperatures_and_intercepts(self) -> None:
        """Calculate the critical temperatures and intercepts on pressure levels for all engine contrail factors."""
        self.critical_temperatures = np.zeros(
            self.engine_mixing_ratios.shape[:2] + self.relative_humidity.shape[1:],
            dtype=np.float32,
        )
        self.critical_intercepts = np.zeros(
            self.engine_mixing_ratios.shape[:2], dtype=np.float32
        )

        svp_table = SaturatedVapourPressureTable(water_only=True).process()
        svp_derivative_table = SaturatedVapourPressureDerivativeTable(
            water_only=True
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

    def _calculate_contrail_persistency(self) -> None:
        """
        Apply four conditions to determine whether non-persistent or persistent contrails will form.

        .. include:: extended_documentation/psychrometric_calculations/condensation_trails/formation_conditions.rst
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

        # saturated vapour pressure with respect to ice, on pressure levels
        svp_table = SaturatedVapourPressureTable(ice_only=True).process()
        saturated_vapour_pressure_ice = np.interp(
            self.temperature, svp_table.coord("air_temperature").points, svp_table.data
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

        # Boolean arrays that are true when the specific contrail type will form.
        # Array axes are [contrail factor, pressure level, latitude, longitude].
        self.nonpersistent_contrails = (
            vapour_pressure_above_threshold
            & temperature_below_threshold
            & ~(air_is_saturated & temperature_below_freezing)
        )
        self.persistent_contrails = (
            vapour_pressure_above_threshold
            & temperature_below_threshold
            & air_is_saturated
            & temperature_below_freezing
        )

    def _boolean_to_categorical(self) -> np.ndarray:
        """
        Combine two boolean arrays of contrail persistency into a single categorical array of contrail formation.

        Returns:
            Array of categorical (integer) data, where 0 = no contrails, 1 = non-persistent contrails and 2 = persistent
            contrails.
        """
        categorical = np.where(
            self.nonpersistent_contrails & ~self.persistent_contrails, 1, 0
        )
        categorical = np.where(
            ~self.nonpersistent_contrails & self.persistent_contrails, 2, categorical
        )
        return categorical

    def _create_contrail_formation_cube(
        self, categorical_data: np.ndarray, template_cube: Cube
    ) -> Cube:
        """
        Create a contrail formation cube, populated with categorical data.

        Args:
            categorical_data: Categorical (integer) data of contrail formation. Leading axes are [contrail factor,
                pressure level].
            template_cube: Cube from which to derive dimensions, coordinates and mandatory attributes.

        Returns:
            Categorical cube of contrail formation, where 0 = no contrails, 1 = non-persistent contrails and
            2 = persistent contrails. Has the same shape as categorical_data.
        """
        contrail_factor_coord = DimCoord(
            points=self._engine_contrail_factors, var_name="engine_contrail_factor"
        )
        template_cube = add_coordinate_to_cube(
            template_cube, new_coord=contrail_factor_coord
        )
        mandatory_attributes = generate_mandatory_attributes([template_cube])

        decision_tree = {
            "0": {"leaf": "None"},
            "1": {"leaf": "Non-persistent"},
            "2": {"leaf": "Persistent"},
        }
        optional_attributes = categorical_attributes(decision_tree, "contrail_type")

        return create_new_diagnostic_cube(
            name="contrails_formation",
            units="1",
            template_cube=template_cube,
            mandatory_attributes=mandatory_attributes,
            optional_attributes=optional_attributes,
            data=categorical_data,
            dtype=np.uint8,
        )

    def process_from_arrays(
        self,
        temperature: np.ndarray,
        relative_humidity: np.ndarray,
        pressure_levels: np.ndarray,
    ) -> np.ndarray:
        """
        Main entry point of this class for data as Numpy arrays.

        Process the temperature, humidity and pressure data to calculate the
        contrails data.

        Args:
            temperature (np.ndarray): Temperature data on pressure levels where pressure is the leading axis (K).
            relative_humidity (np.ndarray): Relative humidity data on pressure levels where pressure is the leading axis (kg/kg).
            pressure_levels (np.ndarray): Pressure levels (Pa).

        Returns:
            Categorical (integer) array of contrail formation

            - 0 = no contrails
            - 1 = non-persistent contrails
            - 2 = persistent contrails

            Array axes are [contrail factor, pressure level, latitude, longitude], where latitude and longitude are
            only included if present in the temperature and relative humidity input arrays.
        """
        arrays = (temperature, relative_humidity, pressure_levels)
        if arrays[2].ndim != 1:
            raise ValueError(f"Expected 1D pressure array, got {arrays[2].ndim}D.")
        if arrays[0].shape != arrays[1].shape:
            raise ValueError(
                f"Temperature and relative humidity arrays must have same shape:"
                f"  {arrays[0].shape}\n  {arrays[1].shape}"
            )
        if arrays[0].shape[0] != arrays[2].size or arrays[1].shape[0] != arrays[2].size:
            raise ValueError(
                f"Leading axes of arrays must match:"
                f"  {arrays[0].shape}\n  {arrays[1].shape}\n  {arrays[2].shape}"
            )

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
        self._calculate_contrail_persistency()
        return self._boolean_to_categorical()

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
            Categorical (integer) cube of contrail formation

            - 0 = no contrails
            - 1 = non-persistent contrails
            - 2 = persistent contrails

            Cube dimensions are [contrail factor, pressure level, latitude, longitude], where latitude and longitude are
            only included if present in the input cubes.
        """
        # Extract input cubes
        cubes = as_cubelist(*cubes)
        names_to_extract = ["air_temperature", "relative_humidity"]
        if len(cubes) != len(names_to_extract):
            raise ValueError(
                f"Expected {len(names_to_extract)} cubes, got {len(cubes)}."
            )
        try:
            (temperature_cube, humidity_cube) = CubeList(cubes).extract(
                names_to_extract
            )
        except Exception as e:
            raise ValueError(
                f"Could not extract names '{names_to_extract}' from cubelist."
            ) from e

        # Check cube dimensions are equal
        if (
            get_dim_coord_names(temperature_cube) != get_dim_coord_names(humidity_cube)
            or temperature_cube.shape != humidity_cube.shape
        ):
            raise ValueError(
                f"Cube dimensional coordinates must match:"
                f"  {temperature_cube.summary(True, 25)}"
                f"  {humidity_cube.summary(True, 25)}"
            )

        temperature_cube.convert_units("K")
        humidity_cube.convert_units("kg kg-1")

        # Get pressure levels
        pressure_coord = temperature_cube.coord("pressure")
        pressure_coord.convert_units("Pa")

        if "pressure".casefold() != get_dim_coord_names(temperature_cube)[0]:
            raise ValueError(
                f"Pressure must be the leading axis (got '{get_dim_coord_names(temperature_cube)}')."
            )

        # Calculate contrail formation using numpy arrays
        contrail_formation_data = self.process_from_arrays(
            temperature_cube.data, humidity_cube.data, pressure_coord.points
        )

        # Create output cube using contrail formation data
        contrail_formation_cube = self._create_contrail_formation_cube(
            contrail_formation_data, temperature_cube
        )

        return contrail_formation_cube
