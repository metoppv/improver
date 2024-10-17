# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing the CloudTopTemperature plugin"""
from typing import Union

import numpy as np
from iris.cube import Cube, CubeList
from numpy import ndarray

from improver import PostProcessingPlugin
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


class CloudTopTemperature(PostProcessingPlugin):
    """Plugin to calculate the convective cloud top temperature from the
    cloud condensation level temperature and pressure, and temperature
    on pressure levels data using saturated ascent.
    The temperature is that of the parcel after saturated ascent at the last pressure level
    where the parcel is buoyant. The interpolation required to get closer is deemed expensive.
    If the cloud top temperature is less than 4K colder than the cloud condensation level,
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
        self.t_at_ccl = Cube(None)
        self.p_at_ccl = Cube(None)
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
        A mask is used to keep track of which columns have already been solved, so that they
        are not included in future iterations (x3 speed-up).
        """
        cct = np.ma.masked_array(self.t_at_ccl.data.copy())
        q_at_ccl = saturated_humidity(self.t_at_ccl.data, self.p_at_ccl.data)
        ccl_with_mask = np.ma.masked_array(self.t_at_ccl.data, False)
        mask = ~ccl_with_mask.mask
        for t in self.temperature.slices_over("pressure"):
            t_environment = np.full_like(t.data, np.nan)
            t_environment[mask] = t.data[mask]

            (pressure,) = t.coord("pressure").points
            t_dry_parcel = dry_adiabatic_temperature(
                self.t_at_ccl.data[mask], self.p_at_ccl.data[mask], pressure,
            )

            t_parcel = np.full_like(t.data, np.nan)
            t_parcel[mask], _ = adjust_for_latent_heat(
                t_dry_parcel, q_at_ccl[mask], pressure
            )
            # Mask out points where parcel temperature, t_parcel, is less than atmosphere
            # temperature, t, but only after the parcel pressure, becomes lower than the
            # cloud base pressure.
            ccl_with_mask[mask] = np.ma.masked_where(
                (t_parcel[mask] < t_environment[mask])
                & (pressure < self.p_at_ccl.data[mask]),
                ccl_with_mask[mask],
            )
            # Find mask with CCL points that are still not masked.
            mask = ~ccl_with_mask.mask

            cct[mask] = t_parcel[mask]
            del t
            if mask.sum() == 0:
                break
        cct = np.ma.masked_where(self.t_at_ccl.data - cct < self.minimum_t_diff, cct)

        return cct

    def _make_cct_cube(self, data: ndarray) -> Cube:
        """Puts the data array into a CF-compliant cube"""
        attributes = {}
        if self.model_id_attr:
            attributes[self.model_id_attr] = self.t_at_ccl.attributes[
                self.model_id_attr
            ]
        cube = create_new_diagnostic_cube(
            "air_temperature_at_convective_cloud_top",
            "K",
            self.t_at_ccl,
            mandatory_attributes=generate_mandatory_attributes(
                [self.t_at_ccl, self.p_at_ccl]
            ),
            optional_attributes=attributes,
            data=data,
        )
        return cube

    def process(self, *cubes: Union[Cube, CubeList]) -> Cube:
        """

        Args:
            cubes:
                Cubes of 'temperature at cloud condensation level',
                'pressure at cloud condensation level' and 'temperature on pressure levels'.

        Returns:
            Cube of cloud top temperature
        """
        cubes = as_cubelist(*cubes)
        self.t_at_ccl, self.p_at_ccl, self.temperature = cubes.extract_cubes(
            [
                "air_temperature_at_condensation_level",
                "air_pressure_at_condensation_level",
                "air_temperature",
            ]
        )

        assert_spatial_coords_match([self.t_at_ccl, self.p_at_ccl, self.temperature])
        self.temperature.convert_units("K")
        self.t_at_ccl.convert_units("K")
        self.p_at_ccl.convert_units("Pa")
        cct = self._make_cct_cube(self._calculate_cct())
        return cct
