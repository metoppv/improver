from typing import List

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


class CloudTopTemperature(BasePlugin):
    """Plugin to calculate the convective cloud top temperature from the
    cloud condensation level temperature and pressure, and temperature
    on pressure levels data."""

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

    def _calculate_cct(self) -> ndarray:
        """Does the main bit of work. Uses mask to determine where more ascent is needed"""
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
        # cct is the saturated adiabatic temperature at the last atmosphere pressure level where
        # the profile is buoyant.
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

    def process(self, cubes: List[Cube]) -> Cube:
        """

        Args:
            cubes:
                Contains an ordered list of cloud condensation level,
                and temperature on pressure levels

        Returns:
            Cube of cloud top temperature
        """
        self.ccl, self.temperature = cubes
        cct = self._make_cct_cube(self._calculate_cct())
        return cct
