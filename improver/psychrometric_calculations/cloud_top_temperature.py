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
    dry_adiabatic_temperature,
    adjust_for_latent_heat,
)


class CloudTopTemperature(BasePlugin):
    """Plugin to calculate the convective cloud top temperature from the
    cloud condensation level temperature and pressure, and temperature and
    humidity on pressure levels data."""

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
        self.humidity = Cube(None)

    def _calculate_cct(self) -> ndarray:
        """Does the main bit of work. Uses mask to determine where more ascent is needed"""
        p_coord = self.temperature.coord("pressure")
        ccl_with_mask = np.ma.masked_all_like(self.ccl.data)
        cct = np.ma.masked_array(self.ccl.data)
        for p, t, q in zip(
            p_coord.points,
            self.temperature.slices_over("pressure"),
            self.humidity.slices_over("pressure"),
        ):
            ccl_with_mask = np.ma.where(
                p < self.ccl.coord("air_pressure").points, self.ccl.data, ccl_with_mask
            )
            t_dry = dry_adiabatic_temperature(
                self.ccl.data, self.ccl.coord("air_pressure").points, p
            )
            t_2, _ = adjust_for_latent_heat(t_dry, q.data, p)
            ccl_with_mask[t_2 > t.data].mask = True
            cct[~ccl_with_mask.mask] = t_2[~ccl_with_mask.mask]
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
                [self.ccl, self.temperature, self.humidity]
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
                temperature on pressure levels and specific humidity on pressure levels

        Returns:
            Cube of cloud top temperature
        """
        self.ccl, self.temperature, self.humidity = cubes
        cct = self._make_cct_cube(self._calculate_cct())
        return cct
