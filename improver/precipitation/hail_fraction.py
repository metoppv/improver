# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing the HailFraction class."""

from typing import Optional, Union

import numpy as np
from iris.cube import Cube, CubeList

from improver import PostProcessingPlugin
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.common_input_handle import as_cubelist
from improver.utilities.cube_checker import assert_spatial_coords_match


class HailFraction(PostProcessingPlugin):
    """
    Calculates the hail fraction using the maximum vertical updraught,
    the hail size, the cloud condensation level temperature, the convective cloud top
    temperature and altitude of the hail to rain phase change.

    """

    def __init__(self, model_id_attr: Optional[str] = None) -> None:
        """
        Initialise the class.

        Args:
            model_id_attr:
                Name of the attribute used to identify the source model for
                blending.
        """
        self.model_id_attr = model_id_attr

    @staticmethod
    def _compute_hail_fraction(
        vertical_updraught: Cube,
        hail_size: Cube,
        cloud_condensation_level: Cube,
        convective_cloud_top: Cube,
        hail_melting_level: Cube,
        orography: Cube,
    ) -> np.ndarray:
        """Computation of a hail fraction using the following steps: firstly,
        the hail fraction is estimated as varying linearly with the maximum vertical
        updraught so that a maximum vertical updraught of 5 m/s has a hail fraction of
        0 whilst a maximum vertical updraught of 50 m/s has a hail fraction of 0.25.
        These values, including the hail fraction upper bound, are based on expert
        elicitation. Next, the hail fraction is set to zero if either the cloud
        condensation level temperature is below -5 Celsius, the convective cloud top
        temperature is above -15 Celsius or the hail melting level is above orography.
        The convective cloud top temperature can also be missing, indicating that
        there is no deep convection, in which case the hail fraction is also set to
        zero.
        As a final check, the hail size is then checked for hail with a size larger
        than 2 mm. If the hail size is above this limit but the hail fraction is
        below 0.05, the hail fraction is set to 0.05.

        The values chosen are based on expert elicitation with some information from
        Dennis & Kumjian, 2017.

        References:
            Dennis, E.J., and M.R. Kumjian. 2017.
            “The Impact of Vertical Wind Shear on Hail Growth in Simulated Supercells.”
            J. Atmos. Sci. 74 (3): 641-663. https://doi.org/10.1175/JAS-D-16-0066.1.

        Returns:
            Hail fraction array.
        """

        # Hail size of 2 mm that always corresponds to a non-zero hail fraction.
        hail_size_limit = 0.002
        # Cloud condensation level temperature in Kelvin. If below this temperature,
        # a hail fraction of 0 is set.
        ccl_limit = 268.15
        # Convective cloud top temperature in Kelvin. If above this temperature, or missing,
        # a hail fraction of 0 is set.
        cct_limit = 258.15

        # Ensure CCT is a masked array.
        ctt_data = convective_cloud_top.data
        if not isinstance(ctt_data, np.ma.MaskedArray):
            ctt_data = np.ma.masked_invalid(ctt_data)

        hail_fraction = np.interp(vertical_updraught.data, [5, 50], [0, 0.25]).astype(
            np.float32
        )
        hail_fraction[
            (cloud_condensation_level.data < ccl_limit)
            | ctt_data.mask
            | (ctt_data > cct_limit)
            | (hail_melting_level.data > orography.data)
        ] = 0
        hail_fraction[(hail_size.data > hail_size_limit) & (hail_fraction < 0.05)] = (
            0.05
        )
        return hail_fraction

    def process(self, *cubes: Union[Cube, CubeList]) -> Cube:
        """Calculates the hail fraction using the maximum vertical updraught,
        the hail_size, the cloud condensation level temperature, the convective cloud
        top temperature, the altitude of the hail to rain phase change and the
        orography.

        Args:
            cubes:
                vertical_updraught: Maximum vertical updraught.
                hail_size: Hail size.
                cloud_condensation_level: Cloud condensation level temperature.
                convective_cloud_top: Convective cloud top.
                hail_melting_level: Altitude of the melting of hail to rain.
                orography: Altitude of the orography.

        Returns:
            Hail fraction cube.
        """
        cubes = as_cubelist(*cubes)
        (
            vertical_updraught,
            hail_size,
            cloud_condensation_level,
            convective_cloud_top,
            hail_melting_level,
            orography,
        ) = cubes.extract(
            [
                "maximum_vertical_updraught",
                "diameter_of_hail_stones",
                "air_temperature_at_condensation_level",
                "air_temperature_at_convective_cloud_top",
                "altitude_of_rain_from_hail_falling_level",
                "surface_altitude",
            ]
        )

        vertical_updraught.convert_units("m s-1")
        hail_size.convert_units("m")
        cloud_condensation_level.convert_units("K")
        convective_cloud_top.convert_units("K")
        hail_melting_level.convert_units("m")
        orography.convert_units("m")
        assert_spatial_coords_match(
            [
                vertical_updraught,
                hail_size,
                cloud_condensation_level,
                convective_cloud_top,
                hail_melting_level,
                orography,
            ]
        )

        hail_fraction = self._compute_hail_fraction(
            vertical_updraught,
            hail_size,
            cloud_condensation_level,
            convective_cloud_top,
            hail_melting_level,
            orography,
        )

        hail_fraction_cube = create_new_diagnostic_cube(
            "hail_fraction",
            "1",
            template_cube=vertical_updraught,
            mandatory_attributes=generate_mandatory_attributes(
                [
                    vertical_updraught,
                    hail_size,
                    cloud_condensation_level,
                    convective_cloud_top,
                    hail_melting_level,
                ],
                model_id_attr=self.model_id_attr,
            ),
            data=hail_fraction,
        )
        return hail_fraction_cube
