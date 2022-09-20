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
"""Module containing the HailFraction class."""

from typing import Optional

import numpy as np
from iris.cube import Cube

from improver import PostProcessingPlugin
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)


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
        The hail size is then checked for hail with a size larger than 2 mm. If the
        hail size is above this limit but the hail fraction is below 0.05, the hail
        fraction is set to 0.05. As a final check, the hail fraction is set to zero if
        either the cloud condensation level temperature is below -5 Celsius, the
        convective cloud top temperature is above -15 Celsius and the hail melting level
        is above orography.

        The values chosen are based on expert elicitation with some information from
        Dennis & Kumjian, 2017.

        References:
            Dennis, E.J., and M.R. Kumjian. 2017.
            “The Impact of Vertical Wind Shear on Hail Growth in Simulated Supercells.”
            J. Atmos. Sci. 74 (3): 641-663. doi:https://doi.org/10.1175/JAS-D-16-0066.1.

        Returns:
            Hail fraction array.
        """

        # Hail size of 2 mm that always corresponds to a non-zero hail fraction.
        hail_size_limit = 0.002
        # Cloud condensation level temperature in Kelvin. If below this temperature,
        # a hail fraction of 0 is set.
        ccl_limit = 268.15
        # Convective cloud top temperature in Kelvin. If above this temperature,
        # a hail fraction of 0 is set.
        cct_limit = 258.15

        hail_fraction = np.interp(vertical_updraught.data, [5, 50], [0, 0.25]).astype(
            np.float32
        )
        hail_fraction[
            (hail_size.data > hail_size_limit) & (hail_fraction < 0.05)
        ] = 0.05
        hail_fraction[
            (cloud_condensation_level.data < ccl_limit)
            | (convective_cloud_top.data > cct_limit)
            | (hail_melting_level.data > orography.data)
        ] = 0
        return hail_fraction

    def _make_hail_fraction_cube(
        self,
        hail_fraction: np.ndarray,
        vertical_updraught: Cube,
        hail_size,
        cloud_condensation_level,
        convective_cloud_top,
        hail_melting_level,
    ) -> Cube:
        """Create a cube containing the hail fraction array using information
        from the other available cubes.

        Returns:
            Hail fraction cube.
        """
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

    def process(
        self,
        vertical_updraught: Cube,
        hail_size: Cube,
        cloud_condensation_level: Cube,
        convective_cloud_top: Cube,
        hail_melting_level: Cube,
        orography: Cube,
    ) -> Cube:
        """Calculates the hail fraction using the maximum vertical updraught,
        the hail_size, the cloud condensation level temperature, the convective cloud
        top temperature, the altitude of the hail to rain phase change and the
        orography.

        Args:
            vertical_updraught: Maximum vertical updraught in m/s.
            hail_size: Hail size in m.
            cloud_condensation_level: Cloud condensation level temperature in K.
            convective_cloud_top: Convective cloud top in K.
            hail_melting_level: Altitude of the melting of hail to rain in metres.
            orography: Altitude of the orography in metres.

        Returns:
            Hail fraction cube.
        """
        hail_fraction = self._compute_hail_fraction(
            vertical_updraught,
            hail_size,
            cloud_condensation_level,
            convective_cloud_top,
            hail_melting_level,
            orography,
        )
        return self._make_hail_fraction_cube(
            hail_fraction,
            vertical_updraught,
            hail_size,
            cloud_condensation_level,
            convective_cloud_top,
            hail_melting_level,
        )
