# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugin to construct the probability that snow could occur at the surface"""

import numpy as np
from iris.cube import Cube

from improver import BasePlugin
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)


class SnowProbabilityAtSurface(BasePlugin):
    """Plugin to calculate the probability that snow could be being present at the surface.

    The probability is calculated by looking at the wet bulb temperature integral at the surface. If the
    wet bulb temperature is 0 Kelvin metres then the probability of snow is set to 1. If it is
    greater than 225 Kelvin metres then the probability is set to zero. For values in between
    0 and 225, the probability varies linearly.
    """

    def __init__(self) -> None:
        self.xp = [0, 225]
        self.yp = [1, 0]

    def process(self, wet_bulb_temp_int_at_surf: Cube) -> Cube:
        """
        Generate a probability that snow could be present at the surface based on input wet bulb temperature integral at surface.

        Args:
            wet_bulb_temp_int_at_surf:
                A cube of wet bulb temperature integral at the surface

        Returns:
            Probability of snow occuring at surface.
        """

        input_shape = wet_bulb_temp_int_at_surf.shape
        data = wet_bulb_temp_int_at_surf.data.flatten()

        prob_snow = np.interp(data, self.xp, self.yp)

        prob_snow_reshaped = np.reshape(prob_snow, input_shape)
        snow_probability_cube = create_new_diagnostic_cube(
            "probability_of_snow_at_surface",
            "1",
            template_cube=wet_bulb_temp_int_at_surf,
            mandatory_attributes=generate_mandatory_attributes(
                [wet_bulb_temp_int_at_surf]
            ),
            data=prob_snow_reshaped,
        )

        return snow_probability_cube
