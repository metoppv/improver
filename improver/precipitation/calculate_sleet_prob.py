# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""A plugin to calculate probability of sleet"""

import numpy as np
from iris.cube import Cube

from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)


def calculate_sleet_probability(prob_of_snow: Cube, prob_of_rain: Cube) -> Cube:
    """
    This calculates the probability of sleet using the calculation:
    prob(sleet) = 1 - (prob(snow) + prob(rain))

    Args:
      prob_of_snow:
        Cube of the probability of snow. This can be a fraction (0 <= x <= 1) or
        categorical (0 or 1)
      prob_of_rain:
        Cube of the probability of rain. This can be a fraction (0 <= x <= 1) or
        categorical (0 or 1)

    Returns:
        Cube of the probability of sleet. This will be fractional or categorical,
        matching the highest precision of the inputs.

    Raises:
        ValueError: If the cube contains negative values for the the
                    probability of sleet.
    """
    if prob_of_snow.dtype != prob_of_rain.dtype:
        msg = (
            "The data types of the input cubes do not match. "
            f"prob_of_snow: {prob_of_snow.dtype}, prob_of_rain: {prob_of_rain.dtype}"
        )
        raise ValueError(msg)

    sleet_prob = (1 - (prob_of_snow.data + prob_of_rain.data)).astype(
        prob_of_snow.dtype
    )
    if np.any(sleet_prob < 0):
        msg = "Negative values of sleet probability have been calculated."
        raise ValueError(msg)

    # Copy all of the attributes from the prob_of_snow cube
    mandatory_attributes = generate_mandatory_attributes([prob_of_rain, prob_of_snow])
    probability_of_sleet = create_new_diagnostic_cube(
        "probability_of_sleet", "1", prob_of_snow, mandatory_attributes, data=sleet_prob
    )
    return probability_of_sleet
