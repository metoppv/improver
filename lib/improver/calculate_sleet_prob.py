# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
"""A plugin to calculate probability of sleet"""

import numpy as np
from improver.metadata.utilities import create_new_diagnostic_cube


def calculate_sleet_probability(prob_of_snow,
                                prob_of_rain):
    """
    This calculates the probability of sleet using the calculation:
    prob(sleet) = 1 - (prob(falling_snow_level at or below surface > 0.8) +
                       prob(falling_rain_level at or above surface > 0.8))

    Args:
      prob_of_snow_falling_level (iris.cube.Cube):
        Cube of snow falling level probabilities

      prob_of_rain_falling_level (iris.cube.Cube):
        Cube of rain falling level probabilities

    Returns:
      iris.cube.Cube:
        Cube of the probability of sleet

        Raises:
            ValueError: If the cube contains.
    """
    ones = np.ones((prob_of_snow.shape), dtype="float32")
    sleet_prob = (ones - (prob_of_snow.data + prob_of_rain.data))
    if np.any(sleet_prob < 0.0):
        msg = ("Error - Negative values found in cube")
        raise ValueError(msg)
    else:
        probability_of_sleet = create_new_diagnostic_cube(
            'probability_of_sleet', '1', prob_of_snow,
            attributes=prob_of_snow.attributes, data=sleet_prob)
        return probability_of_sleet
