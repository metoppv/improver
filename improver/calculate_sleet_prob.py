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

from improver.metadata.utilities import (
    create_new_diagnostic_cube, generate_mandatory_attributes)


def calculate_sleet_probability(prob_of_snow,
                                prob_of_rain):
    """
    This calculates the probability of sleet using the calculation:
    prob(sleet) = 1 - (prob(snow) + prob(rain))

    Args:
      prob_of_snow (iris.cube.Cube):
        Cube of the probability of snow.
      prob_of_rain (iris.cube.Cube):
        Cube of the probability of rain.

    Returns:
      iris.cube.Cube:
        Cube of the probability of sleet.

    Raises:
        ValueError: If the cube contains negative values for the the
                    probability of sleet.
    """
    ones = np.ones((prob_of_snow.shape), dtype="float32")
    sleet_prob = (ones - (prob_of_snow.data + prob_of_rain.data))
    if np.any(sleet_prob < 0.0):
        msg = "Negative values of sleet probability have been calculated."
        raise ValueError(msg)

    # Copy all of the attributes from the prob_of_snow cube
    mandatory_attributes = generate_mandatory_attributes([
        prob_of_rain, prob_of_snow])
    probability_of_sleet = create_new_diagnostic_cube(
        'probability_of_sleet', '1', prob_of_snow, mandatory_attributes,
        data=sleet_prob)
    return probability_of_sleet
