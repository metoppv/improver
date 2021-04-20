# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""Top level constants"""

### Long name definitions
def prob_above_name(diagnostic: str) -> str:
    """Inline function to construct probability cube name"""
    return f"probability_of_{diagnostic}_above_threshold"


LIGHTNING_VICINITY_PROB = prob_above_name(
    "number_of_lightning_flashes_per_unit_area_in_vicinity"
)

CLOUD_NAME = "low_and_medium_type_cloud_area_fraction"
CLOUD_PROB_ABOVE = prob_above_name(CLOUD_NAME)
LOW_CLOUD_PROB_ABOVE = prob_above_name("low_type_cloud_area_fraction")
TEXTURE_PROB_ABOVE = prob_above_name(f"texture_of_{CLOUD_NAME}")
CONVECTION_PROB_ABOVE = prob_above_name("convective_ratio")

PRECIP_PROB_ABOVE = prob_above_name("lwe_precipitation_rate")
PRECIP_VICINITY_PROB_ABOVE = prob_above_name("lwe_precipitation_rate_in_vicinity")
RAIN_PROB_ABOVE = prob_above_name("rainfall_rate")
SLEET_PROB_ABOVE = prob_above_name("lwe_sleetfall_rate")
SNOW_PROB_ABOVE = prob_above_name("lwe_snowfall_rate")

VIS_PROB_BELOW = "probability_of_visibility_in_air_below_threshold"
