# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
"""Module to contain constants used for Ensemble Copula Coupling."""

from collections import namedtuple

from improver.constants import ABSOLUTE_ZERO

# Define a namedtuple class for use in the bounds_for_ecdf dictionary.
Bounds = namedtuple("bounds", "value units")

# For the creation of an empirical cumulative distribution function,
# the following dictionary specifies the end points of the distribution,
# as a first approximation of likely climatological lower and upper bounds.
# The units for the end points of the distribution are specified for each
# phenomenon. SI units are used exclusively.
# Scientific Reference:
# Flowerdew, J., 2014.
# Calibrated ensemble reliability whilst preserving spatial structure.
# Tellus Series A, Dynamic Meteorology and Oceanography, 66, 22662.

# 0.00003 ms-1 ~ 100 mm hr-1 for rainfall. Divided by 3 as a guess for snow!

BOUNDS_FOR_ECDF = {
    "air_temperature": (
        Bounds((-40-ABSOLUTE_ZERO, 50-ABSOLUTE_ZERO), "Kelvin")),
    "wind_speed": Bounds((0, 50), "m s^-1"),
    "air_pressure_at_sea_level": Bounds((94000, 107000), "Pa"),
    ("cloud_base_altitude_assuming_only_consider_cloud_area" +
     "_fraction_greater_than_2p5_oktas"): Bounds((-300, 20000), "m"),
    "cloud_area_fraction": Bounds((0, 1.0), "1"),
    "low_type_cloud_area_fraction": Bounds((0, 1.0), "1"),
    "rainfall_rate": Bounds((0, 0.00003), "m s-1"),
    "rainfall_rate_in_vicinity": Bounds((0, 0.00003), "m s-1"),
    "lwe_snowfall_rate": Bounds((0, 0.00001), "m s-1"),
    "lwe_snowfall_rate_in_vicinity": Bounds((0, 0.00001), "m s-1"),
    "visibility_in_air": Bounds((0, 100000), "m")
}
