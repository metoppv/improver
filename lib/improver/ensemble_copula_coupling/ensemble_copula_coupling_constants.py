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

# Define a namedtuple for use in the bounds_for_ecdf dictionary.
bounds = namedtuple("bounds", "value units")

# For the creation of an empirical cumulative distribution function,
# the following dictionary specifies the end points of the distribution,
# as a first approximation of likely climatological lower and upper bounds.
# The units for the end points of the distribution are specified for each
# phenomenon. SI units are used exclusively.
# Scientific Reference:
# Flowerdew, J., 2014.
# Calibrated ensemble reliability whilst preserving spatial structure.
# Tellus Series A, Dynamic Meteorology and Oceanography, 66, 22662.

bounds_for_ecdf = {
    "air_temperature": (
        bounds((-40-ABSOLUTE_ZERO, 50-ABSOLUTE_ZERO), "Kelvin")),
    "wind_speed": bounds((0, 50), "m s^-1"),
    "air_pressure_at_sea_level": bounds((94000, 107000), "Pa")}
