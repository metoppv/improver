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
"""
Module to contain the default units for use within IMPROVER.

The DEFAULT_UNITS dictionary has the following form.

<str>:
    The principle key is the name of a coordinate or diagnostic.

"unit": <str>
    The standard/default units for the coordinate or diagnostic
    described by the key.  This is mandatory.
"dtype": <dtype>
    The standard/default data type in which the coordinate points
    or diagnostic values should be stored.  This is optional; if
    not set, float32 is assumed.
"""

import numpy as np

DEFAULT_UNITS = {
    # time coordinates and suitable substrings
    "time": {
        "unit": "seconds since 1970-01-01 00:00:00",
        "dtype": np.int64},
    "forecast_period": {
        "unit": "seconds",
        "dtype": np.int32},
    # other standard coordinates and substrings
    "longitude": {"unit": "degrees"},
    "latitude": {"unit": "degrees"},
    "projection_x_coordinate": {"unit": "m"},
    "projection_y_coordinate": {"unit": "m"},
    "percentile": {"unit": "%"},
    "probability": {"unit": "1"},
    # standard diagnostics and suitable substrings (alphabetised for clarity)
    "fall_rate": {"unit": "m s-1"},
    "lapse_rate": {"unit": "K m-1"},
    "precipitation_rate": {"unit": "m s-1"},
    "temperature": {"unit": "K"},
    "thickness": {"unit": "m"}
}
