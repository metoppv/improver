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
    "realization": {"unit": "1", "dtype": np.int32},
    "height": {"unit": "m"},
    "velocity": {"unit": "m s-1"},
    # standard diagnostics and suitable substrings (alphabetised for clarity)
    "air_temperature_lapse_rate": {"unit": "K m-1"},
    "cloud": {"unit": "1"},
    "fall_rate": {"unit": "m s-1"},
    "falling_snow_level": {"unit": "m"},
    "humidity": {"unit": "1"},
    "lapse_rate": {"unit": "K m-1"},
    "orographic_enhancement": {"unit": "m s-1"},
    "precipitation_rate": {"unit": "m s-1"},
    "pressure": {"unit": "Pa"},
    "temperature": {"unit": "K"},
    "thickness": {"unit": "m"},
    "ultraviolet_flux": {"unit": "W m-2"},
    "ultraviolet_index": {"unit": "1"},
    "visibility_in_air": {"unit": "m"},
    "weather_code": {"unit": "1", "dtype": np.int32},
    "wind_from_direction": {"unit": "degrees"},
    "wind_speed": {"unit": "m s-1"},
    "wind_to_direction": {"unit": "degrees"},
    # ancillary diagnostics
    "alphas": {"unit": "1"},
    "grid_with_halo": {"unit": "no_unit"},
    "emos_coefficients": {"unit": "1"},
    "surface_altitude": {"unit": "m"},
    "land_binary_mask": {"unit": "1", "dtype": np.int32},
    "radar_coverage_mask": {"unit": "1"},
    "topographic_zone_weights": {"unit": "1"},
    "topographic_zone": {"unit": "m"},
    "topography_mask": {"unit": "1", "dtype": np.int32},
    "grid_neighbours": {"unit": "1"},
    "silhouette_roughness": {"unit": "1"},
    "standard_deviation_of_height_in_grid_cell": {"unit": "m"},
    "vegetative_roughness_length": {"unit": "m"},
    # emos specific coordinates
    "coefficient_index": {"unit": "1", "dtype": np.int32},
    "coefficient_name": {"unit": "no_unit", "dtype": np.unicode_},
    # spot-data specific coordinates
    "altitude": {"unit": "m"},
    "spot_index": {"unit": "1", "dtype": np.int32},
    "wmo_id": {"unit": "no_unit", "dtype": np.unicode_},
    "neighbour_selection_method": {"unit": "1", "dtype": np.int32},
    "neighbour_selection_method_name": {"unit": "no_unit",
                                        "dtype": np.unicode_},
    "grid_attributes": {"unit": "1", "dtype": np.int32},
    "grid_attributes_key": {"unit": "no_unit", "dtype": np.unicode_},
    # To be removed after input standardisation has been implemented
    "model_level_number": {"unit": "1", "dtype": np.int64},
    "sigma": {"unit": "1"},
}
