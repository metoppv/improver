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
"""Module containing the default datatypes for use within IMPROVER."""

import numpy as np


# DEPRECATED
DEFAULT_UNITS = {
    # time coordinates and suitable substrings
    "time": {
        "unit": "seconds since 1970-01-01 00:00:00",
        "dtype": np.int64},
    "forecast_period": {
        "unit": "seconds",
        "dtype": np.int32},
    # other standard coordinates and substrings
    "height": {"unit": "m"},
    "latitude": {"unit": "degrees"},
    "longitude": {"unit": "degrees"},
    "projection_x_coordinate": {"unit": "m"},
    "projection_y_coordinate": {"unit": "m"},
    "percentile": {"unit": "%"},
    "probability": {"unit": "1"},
    "realization": {"unit": "1", "dtype": np.int32},
    # standard diagnostics and suitable substrings (alphabetised for clarity)
    "air_temperature_lapse_rate": {"unit": "K m-1"},
    "cloud": {"unit": "1"},
    "cloud_base_altitude_assuming_only_consider_cloud_area_fraction_greater_"\
    "than_2p5_oktas": {"unit": "m"},
    "fall_rate": {"unit": "m s-1"},
    "falling_snow_level": {"unit": "m"},
    "humidity": {"unit": "1"},
    "lapse_rate": {"unit": "K m-1"},
    "number_of_lightning_flashes_per_unit_area": {"unit": "m-2"},
    "orographic_enhancement": {"unit": "m s-1"},
    "precipitation_rate": {"unit": "m s-1"},
    "pressure": {"unit": "Pa"},
    "temperature": {"unit": "K"},
    "temperature_at_screen_level_daytime_max": {"unit": "K"},
    "temperature_at_screen_level_nighttime_min": {"unit": "K"},
    "thickness": {"unit": "m"},
    "ultraviolet_flux": {"unit": "W m-2"},
    "ultraviolet_index": {"unit": "1"},
    "velocity": {"unit": "m s-1"},
    "visibility_in_air": {"unit": "m"},
    "weather_code": {"unit": "1", "dtype": np.int32},
    "wind_from_direction": {"unit": "degrees"},
    "wind_speed": {"unit": "m s-1"},
    "wind_to_direction": {"unit": "degrees"},
    # ancillary diagnostics
    "alphas": {"unit": "1"},
    "emos_coefficients": {"unit": "1"},
    "grid_neighbours": {"unit": "1"},
    "grid_with_halo": {"unit": "no_unit"},
    "land_binary_mask": {"unit": "1", "dtype": np.int32},
    "topography_mask": {"unit": "1", "dtype": np.int32},
    "topographic_zone_weights": {"unit": "1"},
    "topographic_zone": {"unit": "m"},
    "silhouette_roughness": {"unit": "1"},
    "standard_deviation_of_height_in_grid_cell": {"unit": "m"},
    "surface_altitude": {"unit": "m"},
    "vegetative_roughness_length": {"unit": "m"},
    # emos specific coordinates
    "coefficient_index": {"unit": "1", "dtype": np.int32},
    "coefficient_name": {"unit": "no_unit", "dtype": np.unicode_},
    # spot-data specific coordinates
    "altitude": {"unit": "m"},
    "grid_attributes": {"unit": "1", "dtype": np.int32},
    "grid_attributes_key": {"unit": "no_unit", "dtype": np.unicode_},
    "neighbour_selection_method": {"unit": "1", "dtype": np.int32},
    "neighbour_selection_method_name": {"unit": "no_unit",
                                        "dtype": np.unicode_},
    "spot_index": {"unit": "1", "dtype": np.int32},
    "wmo_id": {"unit": "no_unit", "dtype": np.unicode_},
    # To be removed after input standardisation has been implemented
    "model_level_number": {"unit": "1", "dtype": np.int64},
    "sigma": {"unit": "1"},
}
