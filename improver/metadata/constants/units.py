# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
"""Units for output variables"""

VARIABLE_UNIT_LOOKUP = {
    "air_pressure": "Pa",
    "air_pressure_at_sea_level": "Pa",
    "air_temperature": "K",
    "air_temperature_lapse_rate": "K m-1",
    "alphas": "1",
    "altitude_of_rain_falling_level": "m",
    "altitude_of_snow_falling_level": "m",
    "cloud_area_fraction": "1",
    "emos_coefficient_alpha": "K",
    "falling_snow_level_asl": "m",
    "grid_eastward_wind": "m s-1",
    "grid_neighbours": "1",
    "land_binary_mask": "1",
    "land_fraction": "1",
    "lapse_rate": "K m-1",
    "low_type_cloud_area_fraction": "1",
    "lwe_precipitation_rate": "m s-1",
    "medium_type_cloud_area_fraction": "1",
    "orographic_enhancement": "m s-1",
    "precipitation_advection_x_velocity": "m s-1",
    "precipitation_advection_y_velocity": "m s-1",
    "radar_coverage_mask": "1",
    "rainfall_rate": "m s-1",
    "rainfall_rate_composite": "mm/h",
    "relative_humidity": "1",
    "reliability_calibration_table": "1",
    "silhouette_roughness": "1",
    "smoothing_coefficient_x": "1",
    "standard_deviation_of_height_in_grid_cell": "m",
    "surface_air_pressure": "Pa",
    "surface_altitude": "m",
    "surface_downwelling_ultraviolet_flux_in_air": "W m-2",
    "surface_temperature": "K",
    "surface_upwelling_ultraviolet_flux_in_air": "W m-2",
    "thickness_of_rainfall_amount": "m",
    "topographic_zone_weights": "1",
    "topography_mask": "1",
    "ultraviolet_index": "1",
    "vegetative_roughness_length": "m",
    "wet_bulb_temperature": "K",
    "wet_bulb_temperature_integral": "K m",
    "wind_from_direction": "degrees",
    "wind_speed": "m s-1",
    "wind_speed_of_gust": "m s-1",
}
