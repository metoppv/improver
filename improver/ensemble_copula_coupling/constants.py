# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to contain constants used for Ensemble Copula Coupling."""

from collections import namedtuple

from improver.constants import ABSOLUTE_ZERO

# Define a namedtuple class for use in the bounds_for_ecdf dictionary.
Bounds = namedtuple("bounds", "value units")

# For the creation of an empirical cumulative distribution function,
# the following dictionary specifies the end points of the distribution,
# as a first approximation of likely climatological lower and upper bounds.
# The units for the end points of the distribution are specified for each
# phenomenon. SI units are used throughout with the exception of precipitation
# rates, where mm/h provides more human-readable values.
# Scientific Reference:
# Flowerdew, J., 2014.
# Calibrated ensemble reliability whilst preserving spatial structure.
# Tellus Series A, Dynamic Meteorology and Oceanography, 66, 22662.

# Grouped by parameter, then sorted alphabetically.
BOUNDS_FOR_ECDF = {
    # Cloud
    "cloud_area_fraction": Bounds((0, 1.0), "1"),
    "cloud_area_fraction_assuming_only_consider_surface_to_1000_feet_asl": Bounds(
        (0, 1.0), "1"
    ),
    "cloud_base_height_assuming_only_consider_cloud_area_fraction_greater_than_2p5_oktas": Bounds(
        (-4000, 20000), "m"
    ),
    "cloud_base_height_assuming_only_consider_cloud_area_fraction_greater_than_4p5_oktas": Bounds(
        (-4000, 20000), "m"
    ),
    "high_type_cloud_area_fraction": Bounds((0, 1.0), "1"),
    "low_and_medium_type_cloud_area_fraction": Bounds((0, 1.0), "1"),
    "low_type_cloud_area_fraction": Bounds((0, 1.0), "1"),
    "medium_type_cloud_area_fraction": Bounds((0, 1.0), "1"),
    # Precipitation amount
    "lwe_thickness_of_freezing_rainfall_amount": Bounds((0, 0.5), "m"),
    "lwe_thickness_of_graupel_and_hail_fall_amount": Bounds((0, 0.5), "m"),
    "lwe_thickness_of_precipitation_amount": Bounds((0, 0.5), "m"),
    "lwe_thickness_of_precipitation_amount_in_vicinity": Bounds((0, 0.5), "m"),
    "lwe_thickness_of_precipitation_amount_in_variable_vicinity": Bounds((0, 0.5), "m"),
    "lwe_thickness_of_sleetfall_amount": Bounds((0, 0.5), "m"),
    "lwe_thickness_of_snowfall_amount": Bounds((0, 0.5), "m"),
    "thickness_of_rainfall_amount": Bounds((0, 0.5), "m"),
    # Precipitation rate
    "lwe_precipitation_rate": Bounds((0, 400.0), "mm h-1"),
    "lwe_precipitation_rate_in_vicinity": Bounds((0, 400.0), "mm h-1"),
    "lwe_precipitation_rate_max": Bounds((0, 400.0), "mm h-1"),
    "lwe_sleetfall_rate": Bounds((0, 400.0), "mm h-1"),
    "lwe_snowfall_rate": Bounds((0, 400.0), "mm h-1"),
    "lwe_snowfall_rate_in_vicinity": Bounds((0, 400.0), "mm h-1"),
    "rainfall_rate": Bounds((0, 400.0), "mm h-1"),
    "rainfall_rate_in_vicinity": Bounds((0, 400.0), "mm h-1"),
    # Temperature
    "air_temperature": (Bounds((-100 - ABSOLUTE_ZERO, 60 - ABSOLUTE_ZERO), "Kelvin")),
    "feels_like_temperature": (
        Bounds((-100 - ABSOLUTE_ZERO, 60 - ABSOLUTE_ZERO), "Kelvin")
    ),
    "temperature_at_screen_level_daytime_max": (
        Bounds((-100 - ABSOLUTE_ZERO, 60 - ABSOLUTE_ZERO), "Kelvin")
    ),
    "temperature_at_screen_level_nighttime_min": Bounds(
        (-100 - ABSOLUTE_ZERO, 60 - ABSOLUTE_ZERO), "Kelvin"
    ),
    # Wind
    "wind_speed": Bounds((0, 50), "m s^-1"),
    "wind_speed_of_gust": Bounds((0, 200), "m s^-1"),
    # Others
    "air_pressure_at_sea_level": Bounds((79600, 108000), "Pa"),
    "dew_point_temperature": Bounds(
        (-100 - ABSOLUTE_ZERO, 60 - ABSOLUTE_ZERO), "Kelvin"
    ),
    "relative_humidity": Bounds((0, 1.2), "1"),
    "visibility_in_air": Bounds((0, 100000), "m"),
    "ultraviolet_index": Bounds((0, 25.0), "1"),
    "ultraviolet_index_daytime_max": Bounds((0, 25.0), "1"),
}
