"""Module to contain constants used for Ensemble Copula Coupling."""

# For the creation of an empirical cumulative distribution function,
# the following dictionary specifies the end points of the distribution,
# as a first approximation of likely climatological lower and upper bounds.
bounds_for_ecdf = {"air_temperature": (-40+273.15, 50+273.15),
                   "wind_speed": (0, 50),
                   "air_pressure_at_sea_level": (94000, 107000)}

# Specify the units for the end points of the distribution for each phenomenon.
# SI units are used exclusively.
units_of_bounds_for_ecdf = {"air_temperature": "Kelvin",
                            "wind_speed": "m s^-1",
                            "air_pressure_at_sea_level": "Pa"}
