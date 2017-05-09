"""Module to contain constants used for Ensemble Copula Coupling."""

# Specify the bounds for each phenomenon for creating the empirical
# cumulative distribution function.
bounds_for_ecdf = {"air_temperature": (-40, 50),
                   "wind_speed": (0, 50),
                   "air_pressure_at_sea_level": (940, 1070)}

# Specify the units for the bounds for each phenomenon
units_of_bounds_for_ecdf = {"air_temperature": "degreesC",
                            "wind_speed": "m s^-1",
                            "air_pressure_at_sea_level": "hPa"}
