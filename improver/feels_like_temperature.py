# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing feels like temperature calculation plugins"""

from typing import Optional

import numpy as np
from iris.cube import Cube
from numpy import ndarray

from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.psychrometric_calculations.psychrometric_calculations import (
    calculate_svp_in_air,
)


def _calculate_wind_chill(temperature: ndarray, wind_speed: ndarray) -> ndarray:
    """
    Calculates the wind chill from 10 m wind speed and temperature based on
    the wind chill temperature index from a linear regression equation detailed
    in THE NEW WIND CHILL EQUIVALENT TEMPERATURE CHART, Osczevski and
    Bluestein, 2005, table 2.

    Args:
        temperature:
            Air temperature in degrees celsius
        wind_speed:
            Wind speed in kilometres per hour

    Returns:
        Wind chill temperatures in degrees celsius

    References:
        Osczevski, R. and Bluestein, M. (2005). THE NEW WIND CHILL EQUIVALENT
        TEMPERATURE CHART. Bulletin of the American Meteorological Society,
        86(10), pp.1453-1458.

        Osczevski, R. and Bluestein, M. (2008). Comments on Inconsistencies in
        the New Windchill Chart at Low Wind Speeds. Journal of Applied
        Meteorology and Climatology, 47(10), pp.2737-2738.

    Science background:
    The 2005 Osczevski and Bluestein paper outlines the research and the
    assumptions made, and the 2008 paper clarifies poorly explained sections
    of the first paper.

    A brief summary of their assumptions are given below:
    The model aims to determine a worst-case scenario of wind chill. The wind
    speed "threshold" of 4.8 kph (1.34 m/s) stated in the 2005 papers does not
    refer to a threshold placed on the input windspeed data, which has no upper
    limit, but is the walking speed of an average person. This is therefore
    used as the minimum wind speed in their wind chill computer model, because
    even where wind speed is zero, a person would still experience wind chill
    from the act of walking (the model assumes that the person is walking into
    the wind). Furthermore, the equation is not valid for very low wind speeds
    and will return wind chill values higher than the air temperature if this
    lower wind speed limit is not imposed. Even with this limit, the calculated
    wind chill will be higher than the air temperature when the temperature is
    above about 11.5C and the wind is 4.8 kph.
    The model introduces a compensation factor where it assumes that
    the wind speed at 1.5 m (face level) is 2/3 that measured at 10 m. It also
    takes into account the thermal resistance of the skin on the human cheek
    with the assumption that the face is the most exposed area of skin
    during winter.

    The equation outlined in their paper is also not the equation used in their
    model (which was computationally expensive) but rather it is a linear
    regression equation which mimics the output of their model where wind
    speeds are greater than 3kph (0.8m/s) (clarified in the 2008 paper). The
    assumption being that lower wind speeds are usually not measured or
    reported accurately anyway.
    """
    eqn_component = np.clip(wind_speed, 4.824, None) ** 0.16
    wind_chill = (
        13.12
        + 0.6215 * temperature
        - 11.37 * eqn_component
        + 0.3965 * temperature * eqn_component
    ).astype(np.float32)
    return wind_chill


def _calculate_apparent_temperature(
    temperature: ndarray,
    wind_speed: ndarray,
    relative_humidity: ndarray,
    pressure: ndarray,
) -> ndarray:
    """
    Calculates the apparent temperature from 10 m wind speed, temperature
    and actual vapour pressure using the linear regression equation
    for shade described in A Universal Scale of Apparent Temperature,
    Steadman, 1984, page 1686, table 5.

    The method used to determine the original values used for the regression
    equation takes into account many variables which are detailed in Steadman's
    paper.

    The paper calculates apparent temperature for wind speeds up to 20 m/s.
    Here, the apparent temperature regression equation has been used for all
    wind speeds.

    Args:
        temperature:
            Temperatures in degrees celsius
        wind_speed:
            10m wind speeds in metres per second
        relative_humidity:
            Relative humidities (fractional)
        pressure:
            Pressure in Pa

    Returns:
        Apparent temperatures in degrees celsius

    References:
        Steadman, R. (1984). A Universal Scale of Apparent Temperature.
        Journal of Climate and Applied Meteorology, 23(12), pp.1674-1687
    """
    t_kelvin = temperature + 273.15
    svp = calculate_svp_in_air(t_kelvin, pressure)
    avp = 0.001 * svp * relative_humidity
    apparent_temperature = (
        -2.7 + 1.04 * temperature + 2.0 * avp - 0.65 * wind_speed
    ).astype(np.float32)
    return apparent_temperature


def _feels_like_temperature(
    temperature: ndarray, apparent_temperature: ndarray, wind_chill: ndarray
) -> ndarray:
    """
    Calculates feels like temperature from inputs in degrees Celsius using a
    combination of the wind chill index and Steadman's apparent temperature
    equation as follows:

    If temperature < 10 degrees C: The feels like temperature is equal to
    the wind chill.

    If temperature > 20 degrees C: The feels like temperature is equal to
    the apparent temperature.

    If 10 <= temperature <= 20 degrees C: A weighting (alpha) is calculated
    in order to blend between the wind chill and the apparent temperature.

    Args:
        temperature
        apparent_temperature
        wind_chill

    Returns:
        Feels like temperature.
    """
    feels_like_temperature = np.zeros(temperature.shape, dtype=np.float32)
    feels_like_temperature[temperature < 10] = wind_chill[temperature < 10]

    alpha = (temperature - 10.0) / 10.0
    temp_flt = alpha * apparent_temperature + ((1 - alpha) * wind_chill)
    between = (temperature >= 10) & (temperature <= 20)
    feels_like_temperature[between] = temp_flt[between]

    feels_like_temperature[temperature > 20] = apparent_temperature[temperature > 20]

    return feels_like_temperature


def calculate_feels_like_temperature(
    temperature: Cube,
    wind_speed: Cube,
    relative_humidity: Cube,
    pressure: Cube,
    model_id_attr: Optional[str] = None,
) -> Cube:
    """
    Calculates the feels like temperature using a combination of
    the wind chill index and Steadman's apparent temperature equation.

    Args:
        temperature:
            Cube of air temperatures
        wind_speed:
            Cube of 10m wind speeds
        relative_humidity:
            Cube of relative humidities
        pressure:
            Cube of air pressure
        model_id_attr:
            Name of the attribute used to identify the source model for
            blending.

    Returns:
        Cube of feels like temperatures in the same units as the input
        temperature cube.
    """
    t_cube = temperature.copy()
    t_cube.convert_units("degC")
    t_celsius = t_cube.data

    w_cube = wind_speed.copy()
    w_cube.convert_units("m s-1")
    p_cube = pressure.copy()
    p_cube.convert_units("Pa")
    rh_cube = relative_humidity.copy()
    rh_cube.convert_units("1")
    apparent_temperature = _calculate_apparent_temperature(
        t_celsius, w_cube.data, rh_cube.data, p_cube.data
    )

    w_cube.convert_units("km h-1")
    wind_chill = _calculate_wind_chill(t_celsius, w_cube.data)

    feels_like_temperature = _feels_like_temperature(
        t_celsius, apparent_temperature, wind_chill
    )

    attributes = generate_mandatory_attributes(
        [temperature, wind_speed, relative_humidity, pressure],
        model_id_attr=model_id_attr,
    )
    feels_like_temperature_cube = create_new_diagnostic_cube(
        "feels_like_temperature",
        "degC",
        temperature,
        attributes,
        data=feels_like_temperature,
    )
    feels_like_temperature_cube.convert_units(temperature.units)

    return feels_like_temperature_cube
