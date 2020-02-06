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
"""Module containing feels like temperature calculation plugins"""

import numpy as np

from improver.metadata.utilities import (
    generate_mandatory_attributes, create_new_diagnostic_cube)
from improver.psychrometric_calculations.psychrometric_calculations \
    import calculate_svp_in_air


def _calculate_wind_chill(temperature, wind_speed):
    """
    Calculates the wind chill from 10 m wind speed and temperature based on
    the wind chill temperature index from a linear regression equation detailed
    in THE NEW WIND CHILL EQUIVALENT TEMPERATURE CHART, Osczevski and
    Bluestein, 2005, table 2.

    Args:
        temperature (numpy.ndarray):
            Air temperature in degrees celsius
        wind_speed (numpy.ndarray):
            Wind speed in kilometres per hour

    Returns:
        numpy.ndarray:
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
    the wind). The model introduces a compensation factor where it assumes that
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
    eqn_component = (wind_speed)**0.16
    wind_chill = (
        13.12 + 0.6215 * temperature - 11.37 * eqn_component +
        0.3965 * temperature * eqn_component).astype(np.float32)
    return wind_chill


def _calculate_apparent_temperature(temperature, wind_speed,
                                    relative_humidity, pressure):
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

    This function looks up a value for the saturation vapour pressure of
    water vapour using the temperature and a table of values. These tabulated
    values are found using lookup_svp and are corrected to the saturated
    vapour pressure in air using pressure_correct_svp, both functions are from
    the WetBulbTemperature plugin which makes use of the Goff-Gratch method.

    Args:
        temperature (numpy.ndarray):
            Temperatures in degrees celsius
        wind_speed (numpy.ndarray):
            10m wind speeds in metres per second
        relative_humidity (numpy.ndarray):
            Relative humidities (fractional)
        pressure (numpy.ndarray):
            Pressure in Pa

    Returns:
        numpy.ndarray:
            Apparent temperatures in degrees celsius

    References:
        Steadman, R. (1984). A Universal Scale of Apparent Temperature.
        Journal of Climate and Applied Meteorology, 23(12), pp.1674-1687
    """
    t_kelvin = temperature + 273.15
    svp = calculate_svp_in_air(t_kelvin, pressure)
    avp = 0.001 * svp * relative_humidity
    apparent_temperature = (-2.7 + 1.04 * temperature + 2.0 * avp -
                            0.65 * wind_speed).astype(np.float32)
    return apparent_temperature


def _feels_like_temperature(temperature, apparent_temperature, wind_chill):
    """
    Calculates feels like temperature from inputs in degrees Celsius using a
    combination of the wind chill index and Steadman's apparent temperature
    equation as follows:

    If temperature < 10 degress C: The feels like temperature is equal to
    the wind chill.

    If temperature > 20 degress C: The feels like temperature is equal to
    the apparent temperature.

    If 10 <= temperature <= 20 degrees C: A weighting (alpha) is calculated
    in order to blend between the wind chill and the apparent temperature.

    Args:
        temperature (numpy.ndarray)
        apparent_temperature (numpy.ndarray)
        wind_chill (numpy.ndarray)

    Returns:
        numpy.ndarray
    """
    feels_like_temperature = np.zeros(temperature.shape, dtype=np.float32)
    feels_like_temperature[temperature < 10] = wind_chill[temperature < 10]

    alpha = (temperature-10.0)/10.0
    temp_flt = (alpha*apparent_temperature + ((1-alpha)*wind_chill))
    between = (temperature >= 10) & (temperature <= 20)
    feels_like_temperature[between] = temp_flt[between]

    feels_like_temperature[temperature > 20] = (
        apparent_temperature[temperature > 20])

    return feels_like_temperature


def calculate_feels_like_temperature(
        temperature, wind_speed, relative_humidity, pressure,
        model_id_attr=None):
    """
    Calculates the feels like temperature using a combination of
    the wind chill index and Steadman's apparent temperature equation.

    Args:
        temperature (iris.cube.Cube):
            Cube of air temperatures
        wind_speed (iris.cube.Cube):
            Cube of 10m wind speeds
        relative_humidity (iris.cube.Cube):
            Cube of relative humidities
        pressure (iris.cube.Cube):
            Cube of air pressure
        model_id_attr (str):
            Name of the attribute used to identify the source model for
            blending.

    Returns:
        iris.cube.Cube:
            Cube of feels like temperatures in the same units as the input
            temperature cube.
    """
    t_cube = temperature.copy()
    t_cube.convert_units('degC')
    t_celsius = t_cube.data

    w_cube = wind_speed.copy()
    w_cube.convert_units('m s-1')
    p_cube = pressure.copy()
    p_cube.convert_units('Pa')
    rh_cube = relative_humidity.copy()
    rh_cube.convert_units('1')
    apparent_temperature = _calculate_apparent_temperature(
        t_celsius, w_cube.data, rh_cube.data, p_cube.data)

    w_cube.convert_units('km h-1')
    wind_chill = _calculate_wind_chill(t_celsius, w_cube.data)

    feels_like_temperature = _feels_like_temperature(
        t_celsius, apparent_temperature, wind_chill)

    attributes = generate_mandatory_attributes(
        [temperature, wind_speed, relative_humidity, pressure],
        model_id_attr=model_id_attr)
    feels_like_temperature_cube = create_new_diagnostic_cube(
        "feels_like_temperature", "degC", temperature,
        attributes, data=feels_like_temperature)
    feels_like_temperature_cube.convert_units(temperature.units)

    return feels_like_temperature_cube
