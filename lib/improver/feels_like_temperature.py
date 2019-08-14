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

from improver.psychrometric_calculations.psychrometric_calculations \
    import WetBulbTemperature


def calculate_wind_chill(temperature, wind_speed):
    """
    Calculates the wind chill from 10 m wind speed and temperature based on
    the wind chill temperature index from a linear regression equation detailed
    in THE NEW WIND CHILL EQUIVALENT TEMPERATURE CHART, Osczevski and
    Bluestein, 2005, table 2.

    Args:
      temperature (iris.cube.Cube):
        Cube of air temperatures

      wind_speed (iris.cube.Cube):
        Cube of 10m wind speeds

    Returns:
      wind_chill (iris.cube.Cube):
        Cube of wind chill temperatures. The units of wind chill will be the
        same as the units of the temperature cube when it is input into the
        function.

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
    temp_units = temperature.copy().units
    wind_units = wind_speed.copy().units
    # convert temperature units
    temperature.convert_units('celsius')
    # convert wind speed to km/h
    wind_speed.convert_units('km h-1')
    eqn_component = (wind_speed.data)**0.16
    wind_chill_data = (
        13.12 + 0.6215 * temperature.data - 11.37 * eqn_component +
        0.3965 * temperature.data * eqn_component).astype(np.float32)
    wind_chill = temperature.copy(data=wind_chill_data)
    wind_chill.rename("wind_chill")
    wind_chill.convert_units(temp_units)
    temperature.convert_units(temp_units)
    wind_speed.convert_units(wind_units)
    return wind_chill


def calculate_apparent_temperature(temperature, wind_speed,
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
      temperature (iris.cube.Cube):
        Cube of air temperatures

      wind_speed (iris.cube.Cube):
        Cube of 10m wind speeds

      relative_humidity (iris.cube.Cube):
        Cube of relative humidities

      pressure (iris.cube.Cube):
        Cube of air pressure

    Returns:
      apparent_temperature (iris.cube.Cube):
        Cube of apparent temperatures. The units of apparent temperature
        will be the same as the units of the temperature cube when it is input
        into the function.

    References:
      Steadman, R. (1984). A Universal Scale of Apparent Temperature.
      Journal of Climate and Applied Meteorology, 23(12), pp.1674-1687
    """
    # take a copy of each cube's original units
    temp_units = temperature.copy().units
    wind_units = wind_speed.copy().units
    pressure_units = pressure.copy().units
    relative_humidity_units = relative_humidity.copy().units

    # ensure units are correct
    wind_speed.convert_units('m s-1')
    pressure.convert_units('Pa')
    relative_humidity.convert_units('1')
    temperature.convert_units('K')
    # look up saturated vapour pressure
    svp = WetBulbTemperature().lookup_svp(temperature)
    # convert to SVP in air
    svp = WetBulbTemperature().pressure_correct_svp(
        svp, temperature, pressure)
    # convert temperature units
    temperature.convert_units('celsius')
    # calculate actual vapour pressure
    # and convert relative humidities to fractional values
    avp_data = svp.data*relative_humidity.data
    avp = svp.copy(data=avp_data)
    avp.rename("actual_vapour_pressure")
    avp.convert_units('kPa')
    # calculate apparent temperature
    apparent_temperature_data = (
        -2.7 + 1.04 * temperature.data + 2.0 * avp.data -
        0.65 * wind_speed.data).astype(np.float32)
    apparent_temperature = temperature.copy(data=apparent_temperature_data)
    apparent_temperature.rename("apparent_temperature")
    apparent_temperature.convert_units(temp_units)

    # convert units back to input units
    temperature.convert_units(temp_units)
    wind_speed.convert_units(wind_units)
    pressure.convert_units(pressure_units)
    relative_humidity.convert_units(relative_humidity_units)

    return apparent_temperature


def calculate_feels_like_temperature(temperature, wind_speed,
                                     relative_humidity, pressure):
    """
    Calculates the feels like temperature using a combination of
    the wind chill index and Steadman's apparent temperature equation with
    the following method:

    If temperature < 10 degress C: The feels like temperature is equal to
    the wind chill.

    If temperature > 20 degress C: The feels like temperature is equal to
    the apparent temperature.

    If 10 <= temperature <= 20 degrees C: A weighting (alpha) is calculated
    in order to blend between the wind chill and the apparent temperature.

    Args:
      temperature (iris.cube.Cube):
        Cube of air temperatures

      wind_speed (iris.cube.Cube):
        Cube of 10m wind speeds

      relative_humidity (iris.cube.Cube):
        Cube of relative humidities

      pressure (iris.cube.Cube):
        Cube of air pressure

    Returns:
      feels_like_temperature (iris.cube.Cube):
        Cube of feels like temperatures. The units of feels like temperature
        will be the same as the units of the temperature cube when it is input
        into the function.
    """
    temp_units = temperature.units
    # convert temperature units
    temperature.convert_units('celsius')

    wind_chill = calculate_wind_chill(temperature, wind_speed)
    apparent_temperature = calculate_apparent_temperature(
        temperature, wind_speed, relative_humidity, pressure)

    t_data = temperature.data
    feels_like_temperature_data = np.zeros(t_data.shape, dtype=np.float32)
    alpha = np.zeros(t_data.shape)

    # if temperature < 10 degrees Celsius:
    feels_like_temperature_data[t_data < 10] = wind_chill.data[t_data < 10]

    # if temperature >= 10 degrees Celsius and <= 20 degrees Celsius:
    # calculate weighting and blend between wind chill index
    # and Steadman equation
    alpha = (t_data-10.0)/10.0
    temp_flt = (
        alpha*apparent_temperature.data + ((1-alpha)*wind_chill.data))
    t_data_between = (t_data >= 10) & (t_data <= 20)
    feels_like_temperature_data[t_data_between] = temp_flt[t_data_between]

    # if temperature > 20 Celsius:
    feels_like_temperature_data[t_data > 20] = (
        apparent_temperature.data[t_data > 20])

    feels_like_temperature = temperature.copy(data=feels_like_temperature_data)
    feels_like_temperature.rename("feels_like_temperature")
    feels_like_temperature.convert_units(temp_units)
    temperature.convert_units(temp_units)

    return feels_like_temperature
