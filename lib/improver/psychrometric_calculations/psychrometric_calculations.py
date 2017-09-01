# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
"""Module to contain Psychrometric Calculations."""

from cf_units import Unit
import improver.constants as constant
import iris.analysis.maths as maths
import numpy as np
import scipy.optimize


def _check_range(cube, low, high):
    """Function to wrap functionality for throwing out temperatures
    too low or high for a method to use safely.

    Args:
        cube : iris.cube.Cube
            A cube of temperature.

        low : int or float
            Lowest allowable temperature for check

        high : int or float
            Highest allowable temperature for check

    Raises:
        ValueError : If any of the values in cube.data are outside the bounds
            set by the low and high variables.
    """

    if cube.data.max() > high or cube.data.min() < low:
        emsg = ("This saturation vapour pressure algorithm is"
                " only valid for temperatures between"
                " {}K and {}K. Input cube has\n"
                "Lowest temperature = {}\n"
                "Highest temperature = {}")
        raise ValueError(emsg.format(low,
                                     high,
                                     cube.data.min(),
                                     cube.data.max()))


def saturation_vapour_pressure_ashrae(temperature):
    ''' Compute Saturation vapour pressure from temperature, using ASHRAE
    (American Society of Heating, Refrigerating and Air-Conditioning
    Engineers) method.

    Args:
        temp : iris.cube.Cube
            Cube of temperature which will be converted to Kelvin
            prior to calculation
            Valid from -100C to 200 C

    Returns:
        result : iris.cube.Cube
            Cube containing the saturation vapour pressure of the
            air in Pa.

    References:
        ASHRAE Fundamentals handbook (2005) p 6.2, equation 5 and 6
    '''
    constants = {1: -5674.5359,
                 2: 6.3925247,
                 3: -0.009677843,
                 4: 0.00000062215701,
                 5: 2.0747825E-09,
                 6: -9.484024E-13,
                 7: 4.1635019,
                 8: -5800.2206,
                 9: 1.3914993,
                 10: -0.048640239,
                 11: 0.000041764768,
                 12: -0.00000001445209310000,
                 13: 6.5459673}
    temp = temperature.copy()
    temp.convert_units('K')
    _check_range(temp, 173., 473.)
    data = temp.data
    for cell in np.nditer(data, op_flags=['readwrite']):
        if cell <= constant.TRIPLE_PT_WATER:
            cell[...] = (np.exp(constants[1] / cell + constants[2] +
                                constants[3] * cell + constants[4] *
                                cell ** 2. + constants[5] * cell ** 3. +
                                constants[6] * cell ** 4. + constants[7] *
                                np.log(cell))) / 1000.
        else:
            cell[...] = (np.exp(constants[8] / cell + constants[9] +
                                constants[10] * cell + constants[11] *
                                cell ** 2. + constants[12] * cell ** 3. +
                                constants[13] * np.log(cell))) / 1000.
    result = temp.copy(data=data)
    result.units = Unit('kPa')
    result.convert_units('Pa')
    result.rename("saturated_vapour_pressure")
    return result


def saturation_vapour_pressure_goff_gratch(temperature, pressure):
    '''
    Saturation Vapour pressure calculation based on the
    Goff-Gratch Equation (WMO standard method) and corrected for pressure.

    Args:
        temperature : iris.cube.Cube
            Cube of temperature which will be converted to Kelvin
            prior to calculation
            Valid from -100C to 200 C

        pressure : iris.cube.Cube
            Cube of pressure which will be converted to hectoPascals
            prior to calculation

    Returns:
        svp : iris.cube.Cube
            Cube containing the saturation vapour pressure of the
            air in Pa

    References:
        Numerical data and functional relationships in science and technology.
        New series. Group V. Volume 4. Meteorology. Subvolume b. Physical and
        chemical properties of the air, P35.

        Gill, Atmosphere-Ocean Dynamics, Appendix 4 Equation A4.7
    '''
    constants = {1: 10.79574,
                 2: 5.028,
                 3: 1.50475E-410000,
                 4: -8.2969,
                 5: 0.42873E-3,
                 6: 4.76955,
                 7: 0.78614,
                 8: -9.09685,
                 9: 3.56654,
                 10: 0.87682,
                 11: 0.78614}
    triple_pt = constant.TRIPLE_PT_WATER

    # Copy cubes to avoid modifying those passed to this function
    # Check that units are correct for the calculations
    press = pressure.copy()
    press.convert_units('Pa')
    temp = temperature.copy()
    temp.convert_units('K')
    _check_range(temp, 173., 373.)

    # create output cube
    data = temp.data
    for cell in np.nditer(data, op_flags=['readwrite']):
        if cell > triple_pt:
            n0 = constants[1] * (1. - triple_pt / cell)
            n1 = constants[2] * np.log10(cell / triple_pt)
            n2 = constants[3] * (1. - np.power(10.,
                                               (constants[4] *
                                                (cell / triple_pt - 1.))))
            n3 = constants[5] * (np.power(10., (constants[6] *
                                                (1. - triple_pt / cell))) -
                                 1.)
            log_es = n0 - n1 + n2 + n3 + constants[7]
            cell[...] = (np.power(10., log_es) * 100)
        else:
            n0 = constants[8] * ((triple_pt / cell) - 1.)
            n1 = constants[9] * np.log10(triple_pt / cell)
            n2 = constants[10] * (1. - (cell / triple_pt))
            log_es = n0 - n1 + n2 + constants[11]
            cell[...] = (np.power(10., log_es) * 100)
    temp.convert_units('celsius')
    correction = 1. + 1E-8 * pressure * (4.5 + 6E-4 * temp.data ** 2)
    svp = temp.copy(data=data)
    svp = svp * correction

    # Tidy Up cube
    svp.units = Unit('Pa')
    svp.rename("saturated_vapour_pressure")
    return svp


def saturation_vapour_pressure_simple(temperature):
    """
    A simple saturation vapour pressure equation, considered to be within
    1% of correct value for the given range.

    Args:
        temperature : iris.cube.Cube
            Cube of temperature which will be converted to Kelvin
            prior to calculation
            Valid from -100C to 200

    Returns:
        results : iris.cube.Cube
        Cube containing the saturation vapour pressure of the
        air in Pa

    References:
        Numerical data and functional relationships in science and technology.
        New series. Group V. Volume 4. Meteorology. Subvolume b. Physical and
        chemical properties of the air, P36.
    """
    triple_point = constant.TRIPLE_PT_WATER
    temp = temperature.copy()
    temp.convert_units('K')
    _check_range(temp, 173., 373.)
    data = temp.data
    for cell in np.nditer(data, op_flags=['readwrite']):
        if cell > triple_point:
            cell[...] = np.exp(-5372.305884 / cell + 21.47824488)
        else:
            cell[...] = np.exp(-6147.795166 / cell + 24.31720686)
    result = temp.copy(data=data)
    result.units = Unit('hPa')
    result.convert_units('Pa')
    result.rename("saturated_vapour_pressure")
    return result


def humidity_ratio_fm_rh(temperature, relative_humidity, pressure):
    """Function to compute humidity ratio given temperature pressure and
    relative humidity.

    Args:
        temperature : iris.cube.Cube
            Cube of temperature which will be converted to celsius
            prior to calculation.
        relative_humidity : iris.cube.Cube
            Cube of relative humidity.
        pressure : iris.cube.Cube
            Cube of pressure which will be converted to kilopPascals
            prior to calculation.

    Returns
        hr : iris.cube.Cube
            humidity ratio cube with units (1).

    References:
        ASHRAE Fundamentals handbook (2005) Equation 22, 24, p6.8
    """
    # Decouple local variables from variables supplied
    # Ensure that variables are in correct units
    temp = temperature.copy()
    temp.convert_units('celsius')
    rh = relative_humidity.copy()
    rh.convert_units(1)
    press = pressure.copy()
    svp = saturation_vapour_pressure_goff_gratch(temp, press)
    press.convert_units('Pa')

    # Calculation
    result_numer = (0.62198 * rh.data * svp.data)
    result_denom = (press.data - (rh.data * svp.data))
    hr = temp.copy(data=result_numer / result_denom)

    # Tidying up cube
    hr.rename("humidity_ratio")
    hr.units = Unit("1")
    return hr


def humidity_ratio_fm_wb(temperature, wet_bulb, pressure):
    """Function to compute humidity ratio given a temperature pressure and
    wet bulb temperature.

    Args:
        temperature: iris.cube.Cube
            Cube of temperature which will be converted to Celsius
            prior to calculation

        wet_bulb: iris.cube.Cube
            Cube of wet_bulb_temperature which will be converted to Celsius
            prior to calculation

        pressure: iris.cube.Cube
            Cube of pressure which will be converted to kiloPascals
            prior to calculation

    Returns:
        humidity_ratio : iris.cube.Cube
            humidity ratio cube with units(1)

    References:
        ASHRAE Fundamentals handbook (2005) Equation 35, 37, p6.9
    """
    temp = temperature.copy()
    temp.convert_units("celsius")
    pressure.convert_units("kPa")
    wet_bulb.convert_units("celsius")

    # Calculation
    wet_bulb_depression = temp - wet_bulb
    svp = saturation_vapour_pressure_goff_gratch(temp, pressure)
    svp.convert_units('kPa')
    ws = 0.62198 * svp / (pressure - svp)
    # ASHRAE Fundamentals handbook (2005) Equation 35 & 37 pp6.9
    above_zero = (((2501. - 2.326 * wet_bulb.data) * ws.data -
                   1.006 * wet_bulb_depression.data) /
                  (2501. + 1.86 * temp.data - 4.186 * wet_bulb.data))
    below_zero = (((2830. - 0.24 * wet_bulb.data) * ws.data -
                   1.006 * wet_bulb_depression.data) /
                  (2830. + 1.86 * temp.data - 2.1 * wet_bulb.data))
    result = np.where(temp.data >= 0, above_zero, below_zero)
    hr = temp.copy(data=result)

    # Tidy up
    hr.units = Unit(1)
    hr.rename("humidity_ratio")
    return hr


def wet_bulb(temperature, relative_humidity, pressure,
             p_units="hPa", precision=0.00001):
    """
    Calculates the Wet Bulb Temperature Using Newton-Raphson iteration

    Args:
        temperature : iris.cube.Cube
            Cube of temperature which will be converted to Celsius
            prior to calculation
        relative_humidity : iris.cube.Cube
            Cube of relative humidity
        pressure : iris.cube.Cube or float
            Cube of pressure which will be converted to kilopascals
            prior to calculation

    Kwargs:
        p_units : string
            Units for pressure where pressure is given as a float.
        precision : float
            Degree of precision the user requires for this algorithm.
            Default value is 0.00001.

    Returns:
        result : iris.cube.Cube
            Cube containing the wet bulb temperature of the air in Kelvin

    Raises:
        ValueError: If any pressure is 0, as this otherwise causes a divide
        by zero error later.
        ValueError: If any relative humidity is 0, as this otherwise causes
        a divide by zero error later.
    """
    rh = relative_humidity.copy()
    rh.convert_units(1)

    # If pressure has been passed in as a single value cast it into a cube
    if isinstance(pressure, float):
        press = temperature.copy(data=np.full(temperature.shape, pressure))
        press.units = Unit(p_units)
    else:
        press = pressure.copy()
    press.convert_units("kPa")

    # check that lowest pressure and rh values are non-zero
    if press.data.min() == 0:
        emsg = ("This Wet Bulb Temperature algorithm is"
                " only valid for pressures greater than"
                " Zero. Input cube has\n"
                "Lowest pressure of:  {}\n")
        raise ValueError(emsg.format(press.data.min()))
    if rh.data.min() == 0:
        emsg = ("This Wet Bulb Temperature algorithm is"
                " only valid for relative humidities greater than"
                " Zero. Input cube has\n"
                "Lowest rh of:  {}\n")
        raise ValueError(emsg.format(rh.data.min()))

    # create a numpy array of precision and cube of increment
    result = temperature.copy()
    result.convert_units('celsius')
    precision = np.full(temperature.data.shape, precision)
    increment = result.copy(data=np.full(result.shape, 0.001))

    # Calculate baseline humidity ratio to compare against results of
    # relative humidity from wet_bulb
    hr_normal = humidity_ratio_fm_rh(temperature, rh, press)
    hr_new = humidity_ratio_fm_wb(temperature, result, press)

    while ((maths.abs((hr_new - hr_normal) / hr_normal)).data >
           precision).any():
        # create a new estimate of humidity ratio from wet bulb
        hr_new2 = humidity_ratio_fm_wb(temperature,
                                       (result - increment),
                                       press)
        dw_dthr = (hr_new - hr_new2) / increment
        # These are necessary because Iris makes temperature deltas
        # Kelvin and then objects to K - C
        result.units = Unit(1)
        dw_dthr.units = Unit(1)
        result2 = result - (hr_new - hr_normal) / dw_dthr
        # Where stopping criterion has been met copy the existing result
        # into that grid cell. Elsewhere replace the cell with the new result
        data = np.where((maths.abs((hr_new - hr_normal) /
                                   hr_normal)).data > precision,
                        result2.data,
                        result.data)
        result = result.copy(data=data)
        result.units = Unit('celsius')
        hr_new = humidity_ratio_fm_wb(temperature, result, press)
    result.convert_units('K')
    result.rename("wet_bulb_temperature")
    return result
