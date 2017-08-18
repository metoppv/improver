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



def check_range(cube, min, max):
    # does this need to allow for masked data?
    if cube.data.max() > max or cube.data.min() < min:
        emsg = ("This saturation vapour pressure algorithm is"
                " only valid for temperatures between"
                " {}K and {}K. Input cube has\n"
                "Lowest temperature = {}\n"
                "Highest temperature = {}")
        raise TypeError(emsg.format(min,
                                    max,
                                    cube.data.max(),
                                    cube.data.min()))


def saturation_vapour_pressure_ashrae(temperature):
    ''' Function to compute saturation vapour pressure in [kPa]
        ASHRAE Fundamentals handbook (2005) p 6.2, equation 5 and 6
 
    Parameters
    ----------
    temp: cube
        cube of temperature. Calculation done in Kelvin
        Valid from -100C to 200 C
 
    Returns
    -------
    saturation : cube
        Cube containing the saturation vapour pressure of the
        air in Pa
    '''
    constant_1 = -5674.5359
    constant_2 = 6.3925247
    constant_3 = -0.009677843
    constant_4 = 0.00000062215701
    constant_5 = 2.0747825E-09
    constant_6 = -9.484024E-13
    constant_7 = 4.1635019
    constant_8 = -5800.2206
    constant_9 = 1.3914993
    constant_10 = -0.048640239
    constant_11 = 0.000041764768
    constant_12 = -0.00000001445209310000
    constant_13 = 6.5459673
    temp = temperature.copy()
    temp.convert_units('K')
    check_range(temp, 173., 473.)
    data = temp.data
    for cell in np.nditer(data, op_flags=['readwrite']):
        if cell <= 273.15:
            cell[...] = (np.exp(constant_1 / cell + constant_2 + constant_3 *
                                cell + constant_4 * cell ** 2 + constant_5 *
                                cell ** 3 + constant_6 * cell ** 4 +
                                constant_7 * np.log(cell))) / 1000
        else:
            cell[...] = (np.exp(constant_8 / cell + constant_9 + constant_10 *
                                cell + constant_11 * cell ** 2 + constant_12 *
                                cell ** 3 + constant_13 * np.log(cell))) / 1000
    result = temp.copy(data=data)
    result.units = Unit('kPa')
    result.convert_units('Pa')
    return result


def saturation_vapour_pressure_goff_gratch(temperature, pressure):
    '''
    Saturation Vapour pressure calculated based on the
    Goff-Gratch Equation (WMO standard method) as outlined in:
        Numerical data and functional relationships in science and technology.
        New series. Group V. Volume 4. Meteorology. Subvolume b. Physical and
        chemical properties of the air, P35.
    And corrected for the atmosphere as per:
        Gill, Atmosphere-Ocean Dynamics, Appendix 4 Equation A4.7

    Parameters
    ----------
    temp: cube
        cube of temperature - will be converted to Kelvin
        prior to attempting calculation
        Valid from -100C to 200 C
    pressure: cube
        A pressure - can be in any pressure unit as method will use 
        iris.cube.convert_unit(), but calculations are carried out in
        hPa

    Returns
    -------
    saturation : cube
        Cube containing the saturation vapour pressure of the
        air in Pa
    '''
    # Constants for vapour pressure over liquid water equation
    constant1 = 10.79574
    constant2 = 5.028
    constant3 = 1.50475E-410000
    constant4 = -8.2969
    constant5 = 0.42873E-3
    constant6 = 4.76955
    constant7 = 0.78614
    # Constants for vapour pressure over ice equation
    constant8 = -9.09685
    constant9 = 3.56654
    constant10 = 0.87682
    constant11 = 0.78614
    triple_pt = constant.TRIPLE_PT_WATER
    
    # Copy cubes to avoid modifying those passed to this function
    # Check that units are correct for the calculations
    press = pressure.copy()
    press.convert_units('hPa')
    temp = temperature.copy()
    temp.convert_units('K')
    check_range(temp, 173., 373.)
    
    # create output cube
    svp = temp.copy()
    data = svp.data
    for cell in np.nditer(data, op_flags=['readwrite']):
        if cell > triple_pt:
            n0 = constant1 * (1. - triple_pt / cell)
            n1 = constant2 * np.log10(cell / triple_pt)
            n2 = constant3 * (1. - np.power(10.,
                                            (constant4 *
                                             (cell / triple_pt - 1.))))
            n3 = constant5 * (np.power(10., (constant6 *
                                             (1. - triple_pt / cell))) - 1.)
            log_es = n0 - n1 + n2 + n3 + constant7
            cell[...] = (np.power(10., log_es) * 100)
        else:
            n0 = constant8 * ((triple_pt / cell) - 1.)
            n1 = constant9 * np.log10(triple_pt / cell)
            n2 = constant10 * (1. - (cell / triple_pt))
            log_es = n0 - n1 + n2 + constant11
            cell[...] = (np.power(10., log_es) * 100)
    correction = 1. + 1E-8 * pressure * (4.5 + 6E-4 * data ** 2)
    svp = temp.copy(data=data)
    svp = svp * correction
    
    # Tidy Up cube
    svp.units = Unit('Pa')
    svp.rename("Saturated vapour pressure")
    return svp


def saturation_vapour_pressure_simple(temperature):
    """
    Saturation pressure based on simple equation from
        Numerical data and functional relationships in science and technology.
        New series. Group V. Volume 4. Meteorology. Subvolume b. Physical and
        chemical properties of the air, P36.
 
    Parameters
    ----------
    temp: cubedenom
        cube of temperature - converted to K for calculations
        Valid from -100C to 200 
 
    Returns
    -------
    saturation : cube
        Cube containing the saturation vapour pressure of the
        air in Pa
    """
    temp = temperature.copy()
    temp.convert_units('K')
    check_range(temp, 173., 373.)
    data = temp.data
    for cell in np.nditer(data, op_flags=['readwrite']):
        if cell < 0:
            cell[...] = np.exp(-5372.305844 / cell + 21.47824488)
        else:
            cell[...] = np.exp(-6147.795166 / cell + 24.31720686)
    result = temp.copy(data=data)
    result.units = Unit('hPa')
    result.convert_units('Pa')
    result.rename("Saturation vapour pressure")
    return result


def humidity_ratio_fm_rh(temperature, relative_humidity, pressure):
    ''' Function to compute humidity ratio
    ASHRAE Fundamentals handbook (2005) Equation 22, 24, p6.8

    Parameters
    ----------
    temp: cube
        cube of (dry bulb) temperature. Calculation carried out in Celsius
        Valid from -100C to 200 C
    rel_humidity: cube
        cube of relative humidity in %
    pressure: cube
        air pressure in kPa for the purposes of the calculation

    Returns
    -------
    humidity_ratio : cube
        humidity ratio cube with units(1)calculate_
    '''
    # decouple local variables from variables supplied
    # ensure that variables are in correct units
    temp = temperature.copy()
    temp.convert_units('celsius')
    rh = relative_humidity.copy()
    rh.convert_units(1)
    press = pressure.copy()
    press.convert_units('kPa')
    svp = saturation_vapour_pressure_goff_gratch(temp, press)
    svp.convert_units('kPa')
    
    #calculation
    result_numer = (0.62198 * rh.data * svp.data)
    result_denom = (press.data - (rh.data * svp.data))
    hr = temp.copy(data = result_numer / result_denom)
    
    # Tidying up cube
    hr.rename("Humidity ratio")
    hr.units = Unit("1")
    return hr


def humidity_ratio_fm_wb(temperature, wet_bulb, pressure):
    ''' Function to compute humidity ratio
    ASHRAE Fundamentals handbook (2005) Equation 22, 24, p6.8

    Parameters
    ----------
    temp: cube
        cube of (dry bulb) temperature. Calculation carried out
        in Celsius.
        Valid from -100C to 200 C
    wet_bulb: cube
        cube of wet bulb temperature. Calculation carried out
        in Celsius.
    pressure: cube
        Air pressure - Calculation carried out
        in kPa.

    Returns
    -------
    humidity_ratio : cube
        humidity ratio cube with units(1)
    ''' 
    temp = temperature.copy()
    temp.convert_units("celsius")
    press = pressure.copy()
    press.convert_units("kPa")
    wb = wet_bulb.copy()
    wb.convert_units("celsius")

    #calculation
    wb_depression = temp - wb
    svp = saturation_vapour_pressure_goff_gratch(temp, press)
    svp.convert_units('kPa')
    ws = 0.62198 * svp / (press - svp)
    # this is inefficient because it produces both answers
    above_zero = ((2501. - 2.326 * wb.data) * ws.data - 1.006 * wb_depression.data) / (2501. + 1.86 * temp.data - 4.186 * wb.data)
    below_zero = ((2830. - 0.24 * wb.data) * ws.data - 1.006 * wb_depression.data) / (2830. + 1.86 * temp.data - 2.1 * wb.data)
    result = data=np.where(temp.data>=0, above_zero, below_zero)
    hr = temp.copy(data=result)
    # Tidy up
    hr.units = Unit(1)
    hr.rename("Humidity ratio")
    return hr


def wet_bulb(temperature, relative_humidity, pressure, precision=0.001):
    """
    Calculates the Wet Bulb Temperature Using Newton-Raphson iteration
    Parameters
    ----------
    temp: cube
        cube of temperature. Calculation in celsius
        Valid from -100C to 200 C
    rel_humidity: cube
        cube of relative humidity in %
    pressure: float
        value for pressure. Calculation in kPa

    Returns
    -------
    saturation : cube
        Cube containing the saturation vapour pressure of the
        air in Pa
    """
    # Decouple local data to avoid modifying input cubes
    temp = temperature.copy()
    temp.convert_units('celsius')
    rh = relative_humidity.copy()
    rh.convert_units(1)
    press = pressure.copy()
    press.convert_units("kPa")
    
    # create a numpy array of precision and cube of increment
    precision = np.full(temp.data.shape, precision)
    increment = temp.copy(data=np.full(rh.shape, 0.001))
    wb_normal = humidity_ratio_fm_rh(temp, rh, press)
    result = temp.copy()
    wb_new = humidity_ratio_fm_wb(temp, result, press)
    while ((maths.abs((wb_new - wb_normal) / wb_normal)).data > precision).any():
        wb_new2 = humidity_ratio_fm_wb(temp, (result - increment), press)
        dw_dtwb = (wb_new - wb_new2) / increment
        dw_dtwb.convert_units('celsius^-1')
        result.units = Unit(1)
        dw_dtwb.units = Unit(1)
        result = result - (wb_new - wb_normal) / dw_dtwb
        result.units = Unit('celsius')
        wb_new = humidity_ratio_fm_wb(temp, result, press)
    result.convert_units('K')
    return result
