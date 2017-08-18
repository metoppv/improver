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
"""Unit tests for the the stand alone functions
in psychrometric_calculations.py"""

from iris.cube import Cube
from iris.tests import IrisTest
from iris.coords import DimCoord
from iris.coord_systems import GeogCS
from iris.fileformats.pp import EARTH_RADIUS
import numpy as np
from cf_units import Unit


from improver.psychrometric_calculations import (
    check_range,
    #saturation_vapour_pressure_ashrae,
    saturation_vapour_pressure_goff_gratch,
    saturation_vapour_pressure_simple,
    humidity_ratio_fm_wb,
    humidity_ratio_fm_rh,
    wet_bulb)

## ADD CHECKS THAT INPUT CUBES ARE !!!___NOT___!!! BEING MODIFIED!

def _make_test_cube(long_name, units, data=None):
    """
    Make a basic cube to run tests on
    """
    cs = GeogCS(EARTH_RADIUS)
    if data is None:
        data = np.array([[270., 270.],
                         [290., 290.]])
    elif data is "relative_humidity":
        data = np.array([[1., 1.],
                         [1., 1.]])
    elif data is "pressure":
        data = np.array([[1000., 1000.],
                         [1000., 1000.]])
    elif data is "wet_bulb_temp":
        data = np.array([[270., 270.],
                         [275., 275.]])
    cube = Cube(data, long_name=long_name)
    x_coord = DimCoord(np.linspace(-45.0, 45.0, 2), 'latitude',
                       units='degrees', coord_system=cs)
    y_coord = DimCoord(np.linspace(0, 180, 2), 'longitude',
                       units='degrees', coord_system=cs)
    cube.add_dim_coord(x_coord, 0)
    cube.add_dim_coord(y_coord, 1)
    cube.units = Unit(units)
    return cube


class Test_check_range(IrisTest):
    """Checks that the test_range method fails
    when out of limits values are given"""
    def test_fail(self):
        data = np.array([[270., 500.],
                         [290., 290.]])
        cube_in = _make_test_cube("temperature", "K", data=data)
        emsg = "only valid for temperatures between"
        with self.assertRaisesRegexp(TypeError, emsg):
            check_range(cube_in, 10., 300.)
        data = np.array([[5., 290.],
                         [290., 290.]])
        cube_in = _make_test_cube("temperature", "K", data=data)
        with self.assertRaisesRegexp(TypeError, emsg):
            check_range(cube_in, 10., 300.)
            
    # check that this works with masked data


class Test_calculate_svp_ashrae(IrisTest):
    """saturation_vapour_press == svp"""
    def test_basic(self):
        """test to check that the saturation_vapour_pressure
        method returns a cube with answers calculated to be correct
        """
        temp = _make_test_cube("temperature", "K")
        expected_data = np.array([[470.015, 470.015], [2259.917,  2259.917]])
        result = saturation_vapour_pressure_simple(temp)
        self.assertArrayAlmostEqual(result.data, expected_data, decimal=3)
        self.assertEqual(result.units, Unit('Pa'))


class Test_calculate_svp_goff_gratch(IrisTest):
    """saturation_vapour_press == svp"""
    def test_basic(self):    def test_non_modification(self):
        temp_data = np.array([[-2, 5], [10, 20]])
        temp = _make_test_cube("temperature", "celsius", data=temp_data)
        pressure = _make_test_cube("pressure", "Pa", data="pressure")
        result = saturation_vapour_pressure_goff_gratch(temp, pressure)
        self.assertEqual(temp, _make_test_cube("temperature", "celsius", data=temp_data))
        self.assertEqual(pressure, _make_test_cube("pressure", "Pa", data="pressure"))
        """test to check that the saturation_vapour_pressure
        method returns a cube with answers calculated to be correct
        """
        temp = _make_test_cube("temperature", "K")
        pressure = _make_test_cube("pressure", "hPa", data="pressure")
        expected_data = np.array([[470.314, 470.314], [1960.136,  1960.136]])
        result = saturation_vapour_pressure_goff_gratch(temp, pressure)
        self.assertArrayAlmostEqual(result.data, expected_data, decimal=3)
        self.assertEqual(result.units, Unit('Pa'))

    def test_non_modification(self):
        """not sure if I need this."""
        temp_data = np.array([[-2, 5], [10, 20]])
        temp = _make_test_cube("temperature", "celsius", data=temp_data)
        pressure = _make_test_cube("pressure", "Pa", data="pressure")
        result = saturation_vapour_pressure_goff_gratch(temp, pressure)
        self.assertEqual(temp, _make_test_cube("temperature", "celsius", data=temp_data))
        self.assertEqual(pressure, _make_test_cube("pressure", "Pa", data="pressure"))


class Test_calculate_svp_simple(IrisTest):
    """saturation_vapour_press == svp"""
    def test_basic(self):
        """test to check that the saturation_vapour_pressure
        method returns a cube with answers calculated to be correct
        """
        temp = _make_test_cube("temperature", "K")
        expected_data = np.array([[470.015, 470.015], [2259.917,  2259.917]])
        result = saturation_vapour_pressure_simple(temp)
        self.assertArrayAlmostEqual(result.data, expected_data, decimal=3)
        self.assertEqual(result.units, Unit('Pa'))


class Test_calculate_humidity_ratio_fm_rh(IrisTest):
    """Checks on method calculate_humidity_ratio_fm_rh"""
    def test_basic(self):
        """Check basic functionality"""
        temperature = _make_test_cube("temperature", "K")
        rel_humidity = _make_test_cube("relative humidity", 1, data="relative_humidity")
        pressure = _make_test_cube("pressure", "hPa", data="pressure")
        result = humidity_ratio_fm_rh(temperature, rel_humidity, pressure)
        expected_data = np.array([[0.003, 0.003 ],
                                  [0.012, 0.012]])
        self.assertEqual(result.units, Unit(1))
        self.assertArrayAlmostEqual(result.data ,expected_data, decimal=3)
        
class Test_calculate_humidity_ratio_fm_wb(IrisTest):
    """Checks on method calculate_humidity_ratio_fm_wb"""
    def test_basic(self):
        temperature = _make_test_cube("temperature", "K")
        wet_bulb_temp = _make_test_cube("wet_bulb_temperature", "K", data="wet_bulb_temp")
        pressure = _make_test_cube("pressure", "hPa", data="pressure")
        expected_data = np.array([[ 0.003, 0.003],
                                  [ 0.006, 0.006]])
        result = humidity_ratio_fm_wb(temperature, wet_bulb_temp, pressure)
        self.assertEqual(result.units, Unit(1))
        self.assertArrayAlmostEqual(result.data ,expected_data, decimal=3)


class Test_wet_bulb(IrisTest):
    def test_basic(self):
        """Given a default value of 100% humidity check that the wet-bulb
        temperature is the same as the dry bulb temperature"""
        temperature = _make_test_cube("temperature", "K")
        rel_humidity = _make_test_cube("relative humidity", 1, data="relative_humidity")
        pressure = _make_test_cube("pressure", "hPa", data="pressure")
        result = wet_bulb(temperature, rel_humidity, pressure)
        self.assertEqual(result.units, Unit("K"))
        self.assertArrayAlmostEqual(result.data ,temperature.data, decimal=3)
        
    def test_different_temperatures(self):
        """Check output for known 100% humidity at varying temperatures"""
        temperature_data = np.array([[220, 270], [320, 370]])
        temperature = _make_test_cube("temperature", "K", temperature_data)
        rel_humidity = _make_test_cube("relative humidity", 1, data="relative_humidity")
        pressure = _make_test_cube("pressure", "hPa", data="pressure")
        result = wet_bulb(temperature, rel_humidity, pressure)
        self.assertArrayAlmostEqual(result.data ,temperature.data, decimal=3)
        
    def test_different_humidities(self):
        """Check output for different Relative Humidities
        checked for a first guess against
        http://go.vaisala.com/humiditycalculator/5.0/"""
        temperature = _make_test_cube("temperature", "K")
        rh_data = np.array([[.99, .95], [.90, .80]])
        rel_humidity = _make_test_cube("relative humidity", 1, data=rh_data)
        pressure = _make_test_cube("pressure", "hPa", data="pressure")
        result = wet_bulb(temperature, rel_humidity, pressure)
        expected_data = np.array([[269.917, 269.587], [287.017,  284.015]])
        self.assertArrayAlmostEqual(result.data ,expected_data, decimal=3)
        