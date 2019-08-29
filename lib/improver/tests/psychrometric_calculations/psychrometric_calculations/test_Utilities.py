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
"""Unit tests for psychrometric_calculations utilities"""

import unittest

import numpy as np
from cf_units import Unit
from iris.tests import IrisTest

from improver.psychrometric_calculations.psychrometric_calculations import (
    Utilities)
from improver.tests.set_up_test_cubes import set_up_variable_cube


class Test_Utilities(IrisTest):

    """Test class for the Utilities tests, setting up cubes."""

    def setUp(self):
        """Set up the initial conditions for tests."""

        data = np.array([[260., 270., 280.]], dtype=np.float32)
        self.temperature = set_up_variable_cube(data)
        data = np.array([[60., 70., 80.]], dtype=np.float32)
        self.relative_humidity = set_up_variable_cube(
            data, name='relative_humidity', units='%')
        data = np.array([[1.E5, 9.9E4, 9.8E4]], dtype=np.float32)
        self.pressure = set_up_variable_cube(
            data, name='air_pressure', units='Pa')
        data = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        self.mixing_ratio = set_up_variable_cube(
            data, name='humidity_mixing_ratio', units='1')


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(Utilities())
        msg = '<Utilities>'
        self.assertEqual(result, msg)


class Test_specific_heat_of_moist_air(Test_Utilities):

    """Test calculations of the specific heat of moist air with an input cube
    of mixing ratios."""

    def test_basic(self):
        """Basic calculation of some moist air specific heat capacities."""
        expected = [[1089.499979, 1174.000017, 1258.50001]]
        result = Utilities.specific_heat_of_moist_air(self.mixing_ratio)
        self.assertArrayAlmostEqual(result.data, expected)
        self.assertEqual(result.units, Unit('J kg-1 K-1'))


class Test_latent_heat_of_condensation(Test_Utilities):

    """Test calculations of the latent heat of condensation with an input cube
    of air temperatures."""

    def test_basic(self):
        """Basic calculation of some latent heats of condensation."""
        expected = [[2531770.999107, 2508371.000223, 2484971.000223]]
        result = Utilities.latent_heat_of_condensation(self.temperature)
        self.assertArrayAlmostEqual(result.data, expected)
        self.assertEqual(result.units, Unit('J kg-1'))


class Test_calculate_enthalpy(Test_Utilities):

    """Test calculations of the enthalpy of air based upon the mixing ratio,
    specific heat, latent heat and temperature input cubes."""

    def test_basic(self):
        """Basic calculation of some enthalpies."""
        specific_heat = self.pressure.copy(data=[[1089.5, 1174., 1258.5]])
        specific_heat.units = 'J kg-1 K-1'
        latent_heat = self.pressure.copy(data=[[2531771., 2508371., 2484971.]])
        latent_heat.units = 'J kg-1'

        expected = [[536447.103773,  818654.207476, 1097871.329623]]
        result = Utilities.calculate_enthalpy(
            self.mixing_ratio, specific_heat, latent_heat, self.temperature)

        self.assertArrayAlmostEqual(result.data, expected)
        self.assertEqual(result.units, Unit('J kg-1'))


class Test_calculate_d_enthalpy_dt(Test_Utilities):

    """Test calculations of the enthalpy gradient with respect to temperature,
    based upon the mixing ratio, specific heat, latent heat and temperature
    input cubes."""

    def test_basic(self):
        """Basic calculation of some enthalpy gradients."""
        specific_heat = self.pressure.copy(data=[[1089.5, 1174., 1258.5]])
        specific_heat.units = 'J kg-1 K-1'
        latent_heat = self.pressure.copy(data=[[2531771., 2508371., 2484971.]])
        latent_heat.units = 'J kg-1'

        expected = [[21631.198581, 38569.575046, 52448.138051]]
        result = Utilities.calculate_d_enthalpy_dt(
            self.mixing_ratio, specific_heat, latent_heat, self.temperature)

        self.assertArrayAlmostEqual(result.data, expected)
        self.assertEqual(result.units, Unit('J kg-1 K-1'))


class Test_saturation_vapour_pressure_goff_gratch(Test_Utilities):

    """Test calculations of the saturated vapour pressure using the Goff-Gratch
    method."""

    def test_basic(self):
        """Basic calculation of some saturated vapour pressures."""
        result = Utilities.saturation_vapour_pressure_goff_gratch(
            self.temperature)
        expected = [[195.6419, 469.67078, 990.9421]]

        np.testing.assert_allclose(result.data, expected, rtol=1.e-5)
        self.assertEqual(result.units, Unit('Pa'))


if __name__ == '__main__':
    unittest.main()
