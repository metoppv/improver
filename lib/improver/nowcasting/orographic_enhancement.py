# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
"""
This module contains a plugin to calculate the enhancement of precipitation
over orography.
"""

import numpy as np
import iris

from improver.utilities.spatial import DifferenceBetweenAdjacentGridSquares
from improver.utilities.psychrometric_calculations import WetBulbTemperature
from improver.constants import R_WATER_VAPOUR


class OrographicEnhancement(object):
    """
    Class to calculate orographic enhancement from horizontal wind components,
    temperature and relative humidity.
    """

    def __init__(self):
        """Initialise the plugin"""
        self.orog_thresh_m = 20.
        self.rh_thres_ratio = 0.8
        self.vgradz_thresh = 0.0005

    @staticmethod
    def _orography_gradients(topography):
        """
        Checks topography height is in same units as spatial dimensions, then
        calculates dimensionless gradient in both directions.

        Args:
            topography (iris.cube.Cube):
                Height of topography above sea level

        Returns:
            (tuple): tuple containing:
                **gradx**: iris.cube.Cube of dimensionless topography gradient
                    in the positive x direction
                **grady**: iris.cube.Cube of dimensionless topography gradient
                    in the positive y direction
        """
        topography.coords(axis='x').convert_units(topography.units)
        topography.coords(axis='y').convert_units(topography.units)

        gradx, grady = DifferenceBetweenAdjacentGridSquares(
            gradient=True).process(topography)

        gradx.units = '1'
        grady.units = '1'

        return gradx, grady

    @staticmethod
    def _site_orogenh(temperature, humidity, svp, vgradz, mask):
        """
        Calculate precipitation enhancement over orography at each site using:

            orogenh = ((humidity * svp * vgradz) / 
                       (R_WATER_VAPOUR * temperature)) * 60 * 60

        Args:
            temperature (iris.cube.Cube):
                Temperature at top of boundary layer (K)
            humidity (iris.cube.Cube):
                Relative humidity at top of boundary layer (fraction)
            svp (iris.cube.Cube):
                Saturation vapour pressure at top of boundary layer (Pa)
            vgradz (np.ndarray):
                2D array of v.gradz in m s-1, matching input cube data shape
            mask (np.ndarray):
                Boolean mask representing conditions for calculating
                enhancement.  Where mask is True, set orogenh to zero.

        Returns:
            site_orogenh (np.ndarray):
                Orographic enhancement values in mm/h
        """
        site_orogenh = np.zeros(temperature.data.shape)

        prefactor = 3600./R_WATER_VAPOUR
        numerator = np.multiply(humidity.data, svp.data)
        numerator = np.multiply(numerator, vgradz)
        site_orogenh[~mask] = prefactor * np.divide(numerator[~mask],
                                                    temperature.data[~mask])
        return np.where(site_orogenh > 0, site_orogenh, 0)

    @staticmethod
    def _include_upstream_component(site_orogenh, uwind, vwind):
        """
        Adds upstream component to site orographic enhancement TODO

http://fcm9/projects/PostProc/browser/PostProc/trunk/blending/steps_core_orogenh.cpp#L1028

        """

        pass


    def process(self, temperature, humidity, pressure, uwind, vwind,
                topography):
        """
        Calculate precipitation enhancement over orography

        Args:
            temperature (iris.cube.Cube):
                Temperature at top of boundary layer
            humidity (iris.cube.Cube):
                Relative humidity at top of boundary layer
            pressure (iris.cube.Cube):
                Pressure at top of boundary layer
            uwind (iris.cube.Cube):
                Positive eastward wind vector component at top of boundary
                layer
            vwind (iris.cube.Cube):
                Positive northward wind vector component at top of boundary
                layer
            topography (iris.cube.Cube):
                Height of topography above sea level

        Returns:
            orogenh (iris.cube.Cube):
                Precipitation enhancement due to orography in mm/h

        Reference:
            Alpert, P. and Shafir, H., 1989: Meso-Gamma-Scale Distribution of
            Orographic Precipitation: Numerical Study and Comparison with
            Precipitation Derived from Radar Measurements.  Journal of Applied
            Meteorology, 28, 1105-1117.

        NOTE code here uses IMPROVER plugins where possible rather than the
        approximations in the STEPS code.  Therefore the outputs may not be
        identical to the old system.
        """
        # TODO check all the input cube coordinates match and are 2D



        # convert units of input cubes
        temperature.convert_units('kelvin')
        pressure.convert_units('Pa')
        
        # TODO humidity needs to be a fraction (not %) - can I do this???

        topography.convert_units('m')

        uwind.convert_units('m s-1')
        vwind.convert_units('m s-1')

        # calculate orography gradients
        gradx, grady = self._orography_gradients(topography)

        # TODO get 3x3 average orography?

        # calculate v.gradZ
        vgradz = (np.multiply(gradx.data, uwind.data) +
                  np.multiply(grady.data, vwind.data))

        # calculate saturation vapour pressure using WetBulbTemperature plugin
        # functionality
        wbt = WetBulbTemperature()
        svp = wbt._pressure_correct_svp(
            wbt._lookup_svp(temperature), temperature, pressure)

        # generate mask defining where to calculate orographic enhancement
        mask = np.full(topography.shape, False, dtype=bool)
        mask = np.where(topography.data < self.orog_thresh_m, True, mask)
        mask = np.where(humidity.data < self.rh_thresh_ratio, True, mask)
        mask = np.where(vgradz < self.vgradz_thresh, True, mask)

        # calculate site-specific orographic enhancement using svp, relative
        # humidity and temperature
        site_orogenh_data = self._site_orogenh(
            temperature, humidity, svp, vgradz, mask)

        # integrate upstream component
        orogenh_data = self._include_upstream_component(site_orogenh, uwind, vwind)

        # TODO create cube containing final data in mm/h
        orogenh = None


        return orogenh
