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
"""Module to contain generally useful constants."""

# Cube comparison tolerances
TIGHT_TOLERANCE = 1e-5
DEFAULT_TOLERANCE = 1e-4
LOOSE_TOLERANCE = 1e-3

# Real Missing Data Indicator
RMDI = -32767.0

# Default percentile boundaries to calculate at for IMPROVER.
DEFAULT_PERCENTILES = (0, 5, 10, 20, 25, 30, 40, 50,
                       60, 70, 75, 80, 90, 95, 100)

# 0 Kelvin in degrees C
ABSOLUTE_ZERO = -273.15
U_ABSOLUTE_ZERO = "celsius"

# Specific gas constant for dry air (J K-1 kg-1)
R_DRY_AIR = 287.0
U_R_DRY_AIR = "J K-1 kg-1"

# Specific gas constant for dry air per mole (J K-1 mol-1)
R_DRY_AIR_MOL = 8.314
U_R_DRY_AIR_MOL = "J K-1 mol-1"

# Specific gas constant for water vapour (J K-1 kg-1)
R_WATER_VAPOUR = 461.6
U_R_WATER_VAPOUR = "J K-1 kg-1"

# Specific heat capacity of dry air (J K-1 kg-1)
CP_DRY_AIR = 1005.0
U_CP_DRY_AIR = "J K-1 kg-1"

# Specific heat capacity of water vapour (J K-1 kg-1)
CP_WATER_VAPOUR = 1850.0
U_CP_WATER_VAPOUR = "J K-1 kg-1"

# Triple Point of Water (K)
TRIPLE_PT_WATER = 273.16
U_TRIPLE_PT_WATER = "K"

# Latent heat of condensation of water at 0C (J kg-1)
LH_CONDENSATION_WATER = 2.501E6
U_LH_CONDENSATION_WATER = "J kg-1"
# Molar mass of water vapour (kg mol-1)
WATER_VAPOUR_MOLAR_MASS = 0.01801
U_WATER_VAPOUR_MOLAR_MASS = "kg mol-1"
# Latent heat temperature dependence (J K-1 kg-1); from Met Office UM.
# Applied to temperatures in Celsius: LH = 2501 - 2.34E3 * T(celsius)
LATENT_HEAT_T_DEPENDENCE = 2.34E3
U_LATENT_HEAT_T_DEPENDENCE = "J K-1 kg-1"

# Repsilon, ratio of molecular weights of water and dry air (Earth)
EARTH_REPSILON = 0.62198
U_EARTH_REPSILON = "1"

# Dry Adiabatic Lapse Rate (DALR) in unit of K m-1
DALR = -0.0098
U_DALR = "K m-1"
