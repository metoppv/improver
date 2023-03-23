# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
DEFAULT_PERCENTILES = (0, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 100)

# Temporal constants
SECONDS_IN_MINUTE = 60
MINUTES_IN_HOUR = 60
HOURS_IN_DAY = 24
DAYS_IN_YEAR = 365

#: 0 Kelvin in degrees C
ABSOLUTE_ZERO = -273.15

#: Specific gas constant for dry air (J K-1 kg-1)
R_DRY_AIR = 287.0

#: Specific gas constant for dry air per mole (J K-1 mol-1)
R_DRY_AIR_MOL = 8.314

#: Specific gas constant for water vapour (J K-1 kg-1)
R_WATER_VAPOUR = 461.6

#: Specific heat capacity of dry air (J K-1 kg-1)
CP_DRY_AIR = 1005.0

#: Specific heat capacity of water vapour (J K-1 kg-1)
CP_WATER_VAPOUR = 1850.0

#: Triple Point of Water (K)
TRIPLE_PT_WATER = 273.16

#: Latent heat of condensation of water at 0C (J kg-1)
LH_CONDENSATION_WATER = 2.501e6

#: Molar mass of water vapour (kg mol-1)
WATER_VAPOUR_MOLAR_MASS = 0.01801

#: Latent heat temperature dependence (J K-1 kg-1); from Met Office UM.
#: Applied to temperatures in Celsius: :math:`LH = 2501 - 2.34 \times 10^3 \times T(celsius)`
LATENT_HEAT_T_DEPENDENCE = 2.34e3

#: Repsilon, ratio of molecular weights of water and dry air (Earth; unitless)
EARTH_REPSILON = 0.62198

#: Dry Adiabatic Lapse Rate (DALR; K m-1)
DALR = -0.0098

#: Environmental Lapse Rate (ELR; K m-1)
#: Also known as Standard Atmosphere Lapse Rate
ELR = -0.0065
