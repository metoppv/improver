# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
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
SECONDS_IN_HOUR = 3600
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
