#!/usr/bin/env bats
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

@test "phase-change-level -h" {
  run improver phase-change-level -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver phase-change-level [-h] [--profile]
                                   [--profile_file PROFILE_FILE]
                                   PHASE_CHANGE WBT WBTI OROGRAPHY
                                   LAND_SEA_MASK OUTPUT_FILE

Calculate a continuous phase change level. This is an altitude at which
precipitation is expected to change phase, e.g. snow to sleet.

positional arguments:
  PHASE_CHANGE          The desired phase change for which the altitudeshould
                        be returned. Options are: 'snow-sleet', the melting of
                        snow to sleet; sleet-rain - the melting of sleet to
                        rain.
  WBT                   Path to a NetCDF file of wet bulb temperatures on
                        height levels.
  WBTI                  Path to a NetCDF file of wet bulb temperature
                        integrals calculated vertically downwards to height
                        levels.
  OROGRAPHY             Path to a NetCDF file containing the orography height
                        in m of the terrain over which the continuous phase
                        change level is being calculated.
  LAND_SEA_MASK         Path to a NetCDF file containing the binary land-sea
                        mask for the points for which the continuous phase
                        change level is being calculated. Land points are set
                        to 1, sea points are set to 0.
  OUTPUT_FILE           The output path for the processed NetCDF

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
__HELP__
  [[ "$output" == "$expected" ]]
}
