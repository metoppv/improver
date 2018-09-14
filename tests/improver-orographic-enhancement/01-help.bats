#!/usr/bin/env bats
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

@test "orographic-enhancement help" {
  run improver orographic-enhancement -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__TEXT__' || true
usage: improver-orographic-enhancement [-h] [--profile]
                                       [--profile_file PROFILE_FILE]
                                       [--boundary_height_metres BOUNDARY_HEIGHT_METRES]
                                       TEMPERATURE_FILEPATH HUMIDITY_FILEPATH
                                       PRESSURE_FILEPATH WINDSPEED_FILEPATH
                                       WINDDIR_FILEPATH OROGRAPHY_FILEPATH
                                       OUTPUT_HIGH_RES OUTPUT_STANDARD

Calculate orographic enhancement.

positional arguments:
  TEMPERATURE_FILEPATH  Full path to input NetCDF temperature file
  HUMIDITY_FILEPATH     Full path to input NetCDF relhumidity file
  PRESSURE_FILEPATH     Full path to input NetCDF pressure file
  WINDSPEED_FILEPATH    Full path to input NetCDF wind speed file
  WINDDIR_FILEPATH      Full path to input NetCDF wind direction file
  OROGRAPHY_FILEPATH    Full path to input NetCDF high resolution (1 km)
                        orography ancillary
  OUTPUT_HIGH_RES       Full path to write orographic enhancement file on high
                        resolution (1 km) grid
  OUTPUT_STANDARD       Full path to write orographic enhancement file on
                        standard grid

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --boundary_height_metres BOUNDARY_HEIGHT_METRES
                        Model height level to extract variables for
                        calculating orographic enhancement, as proxy for the
                        boundary layer.
__TEXT__
  [[ "$output" =~ "$expected" ]]
}



