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

@test "orographic-enhancement help" {
  run improver orographic-enhancement -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__TEXT__' || true
usage: improver orographic-enhancement [-h] [--profile]
                                       [--profile_file PROFILE_FILE]
                                       [--boundary_height BOUNDARY_HEIGHT]
                                       [--boundary_height_units BOUNDARY_HEIGHT_UNITS]
                                       TEMPERATURE_FILEPATH HUMIDITY_FILEPATH
                                       PRESSURE_FILEPATH WINDSPEED_FILEPATH
                                       WINDDIR_FILEPATH OROGRAPHY_FILEPATH
                                       OUTPUT_DIR

Calculate orographic enhancement using the ResolveWindComponents() and
OrographicEnhancement() plugins. Outputs data on the high resolution orography
grid and regridded to the coarser resolution of the input diagnostic
variables.

positional arguments:
  TEMPERATURE_FILEPATH  Full path to input NetCDF file of temperature on
                        height levels
  HUMIDITY_FILEPATH     Full path to input NetCDF file of relative humidity on
                        height levels
  PRESSURE_FILEPATH     Full path to input NetCDF file of pressure on height
                        levels
  WINDSPEED_FILEPATH    Full path to input NetCDF file of wind speed on height
                        levels
  WINDDIR_FILEPATH      Full path to input NetCDF file of wind direction on
                        height levels
  OROGRAPHY_FILEPATH    Full path to input NetCDF high resolution orography
                        ancillary. This should be on the same or a finer
                        resolution grid than the input variables, and defines
                        the grid on which the orographic enhancement will be
                        calculated.
  OUTPUT_DIR            Directory to write output orographic enhancement files

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --boundary_height BOUNDARY_HEIGHT
                        Model height level to extract variables for
                        calculating orographic enhancement, as proxy for the
                        boundary layer.
  --boundary_height_units BOUNDARY_HEIGHT_UNITS
                        Units of the boundary height specified for extracting
                        model levels.
__TEXT__
  [[ "$output" =~ "$expected" ]]
}



