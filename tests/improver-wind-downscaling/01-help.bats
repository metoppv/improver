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

@test "wind downscaling -h" {
  run improver wind-downscaling -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver wind-downscaling [-h] [--profile]
                                 [--profile_file PROFILE_FILE]
                                 [--output_height_level OUTPUT_HEIGHT_LEVEL]
                                 [--output_height_level_units OUTPUT_HEIGHT_LEVEL_UNITS]
                                 [--height_levels_filepath HEIGHT_LEVELS_FILE]
                                 [--veg_roughness_filepath VEGETATIVE_ROUGHNESS_LENGTH_FILE]
                                 WIND_SPEED_FILE AOS_FILE SIGMA_FILE
                                 TARGET_OROGRAPHY_FILE STANDARD_OROGRAPHY_FILE
                                 MODEL_RESOLUTION OUTPUT_FILE

Run wind downscaling to apply roughness correction and height correction to
wind fields (as described in Howard and Clark [2007]). All inputs must be on
the same standard grid

positional arguments:
  WIND_SPEED_FILE       Location of the wind speed on standard grid file. Any
                        units can be supplied.
  AOS_FILE              Location of model silhouette roughness file. Units of
                        field: dimensionless
  SIGMA_FILE            Location of standard deviation of model orography
                        height file. Units of field: m
  TARGET_OROGRAPHY_FILE
                        Location of target orography file to downscale fields
                        to.Units of field: m
  STANDARD_OROGRAPHY_FILE
                        Location of orography on standard grid file
                        (interpolated model orography. Units of field: m
  MODEL_RESOLUTION      Original resolution of model orography (before
                        interpolation to standard grid). Units of field: m
  OUTPUT_FILE           The output path for the processed NetCDF

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --output_height_level OUTPUT_HEIGHT_LEVEL
                        If only a single height level is desired as output
                        from wind-downscaling, this option can be used to
                        select the height level. If no units are provided with
                        the --output_height_level_units option, metres are
                        assumed.
  --output_height_level_units OUTPUT_HEIGHT_LEVEL_UNITS
                        If a single height level is selected as output using
                        the --output_height_level option, this additional
                        argument may be used to specify the units of the value
                        entered to select the level. e.g. hPa
  --height_levels_filepath HEIGHT_LEVELS_FILE
                        Location of file containing height levels coincident
                        with wind speed field.
  --veg_roughness_filepath VEGETATIVE_ROUGHNESS_LENGTH_FILE
                        Location of vegetative roughness length file. Units of
                        field: m
__HELP__
  [[ "$output" == "$expected" ]]
}
