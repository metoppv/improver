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

@test "temp-lapse-rate -h" {
  run improver temp-lapse-rate -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-temp-lapse-rate [-h] [--profile] [--profile_file PROFILE_FILE]
                                [--orography_filepath INPUT_OROGRAPHY_FILE]
                                [--land_sea_mask_filepath LAND_SEA_MASK_FILE]
                                [--max_height_diff MAX_HEIGHT_DIFF]
                                [--nbhood_radius NBHOOD_RADIUS]
                                [--max_lapse_rate MAX_LAPSE_RATE]
                                [--min_lapse_rate MIN_LAPSE_RATE]
                                [--return_dalr]
                                INPUT_TEMPERATURE_FILE OUTPUT_FILE

Calculate temperature lapse rates in units of K m-1 over a given orography
grid.

positional arguments:
  INPUT_TEMPERATURE_FILE
                        A path to an input NetCDF temperature file tobe
                        processed.
  OUTPUT_FILE           The output path for the processed temperature lapse
                        rates NetCDF.

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --orography_filepath INPUT_OROGRAPHY_FILE
                        A path to an input NetCDF orography file.
  --land_sea_mask_filepath LAND_SEA_MASK_FILE
                        A path to an input NetCDF land/sea mask file.
  --max_height_diff MAX_HEIGHT_DIFF
                        Maximum allowable height difference between the
                        central point and points in the neighbourhood over
                        which the lapse rate will be calculated (metres).
  --nbhood_radius NBHOOD_RADIUS
                        Radius of neighbourhood around each point. The
                        neighbourhood will be a square array with side length
                        2*nbhood_radius + 1.
  --max_lapse_rate MAX_LAPSE_RATE
                        Maximum lapse rate allowed which must be provided in
                        units of K m-1. Default is -3*DALR
  --min_lapse_rate MIN_LAPSE_RATE
                        Minimum lapse rate allowed which must be provided in
                        units of K m-1. Default is the DALR
  --return_dalr         Flag to return a cube containing the dry adiabatic
                        lapse rate rather than calculating the true lapse
                        rate.
__HELP__
  [[ "$output" == "$expected" ]]
}
