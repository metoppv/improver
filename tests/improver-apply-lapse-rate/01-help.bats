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

@test "apply-lapse-rate -h" {
  run improver apply-lapse-rate -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__TEXT__' || true
usage: improver apply-lapse-rate [-h] [--profile]
                                 [--profile_file PROFILE_FILE]
                                 TEMPERATURE_FILEPATH LAPSE_RATE_FILEPATH
                                 SOURCE_OROG_FILE TARGET_OROG_FILE OUTPUT_FILE

Apply downscaling temperature adjustment using calculated lapse rate.

positional arguments:
  TEMPERATURE_FILEPATH  Full path to input temperature NetCDF file
  LAPSE_RATE_FILEPATH   Full path to input lapse rate NetCDF file
  SOURCE_OROG_FILE      Full path to NetCDF file containing the source model
                        orography
  TARGET_OROG_FILE      Full path to target orography NetCDF file (to which
                        temperature will be downscaled)
  OUTPUT_FILE           File name to write lapse rate adjusted temperature
                        data

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.

__TEXT__
  [[ "$output" =~ "$expected" ]]
}
