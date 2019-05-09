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

@test "wind-gust-diagnostic -h" {
  run improver wind-gust-diagnostic -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver wind-gust-diagnostic [-h] [--profile]
                                     [--profile_file PROFILE_FILE]
                                     [--percentile_gust PERCENTILE_GUST]
                                     [--percentile_ws PERCENTILE_WIND_SPEED]
                                     INPUT_FILE_GUST INPUT_FILE_WINDSPEED
                                     OUTPUT_FILE

Calculate revised wind-gust data using a specified percentile of wind-gust
data and a specified percentile of wind-speed data through the
WindGustDiagnostic plugin. The wind-gust diagnostic will be the Max of the
specified percentile data.Currently Typical gusts is MAX(wind-gust(50th
percentile),wind-speed(95th percentile))and Extreme gust is MAX(wind-gust(95th
percentile),wind-speed(100th percentile)). If no percentile values are
supplied the code defaults to values for Typical gusts.

positional arguments:
  INPUT_FILE_GUST       A path to an input Wind Gust Percentile NetCDF file
  INPUT_FILE_WINDSPEED  A path to an input Wind Speed Percentile NetCDF file
  OUTPUT_FILE           The output path for the processed NetCDF

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --percentile_gust PERCENTILE_GUST
                        Percentile of wind-gust required. Default=50.0
  --percentile_ws PERCENTILE_WIND_SPEED
                        Percentile of wind-speed required. Default=95.0
__HELP__
  [[ "$output" == "$expected" ]]
}
