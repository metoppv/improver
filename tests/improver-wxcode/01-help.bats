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

@test "wxcode -h" {
  run improver wxcode -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver wxcode [-h] [--profile] [--profile_file PROFILE_FILE]
                       [--wxtree WXTREE]
                       INPUT_FILES [INPUT_FILES ...] OUTPUT_FILE

Calculate gridded weather symbol codes.
This plugin requires a specific set of input diagnostics, where data
may be in any units to which the thresholds given below can
be converted:
 - probability_of_number_of_lightning_flashes_per_unit_area_in_vicinity_above_threshold; thresholds: 0.0 (m-2)
 - probability_of_cloud_area_fraction_above_threshold; thresholds: 0.1875 (1), 0.8125 (1)
 - probability_of_rainfall_rate_above_threshold; thresholds: 0.03 (mm hr-1), 0.1 (mm hr-1), 1.0 (mm hr-1)
 - probability_of_lwe_snowfall_rate_above_threshold; thresholds: 0.1 (mm hr-1), 1.0 (mm hr-1)
 - probability_of_visibility_in_air_below_threshold; thresholds: 1000.0 (m), 5000.0 (m)
 - probability_of_low_type_cloud_area_fraction_above_threshold; thresholds: 0.85 (1)
 - probability_of_rainfall_rate_in_vicinity_above_threshold; thresholds: 0.1 (mm hr-1), 1.0 (mm hr-1)
 - probability_of_lwe_snowfall_rate_in_vicinity_above_threshold; thresholds: 0.1 (mm hr-1), 1.0 (mm hr-1)

 (probability_of_number_of_lightning_flashes data is optional)

 or for global data

 - probability_of_rainfall_rate_above_threshold; thresholds: 0.03 (mm hr-1), 0.1 (mm hr-1), 1.0 (mm hr-1)
 - probability_of_lwe_snowfall_rate_above_threshold; thresholds: 0.1 (mm hr-1), 1.0 (mm hr-1)
 - probability_of_cloud_area_fraction_above_threshold; thresholds: 0.1875 (1), 0.8125 (1)
 - probability_of_visibility_in_air_below_threshold; thresholds: 1000.0 (m), 5000.0 (m)
 - probability_of_low_type_cloud_area_fraction_above_threshold; thresholds: 0.85 (1)

positional arguments:
  INPUT_FILES           Paths to files containing the required input diagnostics.
  OUTPUT_FILE           The output path for the processed NetCDF.

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --wxtree WXTREE       Weather Code tree.
                        Choices are high_resolution or global.
                        Default=high_resolution.

__HELP__
  [[ "$output" == "$expected" ]]
}
