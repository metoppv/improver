#!/usr/bin/env bats
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
  run improver wxcode -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-wxcode [-h]
                       PROBABILITY_OF_RAINFALL_RATE
                       PROBABILITY_OF_LWE_SNOWFALL_RATE
                       PROBABILITY_OF_CLOUD_AREA_FRACTION
                       PROBABILITY_OF_VISIBILITY_IN_AIR
                       PROBABILITY_OF_CLOUD_AREA_FRACTION_BELOW_1000_FEET_ASL
                       PROBABILITY_OF_LWE_SNOWFALL_RATE_IN_VICINITY
                       PROBABILITY_OF_RAINFALL_RATE_IN_VICINITY OUTPUT_FILE

Calculate a cube of weather symbol codes.

positional arguments:
  PROBABILITY_OF_RAINFALL_RATE
                        File path to a cube of probability_of_rainfall_rate at
                        the points for which the weather symbols are being
                        calculated.
  PROBABILITY_OF_LWE_SNOWFALL_RATE
                        File path to a cube of
                        probability_of_lwe_snowfall_rate at the points for
                        which the weather symbols are being calculated.
  PROBABILITY_OF_CLOUD_AREA_FRACTION
                        File path to a cube of
                        probability_of_cloud_area_fraction at the points for
                        which the weather symbols are being calculated.
  PROBABILITY_OF_VISIBILITY_IN_AIR
                        File path to a cube of
                        probability_of_visibility_in_air at the points for
                        which the weather symbols are being calculated.
  PROBABILITY_OF_CLOUD_AREA_FRACTION_BELOW_1000_FEET_ASL
                        File path to a cube of probability_of_cloud_area_fract
                        ion_assuming_only_consider_surface_to_1000_feet_asl at
                        the points for which the weather symbols are being
                        calculated.
  PROBABILITY_OF_LWE_SNOWFALL_RATE_IN_VICINITY
                        File path to a cube of
                        probability_of_lwe_snowfall_rate_in_vicinity at the
                        points for which the weather symbols are being
                        calculated.
  PROBABILITY_OF_RAINFALL_RATE_IN_VICINITY
                        File path to a cube of
                        probability_of_rainfall_rate_in_vicinity at the points
                        for which the weather symbols are being calculated.
  OUTPUT_FILE           The output path for the processed NetCDF.

optional arguments:
  -h, --help            show this help message and exit

__HELP__
  [[ "$output" == "$expected" ]]
}
