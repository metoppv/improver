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

@test "blend-adjacent-points -h" {
  run improver blend-adjacent-points -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-blend-adjacent-points [-h] [--profile]
                                      [--profile_file PROFILE_FILE]
                                      [--parameter_unit UNIT_STRING]
                                      [--calendar CALENDAR]
                                      COORDINATE_TO_BLEND_OVER
                                      WEIGHTED_BLEND_MODE TRIANGLE_WIDTH
                                      INPUT_FILE OUTPUT_FILE

Use the TriangularWeightedBlendAcrossAdjacentPoints to blend across a
particular coordinate. It does not collapse the coordinate, but instead blends
across adjacent points and puts the blending values back in the original
coordinate. Two different types of blending are possible, weighted_mean and
weighted_maximum

positional arguments:
  COORDINATE_TO_BLEND_OVER
                        The coordinate over which the blending will be
                        applied.
  WEIGHTED_BLEND_MODE   The method used in the weighted blend.
                        "weighted_mean": calculate a normal weighted mean
                        across the coordinate. "weighted_maximum": multiplies
                        the values in the coordinate by the weights, and then
                        takes the maximum.
  TRIANGLE_WIDTH        Width of the triangular weighting function used in the
                        blending, in the units of the parameter_unit passed
                        in.
  INPUT_FILE            A path to an input NetCDF file to be processed.
  OUTPUT_FILE           The output path for the processed NetCDF.

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --parameter_unit UNIT_STRING
                        Units for time coordinate. Default= hours since
                        1970-01-01 00:00:00.
  --calendar CALENDAR   Calendar for parameter_unit if required.
                        Default=gregorian
__HELP__
  [[ "$output" == "$expected" ]]
}
