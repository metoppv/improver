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

@test "blend-adjacent-points -h" {
  run improver blend-adjacent-points -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver blend-adjacent-points [-h] [--profile]
                                      [--profile_file PROFILE_FILE] --units
                                      UNIT_STRING [--calendar CALENDAR]
                                      --width TRIANGLE_WIDTH
                                      [--blend_time_using_forecast_period]
                                      COORDINATE_TO_BLEND_OVER CENTRAL_POINT
                                      INPUT_FILES [INPUT_FILES ...]
                                      OUTPUT_FILE

Use the TriangularWeightedBlendAcrossAdjacentPoints to blend across a
particular coordinate. It does not collapse the coordinate, but instead blends
across adjacent points and puts the blended values back in the original
coordinate, with adjusted bounds.

positional arguments:
  COORDINATE_TO_BLEND_OVER
                        The coordinate over which the blending will be
                        applied.
  CENTRAL_POINT         Central point at which the output from the triangular
                        weighted blending will be calculated. This should be
                        in the units of the units argument that is passed in.
                        This value should be a point on the coordinate for
                        blending over.
  INPUT_FILES           Paths to input NetCDF files including and surrounding
                        the central_point.
  OUTPUT_FILE           The output path for the processed NetCDF.

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --units UNIT_STRING   Units of the central_point and width.
  --calendar CALENDAR   Calendar for parameter_unit if required.
                        Default=gregorian
  --width TRIANGLE_WIDTH
                        Width of the triangular weighting function used in the
                        blending, in the units of the units argument passed
                        in.
  --blend_time_using_forecast_period
                        Flag that we are blending over time but using the
                        forecast period coordinate as a proxy. Note this
                        should only be used when time and forecast_period
                        share a dimension: ie when all files provided are from
                        the same forecast cycle.
__HELP__
  [[ "$output" == "$expected" ]]
}
