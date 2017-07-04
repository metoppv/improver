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

@test "weighted-blending -h" {
  run improver weighted-blending -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-weighted-blending [-h] (--linear | --nonlinear)
                                  [--coord_exp_val COORD_EXPECTED_VALUES]
                                  [--slope LINEAR_SLOPE | --ynval LINEAR_END_POINT]
                                  [--y0val LINEAR_STARTING_POINT]
                                  [--cval NON_LINEAR_FACTOR]
                                  [--coord_adj COORD_ADJUSTMENT_FUNCTION]
                                  COORDINATE_TO_AVERAGE_OVER INPUT_FILE
                                  OUTPUT_FILE

Calculate the default weights to apply in weighted blending plugins using the
ChooseDefaultWeightsLinear or ChooseDefaultWeightsNonLinear plugins. Then
apply these weights to the cube using the BasicWeightedAverage plugin.
Required for ChooseDefaultWeightsLinear: y0val and ONE of slope, ynval.
Required for ChooseDefaultWeightsNonLinear: cval.

positional arguments:
  COORDINATE_TO_AVERAGE_OVER
                        The coordinate over which the blending will be
                        applied.
  INPUT_FILE            A path to an input NetCDF file to be processed.
  OUTPUT_FILE           The output path for the processed NetCDF.

optional arguments:
  -h, --help            show this help message and exit
  --linear              Option to use the ChooseDefaultWeightsLinear plugin.
  --nonlinear           Option to use the ChooseDefaultWeightsNonLinear
                        plugin.
  --coord_exp_val COORD_EXPECTED_VALUES
                        Optional string of expected coordinate points
                        seperated by , e.g. "1496289600, 1496293200"
  --coord_adj COORD_ADJUSTMENT_FUNCTION
                        Function to apply to the coordinate after the blending
                        has been applied.

linear weights options:
  Options for the linear weights calculation in ChooseDefaultWeightsLinear

  --slope LINEAR_SLOPE  The slope of the line used for choosing default linear
                        weights. Only one of ynval and slope may be set.
  --ynval LINEAR_END_POINT
                        The relative value of the weighting end point for
                        choosing default linear weights. Only one of ynval and
                        slope may be set.
  --y0val LINEAR_STARTING_POINT
                        The relative value of the weighting start point for
                        choosing default linear weights. This must be a
                        positive float. If not set, default values of
                        y0val=20.0 and ynval=2.0 are set.

nonlinear weights options:
  Options for the non-linear weights calculation in
  ChooseDefaultWeightsNonLinear

  --cval NON_LINEAR_FACTOR
                        Factor used to determine how skewed the non linear
                        weights will be. A value of 1 implies equal weighting.
                        If not set, a default value of cval=0.85 is set.
__HELP__
  [[ "$output" == "$expected" ]]
}
