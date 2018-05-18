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

. $IMPROVER_DIR/tests/lib/utils

@test "weighted-blending --linear --ynval --slope" {
  # Run blending with linear weights calculation but too many args: check it fails.
  run improver weighted-blending 'linear' 'time' 'weighted_mean' --ynval 1.0 --slope 0.0\
      "$IMPROVER_ACC_TEST_DIR/weighted_blending/basic_lin/multiple_probabilities_rain_*H.nc" \
      "NO_OUTPUT_FILE"
  [[ "${status}" -eq 2 ]]
  read -d '' expected <<'__TEXT__' || true
usage: improver-weighted-blending [-h] [--profile]
                                  [--profile_file PROFILE_FILE]
                                  [--coord_exp_val COORD_EXPECTED_VALUES]
                                  [--coordinate_unit UNIT_STRING]
                                  [--calendar CALENDAR]
                                  [--slope LINEAR_SLOPE | --ynval LINEAR_END_POINT]
                                  [--y0val LINEAR_STARTING_POINT]
                                  [--cval NON_LINEAR_FACTOR]
                                  [--coord_adj COORD_ADJUSTMENT_FUNCTION]
                                  [--wts_redistrib_method METHOD_TO_REDISTRIBUTE_WEIGHTS]
                                  [--cycletime CYCLETIME]
                                  [--coords_for_bounds_removal COORDS_FOR_BOUNDS_REMOVAL [COORDS_FOR_BOUNDS_REMOVAL ...]]
                                  WEIGHTS_CALCULATION_METHOD
                                  COORDINATE_TO_AVERAGE_OVER
                                  WEIGHTED_BLEND_MODE INPUT_FILES
                                  [INPUT_FILES ...] OUTPUT_FILE
improver-weighted-blending: error: argument --slope: not allowed with argument --ynval
__TEXT__
  [[ "$output" =~ "$expected" ]]
}
