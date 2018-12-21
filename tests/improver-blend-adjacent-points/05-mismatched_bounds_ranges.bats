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

@test "weighted-blending mismatched bounds ranges" {
  improver_check_skip_acceptance

  # Run triangular time blending with mismatched time bounds ranges and check it fails
  # Note this WON'T fail unless / until we call this to blend over 'time' ...
  run improver blend-adjacent-points 'forecast_period' '2' --units 'hours' \
      --width 3.0 'weighted_mean' \
      "$IMPROVER_ACC_TEST_DIR/blend-adjacent-points/mismatched_bounds_ranges/multiple_probabilities_rain_1H.nc" \
      "$IMPROVER_ACC_TEST_DIR/blend-adjacent-points/basic_mean/multiple_probabilities_rain_2H.nc" \
      "$IMPROVER_ACC_TEST_DIR/blend-adjacent-points/basic_mean/multiple_probabilities_rain_3H.nc" \
      "$TEST_DIR/output.nc"
  [[ "$status" -eq 1 ]]
  read -d '' expected <<'__TEXT__' || true
ValueError: Neighbourhood radius specified is less than zero.
__TEXT__
  [[ "$output" =~ "$expected" ]]
}
