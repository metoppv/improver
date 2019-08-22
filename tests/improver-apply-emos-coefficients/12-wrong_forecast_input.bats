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

. $IMPROVER_DIR/tests/lib/utils

@test "apply-emos-coefficients when input forecast cube is a coefficients cube" {
  improver_check_skip_acceptance
  KGO="ensemble-calibration/gaussian/input.nc"

  # Check it raises an error when there input forecast cube is a
  # coefficients cube.
  run improver apply-emos-coefficients \
  "$IMPROVER_ACC_TEST_DIR/estimate-emos-coefficients/gaussian/kgo.nc" \
  "$IMPROVER_ACC_TEST_DIR/estimate-emos-coefficients/gaussian/kgo.nc" \
      "$TEST_DIR/output.nc" --random_seed 0
  [[ "$status" -eq 1 ]]
  # Check for error
  read -d '' expected <<'__TEXT__' || true
ValueError: The current forecast cube has the name 'emos_coefficients'
__TEXT__

  [[ "$output" =~ "$expected" ]]

}
