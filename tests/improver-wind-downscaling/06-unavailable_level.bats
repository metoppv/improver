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

@test "wind downscaling wind_speed " {
  improver_check_skip_acceptance
  test_path="$IMPROVER_ACC_TEST_DIR/wind_downscaling/basic/"

  # Run wind downscaling processing and check it passes.
  run improver wind-downscaling "$test_path/input.nc" "$test_path/a_over_s.nc" \
      "$test_path/sigma.nc" "$test_path/highres_orog.nc" "$test_path/standard_orog.nc" \
      1500 "NO_OUTPUT_FILE" --output_height_level "9" --output_height_level_units "m"
  echo "status = ${status}"
  [[ "$status" -eq 1 ]]
  read -d '' expected <<'__TEXT__' || true
ValueError: Requested height level not found, no cube returned. Available height levels are:
[  5.00000000e+00   1.00000000e+01   2.00000000e+01   3.00000000e+01
   5.00000000e+01   7.50000000e+01   1.00000000e+02   1.50000000e+02
   2.00000000e+02   2.50000000e+02   3.00000000e+02   4.00000000e+02
   5.00000000e+02   6.00000000e+02   7.00000000e+02   8.00000000e+02
   1.00000000e+03   1.25000000e+03   1.50000000e+03   1.75000000e+03
   2.00000000e+03   2.25000000e+03   2.50000000e+03   2.75000000e+03
   3.00000000e+03   3.25000000e+03   3.50000000e+03   3.75000000e+03
   4.00000000e+03   4.50000000e+03   5.00000000e+03   5.50000000e+03
   6.00000000e+03]
in units of m
__TEXT__
  [[ "$output" =~ "$expected" ]]
}
