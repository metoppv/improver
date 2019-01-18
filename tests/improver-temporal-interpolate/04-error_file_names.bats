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

@test "temporal-interpolate t0 t3 --times t1 t2 --output_files out_t1" {
  improver_check_skip_acceptance

  # Check temporal-interpolation error message for insufficient out files.
  run improver temporal-interpolate \
      "$IMPROVER_ACC_TEST_DIR/temporal-interpolate/basic/20190116T0900Z-PT0033H00M-temperature_at_screen_level.nc" \
      "$IMPROVER_ACC_TEST_DIR/temporal-interpolate/basic/20190116T1200Z-PT0036H00M-temperature_at_screen_level.nc" \
      --times 20190116T1000Z 20190116T1100Z \
      --output_files "$TEST_DIR/output_t1.nc"
  [[ "$status" -eq 1 ]]
  read -d '' expected <<'__TEXT__' || true
ValueError: Output_files do not match cubes created. 1 files given but 2 required.
__TEXT__
  [[ "$output" =~ "$expected" ]]
}
