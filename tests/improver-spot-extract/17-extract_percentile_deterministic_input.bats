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

@test "spot-extract try to extract percentile from deterministic input" {
  improver_check_skip_acceptance
  KGO="spot-extract/outputs/nearest_uk_temperatures.nc"

  # Run spot extract processing and check it passes.
  run improver spot-extract \
      "$IMPROVER_ACC_TEST_DIR/spot-extract/inputs/all_methods_uk.nc" \
      "$IMPROVER_ACC_TEST_DIR/spot-extract/inputs/ukvx_temperature.nc" \
      "$TEST_DIR/output.nc" --extract_percentiles 50
  echo "status = ${status}"
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__TEXT__' || true
UserWarning: Diagnostic cube is not a known probabilistic type. The [50.0] percentile could not be extracted. Extracting data from the cube including any leading dimensions.
__TEXT__
  [[ "$output" =~ "$expected" ]]

  improver_check_recreate_kgo "output.nc" $KGO

  # Run nccmp to compare the output and kgo.
  improver_compare_output "$TEST_DIR/output.nc" \
      "$IMPROVER_ACC_TEST_DIR/$KGO"
}
