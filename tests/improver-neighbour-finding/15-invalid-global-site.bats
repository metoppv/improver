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

@test "neighbour-finding with invalid global site" {
  improver_check_skip_acceptance
  KGO="neighbour-finding/outputs/nearest_global_invalid_site_kgo.nc"

  # Run cube extraction processing and check it passes.
  run improver neighbour-finding \
      "$IMPROVER_ACC_TEST_DIR/neighbour-finding/inputs/global_sites_invalid.json" \
      "$IMPROVER_ACC_TEST_DIR/neighbour-finding/inputs/global_orography.nc" \
      "$IMPROVER_ACC_TEST_DIR/neighbour-finding/inputs/global_landmask.nc" \
      "$TEST_DIR/output.nc"
  echo "status = ${status}"
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__TEXT__' || true
UserWarning: 1 spot sites fall outside the grid domain and will not be processed. These sites are:
{'altitude': 0.0, 'latitude': 125.0, 'longitude': 0.0}
__TEXT__
  [[ "$output" =~ "$expected" ]]

  improver_check_recreate_kgo "output.nc" $KGO

  # Run nccmp to compare the output and kgo.
  improver_compare_output "$TEST_DIR/output.nc" \
      "$IMPROVER_ACC_TEST_DIR/$KGO"
}
