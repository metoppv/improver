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

@test "orographic-enhancement boundary height" {
  improver_check_skip_acceptance
  KGO_HI_RES="orographic_enhancement/boundary_height/kgo_hi_res.nc"
  KGO_STANDARD="orographic_enhancement/boundary_height/kgo_standard.nc"

  # Run orographic enhancement and check it passes
  run improver orographic-enhancement \
      "$IMPROVER_ACC_TEST_DIR/orographic_enhancement/basic/temperature.nc" \
      "$IMPROVER_ACC_TEST_DIR/orographic_enhancement/basic/humidity.nc" \
      "$IMPROVER_ACC_TEST_DIR/orographic_enhancement/basic/pressure.nc" \
      "$IMPROVER_ACC_TEST_DIR/orographic_enhancement/basic/wind_speed.nc" \
      "$IMPROVER_ACC_TEST_DIR/orographic_enhancement/basic/wind_direction.nc" \
      "$IMPROVER_ACC_TEST_DIR/orographic_enhancement/basic/constant_u1096_ng_dtm_height_orography_1km.nc" \
      "$TEST_DIR/output_hi_res.nc" \
      "$TEST_DIR/output_standard.nc" \
      --boundary_height_m=10.
  [[ "$status" -eq 0 ]]

  improver_check_recreate_kgo "output_hi_res.nc" $KGO_HI_RES
  improver_check_recreate_kgo "output_standard.nc" $KGO_STANDARD

  # Run nccmp to compare the output and kgo.
  improver_compare_output "$TEST_DIR/output_hi_res.nc" \
      "$IMPROVER_ACC_TEST_DIR/$KGO_HI_RES"
  improver_compare_output "$TEST_DIR/output_standard.nc" \
      "$IMPROVER_ACC_TEST_DIR/$KGO_STANDARD"
}
