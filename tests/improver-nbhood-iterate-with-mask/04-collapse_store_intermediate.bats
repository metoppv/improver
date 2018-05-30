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

@test "nbhood 'topographic_zone' input mask output" --radius=10000 --collapse_dimension {
  improver_check_skip_acceptance
  KGO1="nbhood-iterate-with-mask/basic_collapse_bands/kgo_collapsed.nc"
  KGO2="nbhood-iterate-with-mask/collapse_store_intermediate/kgo_pre_collapse.nc"

  # Run neighbourhood processing and check it passes.
  run improver nbhood-iterate-with-mask 'topographic_zone' \
     "$IMPROVER_ACC_TEST_DIR/nbhood-iterate-with-mask/basic_collapse_bands/thresholded_input.nc" \
     "$IMPROVER_ACC_TEST_DIR/nbhood-iterate-with-mask/basic_collapse_bands/orographic_bands_mask.nc" \
     "$TEST_DIR/output.nc" --radius 10000 --collapse_dimension \
     --weights_for_collapsing_dim "$IMPROVER_ACC_TEST_DIR/nbhood-iterate-with-mask/basic_collapse_bands/orographic_bands_weights.nc" \
     --intermediate_filepath "$TEST_DIR/output_pre_collapsed.nc"
  [[ "$status" -eq 0 ]]

  improver_check_recreate_kgo "output.nc" $KGO1
  improver_check_recreate_kgo "output_pre_collapsed.nc" $KGO2

  # Run nccmp to compare the output and kgo.
  improver_compare_output "$TEST_DIR/output.nc" \
      "$IMPROVER_ACC_TEST_DIR/$KGO1"
  improver_compare_output "$TEST_DIR/output_pre_collapsed.nc" \
      "$IMPROVER_ACC_TEST_DIR/$KGO2"
}