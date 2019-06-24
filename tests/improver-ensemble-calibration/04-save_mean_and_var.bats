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

@test "ensemble-calibration emos gaussian save_mean_and_variance" {
  improver_check_skip_acceptance
  KGO_MEAN="ensemble-calibration/gaussian/kgo_mean.nc"
  KGO_VARIANCE="ensemble-calibration/gaussian/kgo_variance.nc"

  # Run ensemble calibration with saving of mean and variance and check it passes.
  run improver ensemble-calibration 'ensemble model output statistics' 'K' \
      'gaussian' "$IMPROVER_ACC_TEST_DIR/ensemble-calibration/gaussian/input.nc" \
      "$IMPROVER_ACC_TEST_DIR/ensemble-calibration/gaussian/history/*.nc" \
      "$IMPROVER_ACC_TEST_DIR/ensemble-calibration/gaussian/truth/*.nc" \
      "$TEST_DIR/output.nc" \
      --save_mean "$TEST_DIR/mean.nc" \
      --save_variance "$TEST_DIR/variance.nc"
  [[ "$status" -eq 0 ]]

  improver_check_recreate_kgo "mean.nc" $KGO_MEAN
  improver_check_recreate_kgo "variance.nc" $KGO_VARIANCE

  # Run nccmp to compare the output mean and variance and check it passes.
  improver_compare_output_lower_precision "$TEST_DIR/mean.nc" \
      "$IMPROVER_ACC_TEST_DIR/$KGO_MEAN"
   improver_compare_output_lower_precision "$TEST_DIR/variance.nc" \
      "$IMPROVER_ACC_TEST_DIR/$KGO_VARIANCE"
}
