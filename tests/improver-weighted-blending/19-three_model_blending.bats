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

@test "weighted-blending model blending of three models using spatial weights" {
  improver_check_skip_acceptance
  KGO="weighted_blending/three_models/kgo.nc"

  # Run weighted blending with ukvx and nowcast data using spatial weights
  run improver weighted-blending 'model_configuration' \
      --spatial_weights_from_mask --wts_calc_method 'dict' \
      --weighting_coord forecast_period --cycletime 20190101T0300Z \
      --model_id_attr mosg__model_configuration \
      --wts_dict $IMPROVER_ACC_TEST_DIR/weighted_blending/three_models/blending-weights-preciprate.json \
      "$IMPROVER_ACC_TEST_DIR/weighted_blending/three_models/enukxhrly/20190101T0400Z-PT0004H00M-precip_rate.nc" \
      "$IMPROVER_ACC_TEST_DIR/weighted_blending/three_models/nc/20190101T0400Z-PT0001H00M-precip_rate.nc" \
      "$IMPROVER_ACC_TEST_DIR/weighted_blending/three_models/ukvx/20190101T0400Z-PT0002H00M-precip_rate.nc" \
      "$TEST_DIR/output.nc"
  [[ "$status" -eq 0 ]]

  improver_check_recreate_kgo "output.nc" $KGO

  # Run nccmp to compare the output and kgo.
  improver_compare_output "$TEST_DIR/output.nc" \
      "$IMPROVER_ACC_TEST_DIR/$KGO"
}
