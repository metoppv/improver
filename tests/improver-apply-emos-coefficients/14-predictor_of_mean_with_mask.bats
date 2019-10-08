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

@test "apply-emos-coefficients using non-default predictor 'realizations'" {
  improver_check_skip_acceptance
  if python -c "import statsmodels" &> /dev/null; then
      COEFFS="estimate-emos-coefficients/realizations/with_statsmodels_kgo.nc"
      KGO="apply-emos-coefficients/masked/with_statsmodels_kgo.nc"
  else
      COEFFS="estimate-emos-coefficients/realizations/without_statsmodels_kgo.nc"
      KGO="apply-emos-coefficients/masked/without_statsmodels_kgo.nc"
  fi

  # Apply EMOS coefficients to calibrate the input forecast
  # and check that the calibrated forecast matches the kgo.
  run improver apply-emos-coefficients \
      "$IMPROVER_ACC_TEST_DIR/apply-emos-coefficients/gaussian/input.nc" \
      "$IMPROVER_ACC_TEST_DIR/$COEFFS" \
      "$TEST_DIR/output.nc" \
      --predictor_of_mean 'realizations' --random_seed 0 \
      --landsea_mask "$IMPROVER_ACC_TEST_DIR/estimate-emos-coefficients/landmask.nc"
  [[ "$status" -eq 0 ]]

  improver_check_recreate_kgo "output.nc" $KGO

  # Run nccmp to compare the output and kgo realizations and check it passes.
  improver_compare_output_lower_precision "$TEST_DIR/output.nc" \
      "$IMPROVER_ACC_TEST_DIR/$KGO"
}
