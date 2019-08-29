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

@test "apply-emos-coefficients using realizations rebadged as percentiles as input" {
  improver_check_skip_acceptance
  KGO="apply-emos-coefficients/percentiles/kgo.nc"

  # Run apply-emos-coefficients when percentiles are input as the current forecast.
  run improver apply-emos-coefficients \
      "$IMPROVER_ACC_TEST_DIR/apply-emos-coefficients/rebadged_percentiles/input.nc" \
      "$IMPROVER_ACC_TEST_DIR/estimate-emos-coefficients/gaussian/kgo.nc" \
      "$TEST_DIR/output.nc"
  [[ "$status" -eq 0 ]]

  improver_check_recreate_kgo "output.nc" $KGO

  # Run nccmp to compare the output calibrated realizations when the input
  # is percentiles that have been rebadged as realizations. The known good
  # output in this case is the same as when passing in percentiles directly,
  # apart from a difference in the coordinates, such that the percentile input
  # will have a percentile coordinate, whilst the rebadged percentile input
  # will result in a realization coordinate.
  # The "--exclude=realization" option indicates that the realization coordinate
  # should be ignored.
  # The "--exclude=percentile" option indicates that the percentile coordinate
  # should be ignored.
  # The "-t 1e-3" option indicates a specific absolute tolerance of 1e-3
  # that matches the tolerance used in the
  # improver_compare_output_lower_precision function.
  run nccmp --exclude=realization --exclude=percentile -dNs -t 1e-3 "$TEST_DIR/output.nc" \
      "$IMPROVER_ACC_TEST_DIR/$KGO"
  [[ "$status" -eq 0 ]]
  [[ "$output" =~ "are identical." ]]
}
