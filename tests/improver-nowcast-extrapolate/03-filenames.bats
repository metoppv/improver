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

@test "extrapolate basic" {
  improver_check_skip_acceptance
  KGO1="optical-flow/extrapolate/kgo1.nc"
  KGO2="optical-flow/extrapolate/kgo2.nc"

  UCOMP="$IMPROVER_ACC_TEST_DIR/optical-flow/basic/ucomp_kgo.nc"
  VCOMP="$IMPROVER_ACC_TEST_DIR/optical-flow/basic/vcomp_kgo.nc"
  INFILE="201804100500_radar_rainrate_composite_UK_regridded.nc"

  # Run processing and check it passes
  run improver nowcast-extrapolate \
    "$IMPROVER_ACC_TEST_DIR/optical-flow/basic/$INFILE" \
    --output_filepaths "$TEST_DIR/outfile1.nc" "$TEST_DIR/outfile2.nc" \
    --max_lead_time 30 \
    --eastward_advection "$UCOMP" \
    --northward_advection "$VCOMP"
  [[ "$status" -eq 0 ]]

  improver_check_recreate_kgo "outfile1.nc" $KGO1
  improver_check_recreate_kgo "outfile2.nc" $KGO2

  # Run nccmp to compare the output and kgo.
  improver_compare_output "$TEST_DIR/outfile1.nc" \
      "$IMPROVER_ACC_TEST_DIR/$KGO1"
  improver_compare_output "$TEST_DIR/outfile2.nc" \
      "$IMPROVER_ACC_TEST_DIR/$KGO2"
}
