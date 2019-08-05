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

@test "extrapolate to create accumulations calculated with 5 minute fidelity" {
  improver_check_skip_acceptance
  KGO="nowcast-extrapolate/accumulation/kgo_5_minute_fidelity_30_minute_accum_period.nc"

  UCOMP="$IMPROVER_ACC_TEST_DIR/nowcast-optical-flow/basic/ucomp_kgo.nc"
  VCOMP="$IMPROVER_ACC_TEST_DIR/nowcast-optical-flow/basic/vcomp_kgo.nc"
  INFILE="201811031600_radar_rainrate_composite_UK_regridded.nc"
  OE1="20181103T1600Z-PT0003H00M-orographic_enhancement.nc"

  # Run processing and check it passes
  run improver nowcast-extrapolate \
    "$IMPROVER_ACC_TEST_DIR/nowcast-optical-flow/basic/$INFILE" \
    --output_dir "$TEST_DIR" --max_lead_time 30 \
    --eastward_advection "$UCOMP" \
    --northward_advection "$VCOMP" \
    --orographic_enhancement_filepaths \
    "$IMPROVER_ACC_TEST_DIR/nowcast-optical-flow/basic/$OE1" \
    --accumulation_fidelity 5 --accumulation_period 30 \
    --lead_time_interval 5
  [[ "$status" -eq 0 ]]

  T0="20181103T1630Z-PT0000H30M-lwe_thickness_of_precipitation_amount.nc"

  improver_check_recreate_kgo "$T0" $KGO

  # Run nccmp to compare the output and kgo.
  improver_compare_output "$TEST_DIR/$T0" \
      "$IMPROVER_ACC_TEST_DIR/$KGO"
}
