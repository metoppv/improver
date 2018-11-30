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

@test "extrapolate basic no orographic enhancement" {
  improver_check_skip_acceptance
  KGO0="nowcast-extrapolate/extrapolate_no_orographic_enhancement/kgo0.nc"
  KGO1="nowcast-extrapolate/extrapolate_no_orographic_enhancement/kgo1.nc"
  KGO2="nowcast-extrapolate/extrapolate_no_orographic_enhancement/kgo2.nc"

  UCOMP="$IMPROVER_ACC_TEST_DIR/nowcast-optical-flow/basic/ucomp_kgo.nc"
  VCOMP="$IMPROVER_ACC_TEST_DIR/nowcast-optical-flow/basic/vcomp_kgo.nc"
  INFILE="201811031600_radar_rainrate_composite_UK_regridded.nc"

  # Run processing and check it passes when the input files are not
  # specifically precipitation. In this case, the input radar precipitation
  # files and fields have been renamed as rain to be able to test the
  # CLIs function as intended for a non-precipitation field.
  run improver nowcast-extrapolate \
    "$IMPROVER_ACC_TEST_DIR/nowcast-optical-flow/basic_no_orographic_enhancement/$INFILE" \
    --output_dir "$TEST_DIR" --max_lead_time 30 \
    --eastward_advection "$UCOMP" \
    --northward_advection "$VCOMP"
  [[ "$status" -eq 0 ]]

  T0="20181103T1600Z-PT0000H00M-lwe_rain_rate.nc"
  T1="20181103T1615Z-PT0000H15M-lwe_rain_rate.nc"
  T2="20181103T1630Z-PT0000H30M-lwe_rain_rate.nc"

  improver_check_recreate_kgo "$T0" $KGO0
  improver_check_recreate_kgo "$T1" $KGO1
  improver_check_recreate_kgo "$T2" $KGO2

  # Run nccmp to compare the output and kgo.
  improver_compare_output "$TEST_DIR/$T0" \
      "$IMPROVER_ACC_TEST_DIR/$KGO0"
  improver_compare_output "$TEST_DIR/$T1" \
      "$IMPROVER_ACC_TEST_DIR/$KGO1"
  improver_compare_output "$TEST_DIR/$T2" \
      "$IMPROVER_ACC_TEST_DIR/$KGO2"
}
