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

@test "optical-flow basic" {
  improver_check_skip_acceptance
  KGO1="optical-flow/basic/ucomp_kgo.nc"
  KGO2="optical-flow/basic/vcomp_kgo.nc"

  # Run processing and check it passes
  run improver optical-flow \
    "$IMPROVER_ACC_TEST_DIR/optical-flow/basic/input1.nc" \
    "$IMPROVER_ACC_TEST_DIR/optical-flow/basic/input2.nc" \
    "$IMPROVER_ACC_TEST_DIR/optical-flow/basic/input3.nc" \
    --output_dir "$TEST_DIR"
  [[ "$status" -eq 0 ]]

  improver_check_recreate_kgo "ucomp_kgo.nc" $KGO1
  improver_check_recreate_kgo "vcomp_kgo.nc" $KGO2  

  UCOMP="20180410T0500Z-PT0000H00M-eastward_precipitation_advection.nc"
  VCOMP="20180410T0500Z-PT0000H00M-northward_precipitation_advection.nc"

  # Run nccmp to compare the output and kgo.
  improver_compare_output "$TEST_DIR/$UCOMP" \
      "$IMPROVER_ACC_TEST_DIR/$KGO1"
  improver_compare_output "$TEST_DIR/$VCOMP" \
      "$IMPROVER_ACC_TEST_DIR/$KGO2"
}
