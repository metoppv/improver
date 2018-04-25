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

@test "nbhood-land-and-sea input mask output" --radius=20000 {
  TEST_DIR=$(mktemp -d)
  improver_check_skip_acceptance

  # Run neighbourhood processing and check it passes.
  run improver nbhood-land-and-sea "$IMPROVER_ACC_TEST_DIR/nbhood-land-and-sea/topographic_bands/input.nc" "$IMPROVER_ACC_TEST_DIR/nbhood-land-and-sea/topographic_bands/topographic_bands_any_surface.nc" "$TEST_DIR/output.nc" --radius=20000 --weights "$IMPROVER_ACC_TEST_DIR/nbhood-land-and-sea/topographic_bands/weights_land.nc"
  [[ "$status" -eq 1 ]]
  read -d '' expected <<'__TEXT__' || true
The topographic zones mask cube must have been
__TEXT__
  [[ "$output" =~ "$expected" ]]
  rmdir "$TEST_DIR"
}
