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

@test "neighbour-finding" {
  improver_check_skip_acceptance
  KGO="neighbour-finding/outputs/nearest_global_kgo.nc"

  # Run cube extraction processing and check it passes.
  run improver neighbour-finding \
      "$IMPROVER_ACC_TEST_DIR/neighbour-finding/inputs/LAEA_grid_sites.json" \
      "$IMPROVER_ACC_TEST_DIR/neighbour-finding/inputs/global_orography.nc" \
      "$IMPROVER_ACC_TEST_DIR/neighbour-finding/inputs/global_landmask.nc" \
      "$TEST_DIR/output.nc" \
      --site_coordinate_system "LambertAzimuthalEqualArea" \
      --site_coordinate_options '{"central_latitude": 54.9, "central_longitude": -2.5, "false_easting": 0.0, "false_northing": 0.0, "globe": {"semimajor_axis": 6378137.0, "semiminor_axis": 6356752.314140356}}' \
      --site_x_coordinate 'projection_x_coordinate' \
      --site_y_coordinate 'projection_y_coordinate'
  [[ "$status" -eq 0 ]]

  # Run nccmp to compare the output and kgo.
  # Note this is a special case. The site coordinates are different, but the
  # data (neighbour indices and vertical displacements) should be identical
  # to the 07 test in which sites were defined with latitudes and longitudes.
  # For this reason we invoke nccmp here directly to use different options.
  run nccmp -dm "$TEST_DIR/output.nc" "$IMPROVER_ACC_TEST_DIR/$KGO"
}
