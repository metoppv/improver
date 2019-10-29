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

@test "neighbour-finding no arguments" {
  run improver neighbour-finding
  [[ "$status" -eq 2 ]]
  read -d '' expected <<'__TEXT__' || true
usage: improver neighbour-finding [-h] [--profile]
                                  [--profile_file PROFILE_FILE]
                                  [--all_methods] [--land_constraint]
                                  [--minimum_dz]
                                  [--search_radius SEARCH_RADIUS]
                                  [--node_limit NODE_LIMIT]
                                  [--site_coordinate_system SITE_COORDINATE_SYSTEM]
                                  [--site_coordinate_options SITE_COORDINATE_OPTIONS]
                                  [--site_x_coordinate SITE_X_COORDINATE]
                                  [--site_y_coordinate SITE_Y_COORDINATE]
                                  SITE_LIST_FILEPATH OROGRAPHY_FILEPATH
                                  LANDMASK_FILEPATH OUTPUT_FILEPATH
__TEXT__
  [[ "$output" =~ "$expected" ]]
}
