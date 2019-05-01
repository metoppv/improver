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

@test "nowcast-optical-flow no arguments" {
  run improver nowcast-optical-flow
  [[ "$status" -eq 2 ]]
  read -d '' expected <<'__TEXT__' || true
usage: improver nowcast-optical-flow [-h] [--profile]
                                     [--profile_file PROFILE_FILE]
                                     [--output_dir OUTPUT_DIR]
                                     [--nowcast_filepaths NOWCAST_FILEPATHS [NOWCAST_FILEPATHS ...]]
                                     [--orographic_enhancement_filepaths OROGRAPHIC_ENHANCEMENT_FILEPATHS [OROGRAPHIC_ENHANCEMENT_FILEPATHS ...]]
                                     [--json_file JSON_FILE]
                                     [--ofc_box_size OFC_BOX_SIZE]
                                     [--smart_smoothing_iterations SMART_SMOOTHING_ITERATIONS]
                                     [--extrapolate]
                                     [--max_lead_time MAX_LEAD_TIME]
                                     [--lead_time_interval LEAD_TIME_INTERVAL]
                                     INPUT_FILEPATHS INPUT_FILEPATHS
                                     INPUT_FILEPATHS
__TEXT__
  [[ "$output" =~ "$expected" ]]
}
