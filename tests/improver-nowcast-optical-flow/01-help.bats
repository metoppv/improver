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

@test "nowcast-optical-flow help" {
  run improver nowcast-optical-flow -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__TEXT__' || true
usage: improver-nowcast-optical-flow [-h] [--profile]
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

Calculate optical flow components from input fields and (optionally)
extrapolate to required lead times.

positional arguments:
  INPUT_FILEPATHS       Paths to the input radar files. There should be 3
                        input files at T, T-1 and T-2 from which to calculate
                        optical flow velocities. The files require a 'time'
                        coordinate on which they are sorted, so the order of
                        inputs does not matter.

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --output_dir OUTPUT_DIR
                        Directory to write all output files, or only advection
                        velocity components if NOWCAST_FILEPATHS is specified.
  --nowcast_filepaths NOWCAST_FILEPATHS [NOWCAST_FILEPATHS ...]
                        Optional list of full paths to output nowcast files.
                        Overrides OUTPUT_DIR. Ignored unless '--extrapolate'
                        is set.
  --orographic_enhancement_filepaths OROGRAPHIC_ENHANCEMENT_FILEPATHS [OROGRAPHIC_ENHANCEMENT_FILEPATHS ...]
                        List or wildcarded file specification to the input
                        orographic enhancement files. Orographic enhancement
                        files are compulsory for precipitation fields.
  --json_file JSON_FILE
                        Filename for the json file containing required changes
                        to the metadata. Information describing the intended
                        contents of the json file is available in
                        improver.utilities.cube_metadata.amend_metadata.Every
                        output cube will have the metadata_dict applied.
                        Defaults to None.
  --ofc_box_size OFC_BOX_SIZE
                        Size of square 'box' (in grid squares) within which to
                        solve the optical flow equations.
  --smart_smoothing_iterations SMART_SMOOTHING_ITERATIONS
                        Number of iterations to perform in enforcing
                        smoothness constraint for optical flow velocities.
  --extrapolate         Optional flag to advect current data forward to
                        specified lead times.
  --max_lead_time MAX_LEAD_TIME
                        Maximum lead time required (mins). Ignored unless '--
                        extrapolate' is set.
  --lead_time_interval LEAD_TIME_INTERVAL
                        Interval between required lead times (mins). Ignored
                        unless '--extrapolate' is set.
__TEXT__
  [[ "$output" =~ "$expected" ]]
}
