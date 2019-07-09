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

@test "ensemble-calibration no arguments" {
  run improver ensemble-calibration
  [[ "$status" -eq 2 ]]
  expected="usage: improver ensemble-calibration [-h] [--profile]
                                     [--profile_file PROFILE_FILE]
                                     [--predictor_of_mean CALIBRATE_MEAN_FLAG]
                                     [--save_mean MEAN_FILE]
                                     [--save_variance VARIANCE_FILE]
                                     [--num_realizations NUMBER_OF_REALIZATIONS]
                                     [--random_ordering]
                                     [--random_seed RANDOM_SEED]
                                     [--ecc_bounds_warning]
                                     [--max_iterations MAX_ITERATIONS]
                                     UNITS_TO_CALIBRATE_IN DISTRIBUTION
                                     INPUT_FILE HISTORIC_DATA_FILE
                                     TRUTH_DATA_FILE OUTPUT_FILE
"
  [[ "$output" =~ "$expected" ]]
}
