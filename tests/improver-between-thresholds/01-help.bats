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

@test "between-thresholds -h" {
  run improver between-thresholds -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver between-thresholds [-h] [--profile]
                                   [--profile_file PROFILE_FILE]
                                   [--threshold_units THRESHOLD_UNITS]
                                   INPUT_FILE OUTPUT_FILE THRESHOLD_RANGES

Calculates probabilities of occurrence between thresholds

positional arguments:
  INPUT_FILE            Path to NetCDF file containing probabilities above or
                        below thresholds
  OUTPUT_FILE           Path to NetCDF file to write probabilities of
                        occurrence between thresholds
  THRESHOLD_RANGES      Path to json file specifying threshold ranges

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --threshold_units THRESHOLD_UNITS
                        Units in which thresholds are specified
__HELP__
  [[ "$output" == "$expected" ]]
}
