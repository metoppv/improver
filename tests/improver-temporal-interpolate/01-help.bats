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

@test "temporal-interpolate -h" {
  run improver temporal-interpolate -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-temporal-interpolate [-h] [--profile]
                                     [--profile_file PROFILE_FILE]
                                     (--interval_in_mins INTERVAL_IN_MINS | --times TIMES [TIMES ...])
                                     --output_files OUTPUT_FILES
                                     [OUTPUT_FILES ...]
                                     DATA_T0 DATA_T1

Interpolate data between validity times

positional arguments:
  DATA_T0               Cube contain the data at the beginning of the period
  DATA_T1               Cube contain the data at the end of the period

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --interval_in_mins INTERVAL_IN_MINS
                        Specifies the interval in minutes at which to
                        interpolate between the two input cubes. A number of
                        minutes which does not divide up the interval equally
                        will raise an exception. If intervals_in_mins is set
                        then times can not be set.
  --times TIMES [TIMES ...]
                        Specifies the times in the format {YYYYMMDD}T{HHMM}Z
                        at which to interpolate between the two input
                        cubes.Where {YYYYMMDD} is year, month day and {HHMM}
                        is hour and minutes e.g 20180116T0100Z. More than one
                        timecan be provided separated by a space but if times
                        are set interval_in_mins can not be set
  --output_files OUTPUT_FILES [OUTPUT_FILES ...]
                        List of output files.
__HELP__
  [[ "$output" == "$expected" ]]
}
