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

@test "extract -h" {
  run improver extract -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-extract [-h] [--profile] [--profile_file PROFILE_FILE]
                        [--units UNITS [UNITS ...]] [--ignore-failure]
                        INPUT_FILE OUTPUT_FILE CONSTRAINTS [CONSTRAINTS ...]

Extracts subset of data from a single input file, subject to equality-based
constraints.

positional arguments:
  INPUT_FILE            File containing a dataset to extract from.
  OUTPUT_FILE           File to write the extracted dataset to.
  CONSTRAINTS           The constraint(s) to be applied. These must be of the
                        form "key=value", eg "threshold=1". Scalars, boolean
                        and string values are supported. Comma-separated lists
                        (eg "key=[value1,value2]") are supported. These comma-
                        separated lists can either extract all values
                        specified in the list or all values specified within a
                        range e.g. key=[value1:value2]. When a range is
                        specified, this is inclusive of the endpoints of the
                        range.

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --units UNITS [UNITS ...]
                        Optional: units of coordinate constraint(s) to be
                        applied, for use when the input coordinate units are
                        not ideal (eg for float equality). If used, this list
                        must match the CONSTRAINTS list in order and length
                        (with null values set to None).
  --ignore-failure      Option to ignore constraint match failure and return
                        the input cube.
__HELP__
  [[ "$output" == "$expected" ]]
}
