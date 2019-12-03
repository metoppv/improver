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

@test "remainder -h" {
  run improver remainder -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver remainder [-h] [--profile] [--profile_file PROFILE_FILE]
                          [--operation OPERATION] [--new-name NEW_NAME]
                          [--metadata_jsonfile METADATA_JSONFILE]
                          [--warnings_on]
                          INPUT_FILENAMES [INPUT_FILENAMES ...] OUTPUT_FILE

Combine the input files into a single file using the requested operation. In
this case it needs to subtract the cubes from a cube of ones to find the
remainder by using the operation -

positional arguments:
  INPUT_FILENAMES       Paths to the input NetCDF files. Each input file
                        should be able to be loaded as a single iris.cube.Cube
                        instance. The resulting file metadata will be based on
                        the first file but its metadata can be overwritten via
                        the metadata_jsonfile option.
  OUTPUT_FILE           The output path for the processed NetCDF.

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --operation OPERATION
                        Operation to use in combining NetCDF datasets
                        Default=- i.e. subtract
  --new-name NEW_NAME   New name for the resulting dataset. Will default to
                        the name of the first dataset if not set.
  --metadata_jsonfile METADATA_JSONFILE
                        Filename for the json file containing information for
                        bounds expansion.
  --warnings_on         If warnings_on is set (i.e. True), Warning messages
                        where metadata do not match will be given.
                        Default=False

__HELP__
  [[ "$output" == "$expected" ]]
}
