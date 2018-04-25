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

@test "cube-combiner -h" {
  run improver combine -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-combine [-h] [--operation OPERATION] [--new-name NEW_NAME]
                        [--metadata_jsonfile METADATA_JSONFILE]
                        [--warnings_on]
                        INPUT_FILENAMES [INPUT_FILENAMES ...] OUTPUT_FILE

Combine the input files into a single file using the requested operation e.g.
+ - min max etc.

positional arguments:
  INPUT_FILENAMES       Paths to the input NetCDF files. Each input file
                        should be able to be loaded as a single iris.cube.Cube
                        instance. The resulting file metadata will be based on
                        the first file but its metadata can be overwritten via
                        the metadata_jsonfile option.
  OUTPUT_FILE           The output path for the processed NetCDF.

optional arguments:
  -h, --help            show this help message and exit
  --operation OPERATION
                        Operation to use in combining NetCDF datasets
                        Default=+ i.e. add
  --new-name NEW_NAME   New name for the resulting dataset. Will default to
                        the name of the first dataset if not set.
  --metadata_jsonfile METADATA_JSONFILE
                        Filename for the json file containing required changes
                        to the metadata. default=None
  --warnings_on         If warnings_on is set (i.e. True), Warning messages
                        where metadata do not match will be given.
                        Default=False
__HELP__
  [[ "$output" == "$expected" ]]
}
