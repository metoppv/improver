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

@test "standardise -h" {
  run improver standardise -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-standardise [-h] [--profile] [--profile_file PROFILE_FILE]
                            [--output_filepath OUTPUT_FILE]
                            [--target_grid_filepath TARGET_GRID]
                            [--fix_float64] [--check_float64] [--regrid]
                            [--nearest]
                            [--extrapolation_mode EXTRAPOLATION_MODE]
                            [--json_file JSON_FILE]
                            SOURCE_DATA

Standardise a source data cube. Options are to regrid with further options to
fix float64 data, change metatdata, use iris nearest and extrapolation modes
as part of the regridding process. In addition separate standalone checks can
be made for float64 data, and updating cube metadata

positional arguments:
  SOURCE_DATA           A cube of data that is to be standardised and
                        optionally, fixed for float64 data, regridded and meta
                        data changed

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --output_filepath OUTPUT_FILE
                        The output path for the processed NetCDF
  --target_grid_filepath TARGET_GRID
                        A cube containing the grid to which the source_data is
                        to be regridded.
  --fix_float64         Check and fix cube for float64 data. Without this
                        option a warning will be raised if float64 data is
                        found but no fix applied.
  --check_float64       Check the cube for float64 data. If float64 data is
                        found a warning will be raised but no fix applied.
  --regrid              regrid cube.....
  --nearest             If True, regridding will be performed using
                        iris.analysis.Nearest() instead of Linear().Use for
                        less continuous fields, e.g. precipitation.
  --extrapolation_mode EXTRAPOLATION_MODE
                        Mode to use for extrapolating data into regions beyond
                        the limits of the source_data domain. Modes
                        are:extrapolate - The extrapolation points will take
                        their value from the nearest source point. nan - The
                        extrapolation points will be be set to NaN. error - A
                        ValueError exception will be raised, notifying an
                        attempt to extrapolate. mask - The extrapolation
                        points will always be masked, even if the source data
                        is not a MaskedArray. nanmask - If the source data is
                        a MaskedArray the extrapolation points will be masked.
                        Otherwise they will be set to NaN. Defaults to
                        nanmask.
  --json_file JSON_FILE
                        Filename for the json file containing required changes
                        to the metadata. Defaults to None.
__HELP__
  [[ "$output" == "$expected" ]]
}
