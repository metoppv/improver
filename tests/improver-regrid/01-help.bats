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

@test "regrid -h" {
  run improver regrid -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-regrid [-h] [--profile] [--profile_file PROFILE_FILE]
                       [--nearest] [--extrapolation_mode EXTRAPOLATION_MODE]
                       SOURCE_DATA TARGET_GRID OUTPUT_FILE

Regrid data from source_data on to the grid contained within target_grid using
iris.analysis.Linear() or optionally iris.analysis.Nearest()

positional arguments:
  SOURCE_DATA           A cube of data that is to be regridded onto the
                        target_grid.
  TARGET_GRID           A cube containing the grid to which the source_data is
                        to be regridded.
  OUTPUT_FILE           The output path for the processed NetCDF

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --nearest             If True, regridding will be performed using
                        iris.analysis.Nearest() instead of Linear(). Use for
                        less continuous fields, e.g. precipitation.
  --extrapolation_mode EXTRAPOLATION_MODE
                        Mode to use for extrapolating data into regions beyond
                        the limits of the source_data domain. Modes are:
                        extrapolate - The extrapolation points will take their
                        value from the nearest source point. nan - The
                        extrapolation points will be be set to NaN. error - A
                        ValueError exception will be raised, notifying an
                        attempt to extrapolate. mask - The extrapolation
                        points will always be masked, even if the source data
                        is not a MaskedArray. nanmask - If the source data is
                        a MaskedArray the extrapolation points will be masked.
                        Otherwise they will be set to NaN. Defaults to
                        nanmask.
__HELP__
  [[ "$output" == "$expected" ]]
}
