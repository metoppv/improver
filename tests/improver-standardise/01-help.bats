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

@test "standardise -h" {
  run improver standardise -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver standardise [-h] [--profile] [--profile_file PROFILE_FILE]
                            [--output_filepath OUTPUT_FILE]
                            [--target_grid_filepath TARGET_GRID]
                            [--regrid_mode {bilinear,nearest,nearest-with-mask}]
                            [--extrapolation_mode EXTRAPOLATION_MODE]
                            [--input_landmask_filepath INPUT_LANDMASK_FILE]
                            [--landmask_vicinity LANDMASK_VICINITY]
                            [--fix_float64] [--json_file JSON_FILE]
                            [--coords_to_remove COORDS_TO_REMOVE [COORDS_TO_REMOVE ...]]
                            [--new_name NEW_NAME] [--new_units NEW_UNITS]
                            SOURCE_DATA

Standardise a source data cube. Three main options are available; fixing
float64 data, regridding and updating metadata. If regridding then additional
options are available to use bilinear or nearest-neighbour (optionally with
land-mask awareness) modes. If only a source file is specified with no other
arguments, then an exception will be raised if float64 data are found on the
source.

positional arguments:
  SOURCE_DATA           A cube of data that is to be standardised and
                        optionally fixed for float64 data, regridded and meta
                        data changed

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --output_filepath OUTPUT_FILE
                        The output path for the processed NetCDF. If only a
                        source file is specified and no output file, then the
                        source will be checkedfor float64 data.
  --fix_float64         Check and fix cube for float64 data. Without this
                        option an exception will be raised if float64 data is
                        found but no fix applied.
  --json_file JSON_FILE
                        Filename for the json file containing required changes
                        that will be applied to the attributes. Defaults to
                        None.
  --coords_to_remove COORDS_TO_REMOVE [COORDS_TO_REMOVE ...]
                        List of names of scalar coordinates to be removed from
                        the non-standard input.
  --new_name NEW_NAME   New dataset name.
  --new_units NEW_UNITS
                        Units to convert to.

Regridding options:
  --target_grid_filepath TARGET_GRID
                        If specified then regridding of the source against the
                        target grid is enabled. If also using landmask-aware
                        regridding, then this must be land_binary_mask data.
  --regrid_mode {bilinear,nearest,nearest-with-mask}
                        Selects which regridding technique to use. Default
                        uses iris.analysis.Linear(); "nearest" uses Nearest()
                        (Use for less continuous fields, e.g. precipitation.);
                        "nearest-with-mask" ensures that target data are
                        sourced from points with the same mask value (Use for
                        coast-line-dependent variables like temperature).
  --extrapolation_mode EXTRAPOLATION_MODE
                        Mode to use for extrapolating data into regions beyond
                        the limits of the source_data domain. Refer to online
                        documentation for iris.analysis. Modes are:
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
  --input_landmask_filepath INPUT_LANDMASK_FILE
                        A path to a NetCDF file describing the
                        land_binary_mask on the source-grid if coastline-aware
                        regridding is required.
  --landmask_vicinity LANDMASK_VICINITY
                        Radius of vicinity to search for a coastline, in
                        metres. Default value; 25000 m
__HELP__
  [[ "$output" == "$expected" ]]
}
