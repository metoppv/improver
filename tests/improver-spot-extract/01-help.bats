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


@test "spot-extract -h" {
  run improver spot-extract -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver spot-extract [-h] [--profile] [--profile_file PROFILE_FILE]
                             [--land_constraint] [--minimum_dz]
                             [--extract_percentiles EXTRACT_PERCENTILES [EXTRACT_PERCENTILES ...]]
                             [--ecc_bounds_warning]
                             [--temperature_lapse_rate_filepath TEMPERATURE_LAPSE_RATE_FILEPATH]
                             [--grid_metadata_identifier GRID_METADATA_IDENTIFIER]
                             [--json_file JSON_FILE] [--suppress_warnings]
                             NEIGHBOUR_FILEPATH DIAGNOSTIC_FILEPATH
                             OUTPUT_FILEPATH

Extract diagnostic data from gridded fields for spot data sites. It is
possible to apply a temperature lapse rate adjustment to temperature data that
helps to account for differences between the spot sites real altitude and that
of the grid point from which the temperature data is extracted.

positional arguments:
  NEIGHBOUR_FILEPATH    Path to a NetCDF file of spot-data neighbours. This
                        file also contains the spot site information.
  DIAGNOSTIC_FILEPATH   Path to a NetCDF file containing the diagnostic data
                        to be extracted.
  OUTPUT_FILEPATH       The output path for the resulting NetCDF

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --ecc_bounds_warning  If True, where calculated percentiles are outside the
                        ECC bounds range, raise a warning rather than an
                        exception.

Neighbour finding method:
  If none of these options are set, the nearest grid point to a spot site
  will be used without any other constraints.

  --land_constraint     If set the neighbour cube will be interrogated for
                        grid point neighbours that were identified using a
                        land constraint. This means that the grid points
                        should be land points except for sites where none were
                        found within the search radius when the neighbour cube
                        was created. May be used with minimum_dz.
  --minimum_dz          If set the neighbour cube will be interrogated for
                        grid point neighbours that were identified using a
                        minimum height difference constraint. These are grid
                        points that were found to be the closest in altitude
                        to the spot site within the search radius defined when
                        the neighbour cube was created. May be used with
                        land_constraint.

Extract percentiles:
  Extract particular percentiles from probabilistic, percentile, or
  realization inputs. If deterministic input is provided a warning is raised
  and all leading dimensions are included in the returned spot-data cube.

  --extract_percentiles EXTRACT_PERCENTILES [EXTRACT_PERCENTILES ...]
                        If set to a percentile value or a list of percentile
                        values, data corresponding to those percentiles will
                        be returned. For example setting '--
                        extract_percentiles 25 50 75' will result in the 25th,
                        50th, and 75th percentiles being returned from a cube
                        of probabilities, percentiles, or realizations. Note
                        that for percentile inputs, the desired percentile(s)
                        must exist in the input cube.

Temperature lapse rate adjustment:
  --temperature_lapse_rate_filepath TEMPERATURE_LAPSE_RATE_FILEPATH
                        Filepath to a NetCDF file containing temperature lapse
                        rates. If this cube is provided, and a screen
                        temperature cube is being processed, the lapse rates
                        will be used to adjust the temperatures to better
                        represent each spot's site-altitude.

Metadata:
  --grid_metadata_identifier GRID_METADATA_IDENTIFIER
                        A string (or None) to identify attributes from the
                        input netCDF files that should be compared to ensure
                        that the data is compatible. Spot data works using
                        grid indices, so it is important that the grids are
                        matching or the data extracted may not match the
                        location of the spot data sites. The default is
                        'mosg__grid'. If set to None no check is made; this
                        can be used if the cubes are known to be appropriate
                        but lack relevant metadata.
  --json_file JSON_FILE
                        If provided, this JSON file can be used to modify the
                        metadata of the returned netCDF file. Defaults to
                        None.

Suppress Verbose output:
  --suppress_warnings   Suppress warning output. This option should only be
                        used if it is known that warnings will be generated
                        but they are not required.
__HELP__
  [[ "$output" == "$expected" ]]
}
