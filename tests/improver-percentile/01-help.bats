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

@test "percentile -h" {
  run improver percentile -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-percentile [-h] [--profile] [--profile_file PROFILE_FILE]
                           [--coordinates COORDINATES_TO_COLLAPSE [COORDINATES_TO_COLLAPSE ...]]
                           [--ecc_bounds_warning]
                           [--percentiles PERCENTILES [PERCENTILES ...] |
                           --no-of-percentiles NUMBER_OF_PERCENTILES]
                           INPUT_FILE OUTPUT_FILE

Calculate percentiled data over a given coordinate by collapsing that
coordinate. Typically used to convert realization data into percentiled data,
but may calculate over any dimension coordinate. Alternatively, calling this
CLI with a dataset containing probabilities will convert those to percentiles
using the ensemble copula coupling plugin. If no particular percentiles are
given at which to calculate values and no 'number of percentiles' to calculate
are specified, the following defaults will be used: [0, 5, 10, 20, 25, 30, 40,
50, 60, 70, 75, 80, 90, 95, 100]

positional arguments:
  INPUT_FILE            A path to an input NetCDF file to be processed
  OUTPUT_FILE           The output path for the processed NetCDF

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --coordinates COORDINATES_TO_COLLAPSE [COORDINATES_TO_COLLAPSE ...]
                        Coordinate or coordinates over which to collapse data
                        and calculate percentiles; e.g. 'realization' or
                        'latitude longitude'. This argument must be provided
                        when collapsing a coordinate or coordinates to create
                        percentiles, but is redundant when converting
                        probabilities to percentiles and may be omitted. This
                        coordinate(s) will be removed and replaced by a
                        percentile coordinate.
  --ecc_bounds_warning  If True, where calculated percentiles are outside the
                        ECC bounds range, raise a warning rather than an
                        exception.
  --percentiles PERCENTILES [PERCENTILES ...]
                        Optional definition of percentiles at which to
                        calculate data, e.g. --percentiles 0 33.3 66.6 100
  --no-of-percentiles NUMBER_OF_PERCENTILES
                        Optional definition of the number of percentiles to be
                        generated, these distributed regularly with the aim of
                        dividing into blocks of equal probability.
__HELP__
  [[ "$output" == "$expected" ]]
}
