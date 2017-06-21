#!/usr/bin/env bats
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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

@test "nbhood -h" {
  run improver nbhood -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-ecc [-h]
                    [[--reordering | --raw_forecast_filepath RAW_FORECAST_FILE | --random_ordering]
                    [--rebadging | --member_numbers MEMBER_NUMBERS]
                    [--no_of_percentiles NUMBER_OF_PERCENTILES]
                    [--sampling_method [PERCENTILE_SAMPLING_METHOD]]
                    INPUT_FILE OUTPUT_FILE

Apply the requested neighbourhood method via the NeighbourhoodProcessing
plugin to a file with one cube.

positional arguments:
  INPUT_FILE            A path to an input NetCDF file to be processed
  OUTPUT_FILE           The output path for the processed NetCDF

optional arguments:
  -h, --help            show this help message and exit
  --reordering          The option used to create ensemble members from
                        percentiles by reordering the input percentiles based
                        on the order of the raw ensemble forecast.
  --raw_forecast_filepath RAW_FORECAST_FILE
                        A path to an raw forecast NetCDF file to be processed.
  --random_ordering     Decide whether or not to use random ordering within
                        the ensemble reordering step.
  --rebadging           The option used to create ensemble members from
                        percentiles by rebadging the input percentiles.
  --member_numbers MEMBER_NUMBERS
                        A list of ensemble member numbers to use when
                        rebadging the percentiles into members.
  --no_of_percentiles NUMBER_OF_PERCENTILES
                        The number of percentiles to be generated.
  --sampling_method [PERCENTILE_SAMPLING_METHOD]
                        Method to be used for generating the list of
                        percentiles with forecasts generated at each
                        percentile. "quantile" is the default option.
__HELP__
  [[ "$output" == "$expected" ]]
}
