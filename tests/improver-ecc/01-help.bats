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

@test "ecc -h" {
  run improver ecc -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-ecc [-h] [--profile] [--profile_file PROFILE_FILE]
                    [--no_of_percentiles NUMBER_OF_PERCENTILES]
                    [--sampling_method [PERCENTILE_SAMPLING_METHOD]]
                    (--reordering | --rebadging)
                    [--raw_forecast_filepath RAW_FORECAST_FILE]
                    [--random_ordering] [--random_seed RANDOM_SEED]
                    [--realization_numbers REALIZATION_NUMBERS]
                    INPUT_FILE OUTPUT_FILE

Apply Ensemble Copula Coupling to a file whose data can be loaded as a single
iris.cube.Cube.

positional arguments:
  INPUT_FILE            A path to an input NetCDF file to be processed.
  OUTPUT_FILE           The output path for the processed NetCDF.

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --no_of_percentiles NUMBER_OF_PERCENTILES
                        The number of percentiles to be generated. This is
                        also equal to the number of ensemble realizations that
                        will be generated.
  --sampling_method [PERCENTILE_SAMPLING_METHOD]
                        Method to be used for generating the list of
                        percentiles with forecasts generated at each
                        percentile. The options are "quantile" and "random".
                        "quantile" is the default option.
  --reordering          The option used to create ensemble realizations from
                        percentiles by reordering the input percentiles based
                        on the order of the raw ensemble forecast.
  --rebadging           The option used to create ensemble realizations from
                        percentiles by rebadging the input percentiles.

Reordering options:
  Options for reordering the input percentiles using the raw ensemble
  forecast as required to create ensemble realizations.

  --raw_forecast_filepath RAW_FORECAST_FILE
                        A path to an raw forecast NetCDF file to be processed.
                        This option is compulsory, if the reordering option is
                        selected.
  --random_ordering     Decide whether or not to use random ordering within
                        the ensemble reordering step.
  --random_seed RANDOM_SEED
                        Option to specify a value for the random seed for
                        testing purposes, otherwise, the default random seed
                        behaviour is utilised. The random seed is used in the
                        generation of the random numbers used for either the
                        random_ordering option to order the input percentiles
                        randomly, rather than use the ordering from the raw
                        ensemble, or for splitting tied values within the raw
                        ensemble, so that the values from the input
                        percentiles can be ordered to match the raw ensemble.

Rebadging options:
  Options for rebadging the input percentiles as ensemble realizations.

  --realization_numbers REALIZATION_NUMBERS
                        A list of ensemble realization numbers to use when
                        rebadging the percentiles into realizations.
__HELP__
  [[ "$output" == "$expected" ]]
}
