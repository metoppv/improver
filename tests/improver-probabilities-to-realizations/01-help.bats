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

@test "probabilities-to-realizations -h" {
  run improver probabilities-to-realizations -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-probabilities-to-realizations [-h] [--profile]
                                              [--profile_file PROFILE_FILE]
                                              [--no_of_realizations NUMBER_OF_REALIZATIONS]
                                              (--reordering | --rebadging)
                                              [--raw_forecast_filepath RAW_FORECAST_FILE]
                                              [--random_seed RANDOM_SEED]
                                              [--ecc_bounds_warning]
                                              INPUT_FILE OUTPUT_FILE

Convert a dataset containing probabilities into one containing ensemble
realizations.

positional arguments:
  INPUT_FILE            A path to an input NetCDF file to be processed
  OUTPUT_FILE           The output path for the processed NetCDF

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --no_of_realizations NUMBER_OF_REALIZATIONS
                        Optional definition of the number of ensemble
                        realizations to be generated. These are generated
                        through an intermediate percentile representation.
                        These percentiles will be distributed regularly with
                        the aim of dividing into blocks of equal probability.
                        If the reordering option is specified and the number
                        of realizations is not given then the number of
                        realizations is taken from the number of realizations
                        in the raw forecast NetCDF file.
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
  --random_seed RANDOM_SEED
                        Option to specify a value for the random seed for
                        testing purposes, otherwise, the default random seed
                        behaviour is utilised. The random seed is used in the
                        generation of the random numbers used for splitting
                        tied values within the raw ensemble, so that the
                        values from the input percentiles can be ordered to
                        match the raw ensemble.
  --ecc_bounds_warning  If True, where percentiles (calculated as an
                        intermediate output before realizations) exceed the
                        ECC bounds range, raise a warning rather than an
                        exception.
__HELP__
  [[ "$output" == "$expected" ]]
}
