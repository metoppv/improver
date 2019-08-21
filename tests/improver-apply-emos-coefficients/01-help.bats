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

@test "apply-emos-coefficients -h" {
  run improver apply-emos-coefficients -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver apply-emos-coefficients [-h] [--profile]
                                        [--profile_file PROFILE_FILE]
                                        [--num_realizations NUMBER_OF_REALIZATIONS]
                                        [--random_ordering]
                                        [--random_seed RANDOM_SEED]
                                        [--ecc_bounds_warning]
                                        [--predictor_of_mean PREDICTOR_OF_MEAN]
                                        FORECAST_FILEPATH
                                        [COEFFICIENTS_FILEPATH]
                                        OUTPUT_FILEPATH

Apply coefficients for Ensemble Model Output Statistics (EMOS), otherwise
known as Non-homogeneous Gaussian Regression (NGR). The supported input
formats are realizations, probabilities and percentiles. The forecast will be
converted to realizations before applying the coefficients and then converted
back to match the input format.

positional arguments:
  FORECAST_FILEPATH     A path to an input NetCDF file containing the forecast
                        to be calibrated. The input format could be either
                        realizations, probabilities or percentiles.
  COEFFICIENTS_FILEPATH
                        (Optional) A path to an input NetCDF file containing
                        the coefficients used for calibration. If this file is
                        not provided the input forecast is returned unchanged.
  OUTPUT_FILEPATH       The output path for the processed NetCDF

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --num_realizations NUMBER_OF_REALIZATIONS
                        Optional argument to specify the number of ensemble
                        realizations to produce. If the current forecast is
                        input as probabilities or percentiles then this
                        argument is used to create the requested number of
                        realizations. In addition, this argument is used to
                        construct the requested number of realizations from
                        the mean and variance output after applying the EMOS
                        coefficients.Default will be the number of
                        realizations in the raw input file, if realizations
                        are provided as input, otherwise if the input format
                        is probabilities or percentiles, then an error will be
                        raised if no value is provided.
  --random_ordering     Option to reorder the post-processed forecasts
                        randomly. If not set, the ordering of the raw ensemble
                        is used. This option is only valid when the input
                        format is realizations.
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
  --ecc_bounds_warning  If True, where the percentiles exceed the ECC bounds
                        range, raise a warning rather than an exception. This
                        occurs when the current forecast is in the form of
                        probabilities and is converted to percentiles, as part
                        of converting the input probabilities into
                        realizations.
  --predictor_of_mean PREDICTOR_OF_MEAN
                        String to specify the predictor used to calibrate the
                        forecast mean. Currently the ensemble mean ("mean")
                        and the ensemble realizations ("realizations") are
                        supported as options. Default: "mean".
__HELP__
  [[ "$output" == "$expected" ]]
}
