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

@test "estimate-emos-coefficients -h" {
  run improver estimate-emos-coefficients -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver estimate-emos-coefficients [-h] [--profile]
                                           [--profile_file PROFILE_FILE]
                                           [--historic_filepath HISTORIC_FILEPATH [HISTORIC_FILEPATH ...]]
                                           [--truth_filepath TRUTH_FILEPATH [TRUTH_FILEPATH ...]]
                                           [--combined_filepath COMBINED_FILEPATH [COMBINED_FILEPATH ...]]
                                           [--historic_forecast_identifier HISTORIC_FORECAST_IDENTIFIER]
                                           [--truth_identifier TRUTH_IDENTIFIER]
                                           [--units UNITS]
                                           [--predictor_of_mean PREDICTOR_OF_MEAN]
                                           [--max_iterations MAX_ITERATIONS]
                                           DISTRIBUTION CYCLETIME
                                           OUTPUT_FILEPATH

Estimate coefficients for Ensemble Model Output Statistics (EMOS), otherwise
known as Non-homogeneous Gaussian Regression (NGR). There are two methods for
inputting data into this CLI, either by providing the historic forecasts and
truth separately, or by providing a combined list of historic forecasts and
truths along with historic_forecast_identifier and truth_identifier arguments
to provide metadata that distinguishes between them.

positional arguments:
  DISTRIBUTION          The distribution that will be used for calibration.
                        This will be dependent upon the input phenomenon. This
                        has to be supported by the minimisation functions in
                        ContinuousRankedProbabilityScoreMinimisers.
  CYCLETIME             This denotes the cycle at which forecasts will be
                        calibrated using the calculated EMOS coefficients. The
                        validity time in the output coefficients cube will be
                        calculated relative to this cycletime. This cycletime
                        is in the format YYYYMMDDTHHMMZ.
  OUTPUT_FILEPATH       The output path for the processed NetCDF

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --historic_filepath HISTORIC_FILEPATH [HISTORIC_FILEPATH ...]
                        Paths to the input NetCDF files containing the
                        historic forecast(s) used for calibration. This must
                        be supplied with an associated truth filepath.
                        Specification of either the combined_filepath,
                        historic_forecast_identifier or the truth_identifier
                        is invalid with this argument.
  --truth_filepath TRUTH_FILEPATH [TRUTH_FILEPATH ...]
                        Paths to the input NetCDF files containing the
                        historic truth analyses used for calibration. This
                        must be supplied with an associated historic filepath.
                        Specification of either the combined_filepath,
                        historic_forecast_identifier or the truth_identifier
                        is invalid with this argument.
  --combined_filepath COMBINED_FILEPATH [COMBINED_FILEPATH ...]
                        Paths to the input NetCDF files containing both the
                        historic forecast(s) and truth analyses used for
                        calibration. This must be supplied with both the
                        historic_forecast_identifier and the truth_identifier.
                        Specification of either the historic_filepath or the
                        truth_filepath is invalid with this argument.
  --historic_forecast_identifier HISTORIC_FORECAST_IDENTIFIER
                        The path to a json file containing metadata
                        information that defines the historic forecast. This
                        must be supplied with both the combined_filepath and
                        the truth_identifier. Specification of either the
                        historic_filepathor the truth_filepath is invalid with
                        this argument. The intended contents is described in i
                        mprover.ensemble_calibration.ensemble_calibration_util
                        ities.SplitHistoricForecastAndTruth.
  --truth_identifier TRUTH_IDENTIFIER
                        The path to a json file containing metadata
                        information that defines the truth.This must be
                        supplied with both the combined_filepath and the
                        historic_forecast_identifier. Specification of either
                        the historic_filepath or the truth_filepath is invalid
                        with this argument. The intended contents is described
                        in improver.ensemble_calibration.ensemble_calibration_
                        utilities.SplitHistoricForecastAndTruth.
  --units UNITS         The units that calibration should be undertaken in.
                        The historical forecast and truth will be converted as
                        required.
  --predictor_of_mean PREDICTOR_OF_MEAN
                        String to specify the predictor used to calibrate the
                        forecast mean. Currently the ensemble mean ("mean")
                        and the ensemble realizations ("realizations") are
                        supported as options. Default: "mean".
  --max_iterations MAX_ITERATIONS
                        The maximum number of iterations allowed until the
                        minimisation has converged to a stable solution. If
                        the maximum number of iterations is reached, but the
                        minimisation has not yet converged to a stable
                        solution, then the available solution is used anyway,
                        and a warning is raised.This may be modified for
                        testing purposes but otherwise kept fixed. If the
                        predictor_of_mean is "realizations", then the number
                        of iterations may require increasing, as there will be
                        more coefficients to solve for.
__HELP__
  [[ "$output" == "$expected" ]]
}
