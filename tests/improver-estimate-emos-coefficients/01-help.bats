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
                                           [--units UNITS]
                                           [--predictor_of_mean PREDICTOR_OF_MEAN]
                                           [--max_iterations MAX_ITERATIONS]
                                           DISTRIBUTION CYCLETIME
                                           HISTORIC_FILEPATH TRUTH_FILEPATH
                                           OUTPUT_FILEPATH

Estimate coefficients for Ensemble Model Output Statistics (EMOS), otherwise
known as Non-homogeneous Gaussian Regression (NGR)

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
  HISTORIC_FILEPATH     A path to an input NetCDF file containing the historic
                        forecast(s) used for calibration.
  TRUTH_FILEPATH        A path to an input NetCDF file containing the historic
                        truth analyses used for calibration.
  OUTPUT_FILEPATH       The output path for the processed NetCDF

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
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
                        and a warning is raised. If the predictor_of_mean is
                        "realizations", then the number of iterations may
                        require increasing, as there will be more coefficients
                        to solve for.
__HELP__
  [[ "$output" == "$expected" ]]
}
