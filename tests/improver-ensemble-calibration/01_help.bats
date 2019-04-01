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

@test "ensemble-calibration -h" {
  run improver ensemble-calibration -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-ensemble-calibration [-h] [--profile]
                                     [--profile_file PROFILE_FILE]
                                     [--predictor_of_mean CALIBRATE_MEAN_FLAG]
                                     [--save_mean MEAN_FILE]
                                     [--save_variance VARIANCE_FILE]
                                     [--num_realizations NUMBER_OF_REALIZATIONS]
                                     [--random_ordering]
                                     [--random_seed RANDOM_SEED]
                                     [--ecc_bounds_warning]
                                     ENSEMBLE_CALIBRATION_METHOD
                                     UNITS_TO_CALIBRATE_IN DISTRIBUTION
                                     INPUT_FILE HISTORIC_DATA_FILE
                                     TRUTH_DATA_FILE OUTPUT_FILE

Apply the requested ensemble calibration method using the current forecast (to
be calibrated) in the form of realizations, probabilities, or percentiles,
historical forecasts in the form of realizations and historical truth data (to
use in calibration). The mean and variance output from the EnsembleCalibration
plugin can be written to an output file if required. If the current forecast
is supplied in the form of probabilities or percentiles, these are converted
to realizations prior to calibration. After calibration, the mean and variance
computed in the calibration are converted to match the format of the current
forecast i.e. if realizations are input, realizations are output, if
probabilities are input, probabilities are output, and if percentiles are
input, percentiles are output.If realizations are input, realizations are
regenerated using Ensemble Copula Coupling.

positional arguments:
  ENSEMBLE_CALIBRATION_METHOD
                        The calibration method that will be applied. Supported
                        methods are: "emos" (ensemble model output statistics)
                        and "ngr" (nonhomogeneous gaussian regression).
  UNITS_TO_CALIBRATE_IN
                        The unit that calibration should be undertaken in. The
                        current forecast, historical forecast and truth will
                        be converted as required.
  DISTRIBUTION          The distribution that will be used for calibration.
                        This will be dependent upon the input phenomenon. This
                        has to be supported by the minimisation functions in
                        ContinuousRankedProbabilityScoreMinimisers.
  INPUT_FILE            A path to an input NetCDF file containing the current
                        forecast to be processed.The file provided could be in
                        the form of realizations, probabilities or
                        percentiles.
  HISTORIC_DATA_FILE    A path to an input NetCDF file containing the historic
                        forecast(s) used for calibration.The file provided
                        must be in the form of realizations.
  TRUTH_DATA_FILE       A path to an input NetCDF file containing the historic
                        truth analyses used for calibration.
  OUTPUT_FILE           The output path for the processed NetCDF

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --predictor_of_mean CALIBRATE_MEAN_FLAG
                        String to specify the input to calculate the
                        calibrated mean. Currently the ensemble mean ("mean")
                        and the ensemble realizations ("realizations") are
                        supported as the predictors. Default: "mean".
  --save_mean MEAN_FILE
                        Option to save the mean output from
                        EnsembleCalibration plugin. If used, a path to save
                        the output to must be provided.
  --save_variance VARIANCE_FILE
                        Option to save the variance output from
                        EnsembleCalibration plugin. If used, a path to save
                        the output to must be provided.
  --num_realizations NUMBER_OF_REALIZATIONS
                        Optional argument to specify the number of ensemble
                        realizations to produce. If the current forecast is
                        input as probabilities or percentiles then this
                        argument is used to create the requested number of
                        realizations. In addition, this argument is used to
                        construct the requested number of realizations from
                        the mean and variance output from the ensemble
                        calibration.Default will be the number of realizations
                        in the raw input file.
  --random_ordering     Option to reorder the post-processed forecasts
                        randomly. If not set, the ordering of the raw ensemble
                        is used.
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
__HELP__
  [[ "$output" == "$expected" ]]
}
