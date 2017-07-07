#!/usr/bin/env bats

@test "ensemble-calibration no arguments" {
  run improver ensemble-calibration
  [[ "$status" -eq 2 ]]
  expected="usage: improver-ensemble-calibration [-h]
                                     [--predictor_of_mean CALIBRATE_MEAN_FLAG]
                                     [--save_mean_variance MEAN_VARIANCE_FILE]
                                     [--num_members NUMBER_OF_MEMBERS]
                                     [--random_ordering]
                                     ENSEMBLE_CALIBRATION_METHOD
                                     UNITS_TO_CALIBRATE_IN DISTRIBUTION
                                     INPUT_FILE HISTORIC_DATA_FILE
                                     TRUTH_DATA_FILE OUTPUT_FILE
"
  [[ "$output" =~ "$expected" ]]
}
