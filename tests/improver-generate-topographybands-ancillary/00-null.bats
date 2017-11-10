#!/usr/bin/env bats

@test "generate-topographybands-ancillary no arguments" {
  run improver generate-topographybands-ancillary
  [[ "$status" -eq 2 ]]
  expected="usage: improver-generate-topographybands-ancillary [-h] [--force]
                                                   [--thresholds_filepath THRESHOLDS_FILEPATH]
                                                   INPUT_FILE_STANDARD_OROGRAPHY
                                                   INPUT_FILE_LAND OUTPUT_FILE
improver-generate-topographybands-ancillary: error: too few arguments"
  [[ "$output" =~ "$expected" ]]
}
