#!/usr/bin/env bats

@test "generate-topographybands-ancillary no arguments" {
  run improver generate-topographybands-ancillary
  [[ "$status" -eq 2 ]]
  expected="usage: improver-generate-topographybands-ancillary [-h] [--force]
                                                   [--thresholds_filepath THRESHOLD_DICT]
                                                   INPUT_FILE_STANDARD_OROGRAPHY
                                                   INPUT_FILE_LAND OUTPUT_FILE"
  [[ "$output" =~ "$expected" ]]
}
