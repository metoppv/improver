#!/usr/bin/env bats

@test "generate-topographybands-ancillary no arguments" {
  run improver generate-topographybands-ancillary
  [[ "$status" -eq 2 ]]
  expected="usage: improver-generate-topographybands-ancillary [-h] [--force]
                                                   INPUT_FILE_STAGE
                                                   INPUT_FILE_LAND OUTPUT_FILE
                                                   STANDARD_GRID"
  [[ "$output" =~ "$expected" ]]
}
