#!/usr/bin/env bats

@test "generate-topographybands-ancillary no arguments" {
  run improver generate-topography-bands-mask
  [[ "$status" -eq 2 ]]
  read -d '' expected <<'__TEXT__' || true
usage: improver-generate-topography-bands-weights [-h] [--force]
                                                  [--thresholds_filepath THRESHOLDS_FILEPATH]
                                                  INPUT_FILE_STANDARD_OROGRAPHY
                                                  INPUT_FILE_LAND OUTPUT_FILE
__TEXT__
  [[ "$output" =~ "$expected" ]]
}
