#!/usr/bin/env bats

@test "generate-topography-bands-mask no arguments" {
  run improver generate-topography-bands-mask
  [[ "$status" -eq 2 ]]
  read -d '' expected <<'__TEXT__' || true
usage: improver-generate-topography-bands-mask [-h]
                                               [--input_filepath_landmask INPUT_FILE_LAND]
                                               [--force]
                                               [--thresholds_filepath THRESHOLDS_FILEPATH]
                                               INPUT_FILE_STANDARD_OROGRAPHY
                                               OUTPUT_FILE
__TEXT__
  [[ "$output" =~ "$expected" ]]
}
