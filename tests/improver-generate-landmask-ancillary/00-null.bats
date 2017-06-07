#!/usr/bin/env bats

@test "landmask generation no arguments" {
  run improver generate-landmask-ancillary
  [[ "$status" -eq 2 ]]
  expected="usage: improver-generate-landmask-ancillary [-h] [--force]
                                            INPUT_FILE_STAGE OUTPUT_FILE
                                            STANDARD_GRID"
  [[ "$output" =~ "$expected" ]]
}
