#!/usr/bin/env bats

@test "landmask generation no arguments" {
  run improver generate-landmask-ancillary
  [[ "$status" -eq 2 ]]
  read -d '' expected <<'__TEXT__' || true  
usage: improver generate-landmask-ancillary [-h] [--profile]
                                            [--profile_file PROFILE_FILE]
                                            [--force]
                                            INPUT_FILE_STANDARD OUTPUT_FILE
__TEXT__
  [[ "$output" =~ "$expected" ]]
}
