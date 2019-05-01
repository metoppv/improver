#!/usr/bin/env bats

@test "generate-landmask-ancillary -h" {
  run improver generate-landmask-ancillary -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver generate-landmask-ancillary [-h] [--profile]
                                            [--profile_file PROFILE_FILE]
                                            [--force]
                                            INPUT_FILE_STANDARD OUTPUT_FILE

Read the input landmask, and correct to boolean values.

positional arguments:
  INPUT_FILE_STANDARD   A path to an input NetCDF file to be processed
  OUTPUT_FILE           The output path for the processed NetCDF

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --force               If True, ancillaries will be generated even if doing
                        so will overwrite existing files.
__HELP__
  [[ "$output" == "$expected" ]]
}
