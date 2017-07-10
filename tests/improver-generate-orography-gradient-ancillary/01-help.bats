#!/usr/bin/env bats

@test "generate-orography-gradient-ancillary -h" {
  run improver generate-orography-gradient -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-generate-orography-gradient [-h] [--force]
                                            INPUT_FILE_STANDARD OUTPUT_FILE

Read the input orography field, and calculate the gradient in x and y
directions.

positional arguments:
  INPUT_FILE_STANDARD  A path to an input NetCDF file to be processed
  OUTPUT_FILE          The output path for the processed NetCDF

optional arguments:
  -h, --help           show this help message and exit
  --force              If True, ancillaries will be generated even if doing so
                       will overwrite existing files.
__HELP__
  [[ "$output" == "$expected" ]]
}