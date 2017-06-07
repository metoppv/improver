#!/usr/bin/env bats

@test "generate-landmask-ancillary -h" {
  run improver generate-landmask-ancillary -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-generate-landmask-ancillary [-h] [--force]
                                            INPUT_FILE_STAGE OUTPUT_FILE
                                            STANDARD_GRID

Read the model landmask, interpolate to the standard grid and correct to
boolean values.

positional arguments:
  INPUT_FILE_STAGE  A path to an input NetCDF file (from StaGE) to be
                    processed.
  OUTPUT_FILE       The output path for the processed NetCDF
  STANDARD_GRID     The standard grid to process landmask on.

optional arguments:
  -h, --help        show this help message and exit
  --force           If True, ancillaries will be generated even if doing so
                    will overwrite existing files.

__HELP__
  [[ "$output" == "$expected" ]]
}
