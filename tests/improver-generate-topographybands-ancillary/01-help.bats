#!/usr/bin/env bats

@test "generate-topographybands-ancillary -h" {
  run improver generate-topographybands-ancillary -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-generate-topographybands-ancillary [-h] [--force]
                                                   INPUT_FILE_STAGE
                                                   INPUT_FILE_LAND OUTPUT_FILE
                                                   STANDARD_GRID

Read input orography and landmask fields. Mask data inside the bounds
specified in the THRESHOLD_DICT dictionary

positional arguments:
  INPUT_FILE_STAGE  A path to an input NetCDF file (from StaGE) containing the
                    orography to be processed.
  INPUT_FILE_LAND   A path to an input land mask file.
  OUTPUT_FILE       The output path for the processed NetCDF.
  STANDARD_GRID     The standard grid to process landmask on.

optional arguments:
  -h, --help        show this help message and exit
  --force           If True, ancillaries will be generated even if doing so
                    will overwrite existing files
__HELP__
  [[ "$output" == "$expected" ]]
}
