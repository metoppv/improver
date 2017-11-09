#!/usr/bin/env bats

@test "generate-topographybands-ancillary -h" {
  run improver generate-topographybands-ancillary -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-generate-topographybands-ancillary [-h] [--force]
                                                   INPUT_FILE_STANDARD_OROGRAPHY
                                                   INPUT_FILE_LAND OUTPUT_FILE

Read input orography and landmask fields. Mask data inside the bounds
specified in the THRESHOLD_DICT dictionary

positional arguments:
  INPUT_FILE_STANDARD_OROGRAPHY
                        A path to an input NetCDF orography file to be
                        processed
  INPUT_FILE_LAND       A path to an input NetCDF land mask file to be
                        processed
  OUTPUT_FILE           The output path for the processed NetCDF.

optional arguments:
  -h, --help            show this help message and exit
  --force               If True, ancillaries will be generated even if doing
                        so will overwrite existing files
  --thresholds THRESHOLD_DICT
                        The address of a json file which can be used to set
                        the number and size of topographic bounds                        
__HELP__
  [[ "$output" == "$expected" ]]
}
