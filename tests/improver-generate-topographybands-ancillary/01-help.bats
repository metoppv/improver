#!/usr/bin/env bats

@test "generate-topographybands-ancillary -h" {
  run improver generate-topographybands-ancillary -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-generate-topographybands-ancillary [-h] [--force]
                                                   [--thresholds_filepath THRESHOLDS_FILEPATH]
                                                   INPUT_FILE_STANDARD_OROGRAPHY
                                                   INPUT_FILE_LAND OUTPUT_FILE

Read input orography and landmask fields. Return a a cube of masks, where each
mask excludes data below or equal to the lower threshold, and excludes data
above the upper threshold.

positional arguments:
  INPUT_FILE_STANDARD_OROGRAPHY
                        A path to an input NetCDF orography file to be
                        processed
  INPUT_FILE_LAND       A path to an input NetCDF land mask file to be
                        processed
  OUTPUT_FILE           The output path for the processed NetCDF.

optional arguments:
  -h, --help            show this help message and exit
  --force               If keyword is set (i.e. True), ancillaries will be
                        generated even if doing so will overwrite existing
                        files
  --thresholds_filepath THRESHOLDS_FILEPATH
                        The path to a json file which can be used to set the
                        number and size of topographic bounds. If unset a
                        default bounds dictionary will be used:{'land':
                        {'bounds' : [[-500, 0], [0, 50], [50, 100], [100,
                        150],[150, 200], [200, 250], [250, 300], [300, 400],
                        [400, 500], [500, 600]],'units': 'm'}}
__HELP__
  [[ "$output" == "$expected" ]]
}
