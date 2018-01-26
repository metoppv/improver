#!/usr/bin/env bats

@test "generate-topography-bands-mask -h" {
  run improver generate-topography-bands-mask -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-generate-topography-bands-mask [-h]
                                               [--input_filepath_landmask INPUT_FILE_LAND]
                                               [--force]
                                               [--thresholds_filepath THRESHOLDS_FILEPATH]
                                               INPUT_FILE_STANDARD_OROGRAPHY
                                               OUTPUT_FILE

Read input orography and landmask fields. Return a a cube of masks, where each
mask excludes data below or equal to the lower threshold, and excludes data
above the upper threshold.

positional arguments:
  INPUT_FILE_STANDARD_OROGRAPHY
                        A path to an input NetCDF orography file to be
                        processed
  OUTPUT_FILE           The output path for the processed NetCDF.

optional arguments:
  -h, --help            show this help message and exit
  --input_filepath_landmask INPUT_FILE_LAND
                        A path to an input NetCDF land mask file to be
                        processed. If provided, sea points will be masked and
                        set to the default fill value. If no land mask is
                        provided, sea points will be included in the
                        appropriate topographic band.
  --force               If keyword is set (i.e. True), ancillaries will be
                        generated even if doing so will overwrite existing
                        files
  --thresholds_filepath THRESHOLDS_FILEPATH
                        The path to a json file which can be used to set the
                        number and size of topographic bounds. If unset a
                        default bounds dictionary will be used, where the
                        primary key will be 'any_surface_type' unless a land
                        mask is provided in which case it will be 'land'. The
                        dictionary has the following form:
                        {'any_surface_type': {'bounds' : [[-500, 0], [0, 50],
                        [50, 100], [100, 150],[150, 200], [200, 250], [250,
                        300], [300, 400], [400, 500], [500, 600]],'units':
                        'm'}}
__HELP__
  [[ "$output" == "$expected" ]]
}
