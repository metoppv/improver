#!/usr/bin/env bats

@test "generate-topography-bands-weights -h" {
  run improver generate-topography-bands-weights -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-generate-topography-bands-weights [-h]
                                                  [--input_filepath_landmask INPUT_FILE_LAND]
                                                  [--force]
                                                  [--thresholds_filepath THRESHOLDS_FILEPATH]
                                                  INPUT_FILE_STANDARD_OROGRAPHY
                                                  OUTPUT_FILE

Read input orography and landmask fields. Return a a cube of topographic zone
weights to indicate where an orography point sits within the defined
topographic bands. If the orography point is in the centre of a topographic
band, then a single band will have a weight of 1.0. If the orography point is
at the edge of a topographic band, then the upper band will have a 0.5 weight
whilst the lower band will also have a 0.5 weight. Otherwise, the weight will
vary linearly between the centre of a topographic band and the edge.

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
                        provided, weights will be generated for sea points as
                        well as land. included in the appropriate topographic
                        band.
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
