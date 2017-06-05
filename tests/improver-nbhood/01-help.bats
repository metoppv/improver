#!/usr/bin/env bats

@test "nbhood -h" {
  run improver nbhood -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-nbhood [-h]
                       [--radius-in-km RADIUS | --radii-in-km-by-lead-time \
RADIUS_BY_LEAD_TIME RADIUS_BY_LEAD_TIME]
                       INPUT_FILE OUTPUT_FILE

Apply basic weighted circle smoothing via the BasicNeighbourhoodProcessing
plugin to a file with one cube.

positional arguments:
  INPUT_FILE            A path to an input NetCDF file to be processed
  OUTPUT_FILE           The output path for the processed NetCDF

optional arguments:
  -h, --help            show this help message and exit
  --radius-in-km RADIUS
                        The kernel radius for neighbourhood processing
  --radii-in-km-by-lead-time RADIUS_BY_LEAD_TIME RADIUS_BY_LEAD_TIME
                        The kernel radii for neighbourhood processing and the
                        associated lead times at which the radii are valid.
                        The radii are in km whilst the lead time has units of
                        hours. The radii and lead times are expected as
                        individual comma-separated lists with the list of
                        radii given first followed by a list of lead times to
                        indicate at what lead time each radii should be used.
                        For example: 10,12,14 1,2,3 where a lead time of 1
                        hour uses a radius of 10km, a lead time of 2 hours
                        uses a radius of 12km, etc.
__HELP__
  [[ "$output" == "$expected" ]]
}
