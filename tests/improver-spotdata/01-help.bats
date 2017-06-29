#!/usr/bin/env bats
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

@test "spotdata -h" {
  run improver spotdata -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-spotdata [-h] [--diagnostics DIAGNOSTICS [DIAGNOSTICS ...]]
                         [--site_path SITE_PATH]
                         [--constants_path CONSTANTS_PATH]
                         [--latitudes -90,90) [(-90,90) ...]]
                         [--longitudes (-180,180) [(-180,180 ...]]
                         [--altitudes ALTITUDES [ALTITUDES ...]]
                         [--site_ids SITE_IDS [SITE_IDS ...]]
                         [--forecast_date FORECAST_DATE]
                         [--forecast_time FORECAST_TIME]
                         [--forecast_length FORECAST_LENGTH]
                         [--output_path OUTPUT_PATH]
                         [--multiprocess MULTIPROCESS]
                         config_file_path data_path ancillary_path

SpotData : A configurable tool to extract spot-data from gridded diagnostics.
The method of interpolating and adjusting the resulting data can be set by
defining suitable diagnostics configurations.

positional arguments:
  config_file_path      Path to a json file defining the recipes for
                        extracting diagnostics at SpotData sites from gridded
                        data.
  data_path             Path to diagnostic data files.
  ancillary_path        Path to ancillary (time invariant) data files.

optional arguments:
  -h, --help            show this help message and exit
  --diagnostics DIAGNOSTICS [DIAGNOSTICS ...]
                        A list of diagnostics that are to be processed. If
                        unset, all diagnostics defined in the config_file will
                        be produced; e.g. temperature wind_speed
  --site_path SITE_PATH
                        Path to site data file if this is being used to choose
                        sites.
  --constants_path CONSTANTS_PATH
                        Path to json file containing constants to use in
                        SpotData methods.
  --latitudes (-90,90) [(-90,90) ...]
                        List of latitudes of sites of interest.
  --longitudes (-180,180) [(-180,180) ...]
                        List of longitudes of sites of interest.
  --altitudes ALTITUDES [ALTITUDES ...]
                        List of altitudes of sites of interest.
  --site_ids SITE_IDS [SITE_IDS ...]
                        ID numbers for sites can be set if desired.
  --forecast_date FORECAST_DATE
                        Start date of forecast in format YYYYMMDD (e.g.
                        20170327 = 27th March 2017).
  --forecast_time FORECAST_TIME
                        Starting hour of forecast in 24hr clock. (e.g. 3 =
                        03Z, 14 = 14Z).
  --forecast_length FORECAST_LENGTH
                        Length of forecast in hours.
  --output_path OUTPUT_PATH
                        Path to which output files should be written.
  --multiprocess MULTIPROCESS
                        Process diagnostics using multiprocessing.
__HELP__
  [[ "$output" == "$expected" ]]
}
