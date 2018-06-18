#!/usr/bin/env bats
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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

@test "spot-extract -h" {
  run improver spot-extract -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-spot-extract [-h] [--profile] [--profile_file PROFILE_FILE]
                             [--diagnostics DIAGNOSTICS [DIAGNOSTICS ...]]
                             [--site_path SITE_PATH]
                             [--constants_path CONSTANTS_PATH]
                             [--latitudes -90,90) [(-90,90) ...]]
                             [--longitudes (-180,180) [(-180,180 ...]]
                             [--altitudes ALTITUDES [ALTITUDES ...]]
                             [--multiprocess]
                             config_file_path data_path ancillary_path
                             output_path

SpotData : A configurable tool to extract spot-data from gridded diagnostics.
The method of interpolating and adjusting the resulting data can be set by
defining suitable diagnostics configurations.

positional arguments:
  config_file_path      Path to a json file defining the recipes for
                        extracting diagnostics at SpotData sites from gridded
                        data.
  data_path             Path to a file containing the diagnostic to be
                        processed.
  ancillary_path        Path to ancillary (time invariant) data files.
  output_path           Path to which output files should be written.

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
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
  --multiprocess        Process diagnostics using multiprocessing.
__HELP__
  [[ "$output" == "$expected" ]]
}
