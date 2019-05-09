#!/usr/bin/env bats
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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

@test "threshold -h" {
  run improver threshold -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver threshold [-h] [--profile] [--profile_file PROFILE_FILE]
                          [--threshold_config THRESHOLD_CONFIG]
                          [--threshold_units THRESHOLD_UNITS]
                          [--below_threshold] [--fuzzy_factor FUZZY_FACTOR]
                          [--collapse-coord COLLAPSE-COORD]
                          [--vicinity VICINITY]
                          INPUT_FILE OUTPUT_FILE
                          [THRESHOLD_VALUES [THRESHOLD_VALUES ...]]

Calculate the threshold truth value of input data relative to the provided
threshold value. By default data are tested to be above the thresholds, though
the --below_threshold flag enables testing below thresholds. A fuzzy factor or
fuzzy bounds may be provided to capture data that is close to the threshold.

positional arguments:
  INPUT_FILE            A path to an input NetCDF file to be processed
  OUTPUT_FILE           The output path for the processed NetCDF
  THRESHOLD_VALUES      Threshold value or values about which to calculate the
                        truth values; e.g. 270 300. Must be omitted if
                        --threshold_config is used.

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --threshold_config THRESHOLD_CONFIG
                        Threshold configuration JSON file containing
                        thresholds and (optionally) fuzzy bounds. Best used in
                        combination with --threshold_units. It should contain
                        a dictionary of strings that can be interpreted as
                        floats with the structure: "THRESHOLD_VALUE":
                        [LOWER_BOUND, UPPER_BOUND] e.g: {"280.0": [278.0,
                        282.0], "290.0": [288.0, 292.0]}, or with structure
                        "THRESHOLD_VALUE": "None" (no fuzzy bounds). Repeated
                        thresholds with different bounds are not handled well.
                        Only the last duplicate will be used.
  --threshold_units THRESHOLD_UNITS
                        Units of the threshold values. If not provided the
                        units are assumed to be the same as those of the input
                        dataset. Specifying the units here will allow a
                        suitable conversion to match the input units if
                        possible.
  --below_threshold     By default truth values of 1 are returned for data
                        ABOVE the threshold value(s). Using this flag changes
                        this behaviour to return 1 for data below the
                        threshold values.
  --fuzzy_factor FUZZY_FACTOR
                        A decimal fraction defining the factor about the
                        threshold value(s) which should be treated as fuzzy.
                        Data which fail a test against the hard threshold
                        value may return a fractional truth value if they fall
                        within this fuzzy factor region. Fuzzy factor must be
                        in the range 0-1, with higher values indicating a
                        narrower fuzzy factor region / sharper threshold. NB A
                        fuzzy factor cannot be used with a zero threshold or a
                        threshold_config file.
  --collapse-coord COLLAPSE-COORD
                        An optional ability to set which coordinate we want to
                        collapse over. The default is set to None.
  --vicinity VICINITY   If set, distance in metres used to define the vicinity
                        within which to search for an occurrence.
__HELP__
  [[ "$output" == "$expected" ]]
}
