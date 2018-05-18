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

@test "recursive-filter -h" {
  run improver recursive-filter -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-recursive-filter [-h] [--profile]
                                 [--profile_file PROFILE_FILE]
                                 [--input_filepath_alphas_x ALPHAS_X_FILE]
                                 [--input_filepath_alphas_y ALPHAS_Y_FILE]
                                 [--alpha_x ALPHA_X] [--alpha_y ALPHA_Y]
                                 [--iterations ITERATIONS]
                                 [--input_mask_filepath INPUT_MASK_FILE]
                                 [--re_mask]
                                 INPUT_FILE OUTPUT_FILE

Run a recursive filter to convert a square neighbourhood into a Gaussian-like
kernel or smooth over short distances. The filter uses an alpha parameter (0 <
alpha < 1) to control what proportion of the probability is passed onto the
next grid-square in the x and y directions. The alpha parameter can be set on
a grid-square by grid-square basis for the x and y directions separately
(using two arrays of alpha parameters of the same dimensionality as the
domain). Alternatively a single alpha value can be set for each of the x and y
directions. These methods can be mixed, e.g. an array for the x direction and
a float for the y direction and vice versa.

positional arguments:
  INPUT_FILE            A path to an input NetCDF file to be processed
  OUTPUT_FILE           The output path for the processed NetCDF

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --input_filepath_alphas_x ALPHAS_X_FILE
                        A path to a NetCDF file describing the alpha factors
                        to be used for smoothing in the x direction
  --input_filepath_alphas_y ALPHAS_Y_FILE
                        A path to a NetCDF file describing the alpha factors
                        to be used for smoothing in the y direction
  --alpha_x ALPHA_X     A single alpha factor (0 < alpha_x < 1) to be applied
                        to every grid square in the x direction.
  --alpha_y ALPHA_Y     A single alpha factor (0 < alpha_y < 1) to be applied
                        to every grid square in the y direction.
  --iterations ITERATIONS
                        Number of times to apply the filter, default=1
                        (typically < 5)
  --input_mask_filepath INPUT_MASK_FILE
                        A path to an input mask NetCDF file to be used to mask
                        the input file.
  --re_mask             Re-apply mask to recursively filtered output.
__HELP__
  [[ "$output" == "$expected" ]]
}
