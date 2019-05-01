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

@test "nbhood -h" {
  run improver nbhood-land-and-sea -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver nbhood-land-and-sea [-h] [--profile]
                                    [--profile_file PROFILE_FILE]
                                    [--weights_for_collapsing_dim WEIGHTS]
                                    [--radius RADIUS | --radii-by-lead-time RADII_BY_LEAD_TIME LEAD_TIME_IN_HOURS]
                                    [--sum_or_fraction {sum,fraction}]
                                    [--intermediate_filepath INTERMEDIATE_FILEPATH]
                                    INPUT_FILE INPUT_MASK OUTPUT_FILE

Neighbourhood the input dataset over two distinct regions of land and sea. If
performed as a single level neighbourhood, a land-sea mask should be provided.
If instead topographic_zone neighbourhooding is being employed, the mask
should be one of topographic zones. In the latter case a weights array is also
needed to collapse the topographic_zone coordinate. These weights are created
with the improver generate-topography-bands-weights CLI and should be made
using a land-sea mask, which will then be employed within this code to draw
the distinction between the two surface types.

positional arguments:
  INPUT_FILE            A path to an input NetCDF file to be processed.
  INPUT_MASK            A path to an input NetCDF file containing either a
                        mask of topographic zones over land or a land-sea
                        mask.
  OUTPUT_FILE           The output path for the processed NetCDF.

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --sum_or_fraction {sum,fraction}
                        The neighbourhood output can either be in the form of
                        a sum of the neighbourhood, or a fraction calculated
                        by dividing the sum of the neighbourhood by the
                        neighbourhood area. "fraction" is the default option.
  --intermediate_filepath INTERMEDIATE_FILEPATH
                        Intermediate filepath for results following
                        topographic masked neighbourhood processing of land
                        points and prior to collapsing the topographic_zone
                        coordinate. Intermediate files will not be produced if
                        no topographic masked neighbourhood processing occurs.

Collapse weights - required if using a topographic zones mask:
  --weights_for_collapsing_dim WEIGHTS
                        A path to an weights NetCDF file containing the
                        weights which are used for collapsing the dimension
                        gained through masking. These weights must have been
                        created using a land-sea mask.

Neighbourhooding Radius - Set only one of the options:
  --radius RADIUS       The radius (in m) for neighbourhood processing.
  --radii-by-lead-time RADII_BY_LEAD_TIME LEAD_TIME_IN_HOURS
                        The radii for neighbourhood processing and the
                        associated lead times at which the radii are valid.
                        The radii are in metres whilst the lead time has units
                        of hours. The radii and lead times are expected as
                        individual comma-separated lists with the list of
                        radii given first followed by a list of lead times to
                        indicate at what lead time each radii should be used.
                        For example: 10000,12000,14000 1,2,3 where a lead time
                        of 1 hour uses a radius of 10000m, a lead time of 2
                        hours uses a radius of 12000m, etc.
__HELP__
  [[ "$output" == "$expected" ]]
}
