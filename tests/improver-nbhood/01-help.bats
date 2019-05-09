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
  run improver nbhood -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver nbhood [-h] [--profile] [--profile_file PROFILE_FILE]
                       [--radius RADIUS | --radii-by-lead-time RADII_BY_LEAD_TIME LEAD_TIME_IN_HOURS]
                       [--degrees_as_complex] [--weighted_mode]
                       [--sum_or_fraction {sum,fraction}] [--re_mask]
                       [--percentiles PERCENTILES [PERCENTILES ...]]
                       [--input_mask_filepath INPUT_MASK_FILE]
                       [--halo_radius HALO_RADIUS] [--apply-recursive-filter]
                       [--input_filepath_alphas_x_cube ALPHAS_X_FILE]
                       [--input_filepath_alphas_y_cube ALPHAS_Y_FILE]
                       [--alpha_x ALPHA_X] [--alpha_y ALPHA_Y]
                       [--iterations ITERATIONS]
                       NEIGHBOURHOOD_OUTPUT NEIGHBOURHOOD_SHAPE INPUT_FILE
                       OUTPUT_FILE

Apply the requested neighbourhood method via the NeighbourhoodProcessing
plugin to a file whose data can be loaded as a single iris.cube.Cube.

positional arguments:
  NEIGHBOURHOOD_OUTPUT  The form of the results generated using neighbourhood
                        processing. If "probabilities" is selected, the mean
                        probability within a neighbourhood is calculated. If
                        "percentiles" is selected, then the percentiles are
                        calculated within a neighbourhood. Calculating
                        percentiles from a neighbourhood is only supported for
                        a circular neighbourhood. Options: "probabilities",
                        "percentiles".
  NEIGHBOURHOOD_SHAPE   The shape of the neighbourhood to apply in
                        neighbourhood processing. Only a "circular"
                        neighbourhood shape is applicable for calculating
                        "percentiles" output. Options: "circular", "square".
  INPUT_FILE            A path to an input NetCDF file to be processed.
  OUTPUT_FILE           The output path for the processed NetCDF.

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
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
  --degrees_as_complex  Set this flag to process angles, eg wind directions,
                        as complex numbers. Not compatible with circular
                        kernel, percentiles or recursive filter.
  --weighted_mode       For neighbourhood processing using a circular kernel,
                        setting the weighted_mode indicates the weighting
                        decreases with radius. If weighted_mode is not set, a
                        constant weighting is assumed. weighted_mode is only
                        applicable for calculating "probability" neighbourhood
                        output.
  --sum_or_fraction {sum,fraction}
                        The neighbourhood output can either be in the form of
                        a sum of the neighbourhood, or a fraction calculated
                        by dividing the sum of the neighbourhood by the
                        neighbourhood area. "fraction" is the default option.
  --re_mask             If re_mask is set (i.e. True), the original un-
                        neighbourhood processed mask is applied to mask out
                        the neighbourhood processed dataset. If not set,
                        re_mask defaults to False and the original un-
                        neighbourhood processed mask is not applied.
                        Therefore, the neighbourhood processing may result in
                        values being present in areas that were originally
                        masked.
  --percentiles PERCENTILES [PERCENTILES ...]
                        Calculate values at the specified percentiles from the
                        neighbourhood surrounding each grid point.
  --input_mask_filepath INPUT_MASK_FILE
                        A path to an input mask NetCDF file to be used to mask
                        the input file. This is currently only supported for
                        square neighbourhoods. The data should contain 1 for
                        usable points and 0 for discarded points, e.g. a land-
                        mask.
  --halo_radius HALO_RADIUS
                        radius in metres of excess halo to clip. Used where a
                        larger grid was defined than the standard grid and we
                        want to clip the grid back to the standard grid e.g.
                        for global data regridded to UK area. Default=None
  --apply-recursive-filter
                        Option to apply the recursive filter to a square
                        neighbourhooded output dataset, converting it into a
                        Gaussian-like kernel or smoothing over short
                        distances. The filter uses an alpha parameter (0 <
                        alpha < 1) to control what proportion of the
                        probability is passed onto the next grid-square in the
                        x and y directions. The alpha parameter can be set on
                        a grid-square by grid-square basis for the x and y
                        directions separately (using two arrays of alpha
                        parameters of the same dimensionality as the domain).
                        Alternatively a single alpha value can be set for each
                        of the x and y directions. These methods can be mixed,
                        e.g. an array for the x direction and a float for the
                        y direction and vice versa. The recursive filter
                        cannot be applied to a circular kernel
  --input_filepath_alphas_x_cube ALPHAS_X_FILE
                        A path to a NetCDF file describing the alpha factors
                        to be used for smoothing in the x direction when
                        applying the recursive filter
  --input_filepath_alphas_y_cube ALPHAS_Y_FILE
                        A path to a NetCDF file describing the alpha factors
                        to be used for smoothing in the y direction when
                        applying the recursive filter
  --alpha_x ALPHA_X     A single alpha factor (0 < alpha_x < 1) to be applied
                        to every grid square in the x direction when applying
                        the recursive filter
  --alpha_y ALPHA_Y     A single alpha factor (0 < alpha_y < 1) to be applied
                        to every grid square in the y direction when applying
                        the recursive filter.
  --iterations ITERATIONS
                        Number of times to apply the filter, default=1
                        (typically < 5)
__HELP__
  [[ "$output" == "$expected" ]]
}
