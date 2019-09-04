#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
"""Script to run neighbourhooding processing when iterating over a coordinate
defining a series of masks."""

from improver.argparser import ArgParser
from improver.nbhood.use_nbhood import (
    ApplyNeighbourhoodProcessingWithAMask,
    CollapseMaskedNeighbourhoodCoordinate)
from improver.utilities.cli_utilities import radius_or_radii_and_lead
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Load in arguments for applying neighbourhood processing when using a
    mask."""
    parser = ArgParser(
        description='Apply the requested neighbourhood method via the '
                    'ApplyNeighbourhoodProcessingWithAMask plugin to a file '
                    'with one diagnostic dataset in combination with a file '
                    'containing one or more masks. The mask dataset may have '
                    'an extra dimension compared to the input diagnostic. '
                    'In this case, the user specifies the name of '
                    'the extra coordinate and this coordinate is iterated '
                    'over so each mask is applied to separate slices over the'
                    ' input data. These intermediate masked datasets are then'
                    ' concatenated, resulting in a dataset that has been '
                    ' processed using multiple masks and has gained an extra '
                    'dimension from the masking.  There is also an option to '
                    're-mask the output dataset, so that after '
                    'neighbourhood processing, non-zero values are only '
                    'present for unmasked grid points. '
                    'There is an alternative option of collapsing the '
                    'dimension that we gain using this processing using a '
                    'weighted average.')
    parser.add_argument('coord_for_masking', metavar='COORD_FOR_MASKING',
                        help='Coordinate to iterate over when applying a mask '
                             'to the neighbourhood processing. ')
    parser.add_argument('input_filepath', metavar='INPUT_FILE',
                        help='A path to an input NetCDF file to be processed.')
    parser.add_argument('input_mask_filepath', metavar='INPUT_MASK_FILE',
                        help='A path to an input mask NetCDF file to be '
                             'used to mask the input file.')
    parser.add_argument('output_filepath', metavar='OUTPUT_FILE',
                        help='The output path for the processed NetCDF.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--radius', metavar='RADIUS', type=float,
                       help='The radius (in m) for neighbourhood processing.')
    group.add_argument('--radii-by-lead-time',
                       metavar=('RADII_BY_LEAD_TIME', 'LEAD_TIME_IN_HOURS'),
                       nargs=2,
                       help='The radii for neighbourhood processing '
                       'and the associated lead times at which the radii are '
                       'valid. The radii are in metres whilst the lead time '
                       'has units of hours. The radii and lead times are '
                       'expected as individual comma-separated lists with '
                       'the list of radii given first followed by a list of '
                       'lead times to indicate at what lead time each radii '
                       'should be used. For example: 10000,12000,14000 1,2,3 '
                       'where a lead time of 1 hour uses a radius of 10000m, '
                       'a lead time of 2 hours uses a radius of 12000m, etc.')
    parser.add_argument('--sum_or_fraction', default="fraction",
                        choices=["sum", "fraction"],
                        help='The neighbourhood output can either be in the '
                             'form of a sum of the neighbourhood, or a '
                             'fraction calculated by dividing the sum of the '
                             'neighbourhood by the neighbourhood area. '
                             '"fraction" is the default option.')
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--re_mask', action='store_true',
                        help='If re_mask is set (i.e. True), the output data '
                             'following neighbourhood processing is '
                             're-masked. This re-masking removes any values '
                             'that have been generated by neighbourhood '
                             'processing at grid points that were '
                             'originally masked. '
                             'If not set, re_mask defaults to False and no '
                             're-masking is applied to the neighbourhood '
                             'processed output. Therefore, the neighbourhood '
                             'processing may result in values being present '
                             'in areas that were originally masked. This '
                             'allows the the values in adjacent bands to be'
                             'weighted together if the additional dimension'
                             'from the masking process is collapsed.')
    group2.add_argument('--collapse_dimension', action='store_true',
                        help='Collapse the dimension from the mask, by doing '
                             'a weighted mean using the weights provided. '
                             'This is only suitable when the result is is '
                             'left unmasked, so there is data to weight '
                             'between the points in coordinate we are '
                             'collapsing.')
    parser.add_argument('--weights_for_collapsing_dim', metavar='WEIGHTS',
                        default=None,
                        help='A path to an weights NetCDF file containing the '
                             'weights which are used for collapsing the '
                             'dimension gained through masking.')
    parser.add_argument('--intermediate_filepath', default=None,
                        help='If provided the result after neighbourhooding, '
                             'before collapsing the extra dimension is saved '
                             'in the given filepath.')

    args = parser.parse_args(args=argv)

    # Load Cubes
    cube = load_cube(args.input_filepath)
    mask_cube = load_cube(args.input_mask_filepath)
    weights = load_cube(args.weights_for_collapsing_dim) if \
        args.collapse_dimension else None

    # Process Cube
    result, intermediate_cube = process(
        cube, mask_cube, weights, args.coord_for_masking, args.radius,
        args.radii_by_lead_time, args.sum_or_fraction, args.re_mask,
        args.collapse_dimension)

    # Save Cube
    save_netcdf(result, args.output_filepath)
    if args.intermediate_filepath is not None:
        save_netcdf(intermediate_cube, args.intermediate_filepath)


def process(cube, mask_cube, weights, coord_for_masking, radius=None,
            radii_by_lead_time=None, sum_or_fraction="fraction", re_mask=False,
            collapse_dimension=False):
    """Runs neighbourhooding processing iterating over a coordinate by mask.

    Apply the requested neighbourhood method via the
    ApplyNeighbourhoodProcessingWithMask plugin to a file with one diagnostic
    dataset in combination with a cube containing one or more masks.
    The mask dataset may have an extra dimension compared to the input
    diagnostic. In this case, the user specifies the name of the extra
    coordinate and this coordinate is iterated over so each mask is applied
    to separate slices over the input cube. These intermediate masked datasets
    are then concatenated, resulting in a dataset that has been processed
    using multiple masks and has gained an extra dimension from the masking.
    There is also an option to re-mask the output dataset, so that after
    neighbourhood processing non-zero values are only present for unmasked
    grid points.
    There is an alternative option of collapsing the dimension that we gain
    using this processing using a weighted average.

    Args:
        cube (iris.cube.Cube):
            Cube to be processed.
        mask_cube (iris.cube.Cube):
            Cube to act as a mask.
        weights (iris.cube.Cube):
            Cube containing the weights which are used for collapsing the
            dimension gained through masking.
        coord_for_masking (str):
            String matching the name of the coordinate that will be used
            for masking.
        radius (float):
            The radius in metres of the neighbourhood to apply.
            Rounded up to convert into integer number of grid points east and
            north, based on the characteristic spacing at the zero indices of
            the cube projection-x and y coordinates.
            Default is None.
        radii_by_lead_time (float or list of float):
            A list with the radius in metres at [0] and the lead_time at [1]
            Lead time is a List of lead times or forecast periods, at which
            the radii within 'radii' are defined. The lead times are expected
            in hours.
            Default is None.
        sum_or_fraction (str):
            Identifier for whether sum or fraction should be returned from
            neighbourhooding.
            Sum represents the sum of the neighbourhood.
            Fraction represents the sum of the neighbourhood divided by the
            neighbourhood area.
            Default is fraction.
        re_mask (bool):
            If True, the original un-neighbourhood processed mask
            is applied to mask out the neighbourhood processed cube.
            If False, the original un-neighbourhood processed mask is not
            applied.
            Therefore, the neighbourhood processing may result in
            values being present in areas that were originally masked.
            Default is False.
        collapse_dimension (bool):
            Collapse the dimension from the mask, by doing a weighted mean
            using the weights provided.  This is only suitable when the result
            is left unmasked, so there is data to weight between the points
            in the coordinate we are collapsing.
            Default is False.

    Returns:
        (tuple): tuple containing:
            **result** (iris.cube.Cube):
                A cube after being fully processed.
            **intermediate_cube** (iris.cube.Cube):
                A cube before it is collapsed, if 'collapse_dimension' is True.

    """
    radius_or_radii, lead_times = radius_or_radii_and_lead(radius,
                                                           radii_by_lead_time)

    result = ApplyNeighbourhoodProcessingWithAMask(
        coord_for_masking, radius_or_radii, lead_times=lead_times,
        sum_or_fraction=sum_or_fraction,
        re_mask=re_mask).process(cube, mask_cube)
    intermediate_cube = result.copy()

    # Collapse with the masking dimension.
    if collapse_dimension:
        result = CollapseMaskedNeighbourhoodCoordinate(
            coord_for_masking, weights).process(result)
    return result, intermediate_cube


if __name__ == "__main__":
    main()
