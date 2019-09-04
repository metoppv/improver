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
"""Script to run neighbourhood processing."""

from improver.argparser import ArgParser
from improver.constants import DEFAULT_PERCENTILES
from improver.nbhood.nbhood import (
    GeneratePercentilesFromANeighbourhood, NeighbourhoodProcessing)
from improver.nbhood.recursive_filter import RecursiveFilter
from improver.utilities.cli_utilities import radius_or_radii_and_lead
from improver.utilities.load import load_cube
from improver.utilities.pad_spatial import remove_cube_halo
from improver.utilities.save import save_netcdf
from improver.wind_calculations.wind_direction import WindDirection


def main(argv=None):
    """Load in arguments and get going."""
    parser = ArgParser(
        description='Apply the requested neighbourhood method via '
                    'the NeighbourhoodProcessing plugin to a file '
                    'whose data can be loaded as a single iris.cube.Cube.')
    parser.add_argument(
        'neighbourhood_output', metavar='NEIGHBOURHOOD_OUTPUT',
        help='The form of the results generated using neighbourhood '
             'processing. If "probabilities" is selected, the mean '
             'probability within a neighbourhood is calculated. If '
             '"percentiles" is selected, then the percentiles are calculated '
             'within a neighbourhood. Calculating percentiles from a '
             'neighbourhood is only supported for a circular neighbourhood. '
             'Options: "probabilities", "percentiles".')
    parser.add_argument('neighbourhood_shape', metavar='NEIGHBOURHOOD_SHAPE',
                        choices=["circular", "square"],
                        help='The shape of the neighbourhood to apply in '
                             'neighbourhood processing. Only a "circular" '
                             'neighbourhood shape is applicable for '
                             'calculating "percentiles" output. '
                             'Options: "circular", "square".')
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
    parser.add_argument('--degrees_as_complex', action='store_true',
                        default=False, help='Set this flag to process angles,'
                        ' eg wind directions, as complex numbers. Not '
                        'compatible with circular kernel, percentiles or '
                        'recursive filter.')
    parser.add_argument('--weighted_mode', action='store_true', default=False,
                        help='For neighbourhood processing using a circular '
                             'kernel, setting the weighted_mode indicates the '
                             'weighting decreases with radius. '
                             'If weighted_mode is not set, a constant '
                             'weighting is assumed. weighted_mode is only '
                             'applicable for calculating "probability" '
                             'neighbourhood output.')
    parser.add_argument('--sum_or_fraction', default="fraction",
                        choices=["sum", "fraction"],
                        help='The neighbourhood output can either be in the '
                             'form of a sum of the neighbourhood, or a '
                             'fraction calculated by dividing the sum of the '
                             'neighbourhood by the neighbourhood area. '
                             '"fraction" is the default option.')
    parser.add_argument('--re_mask', action='store_true',
                        help='If re_mask is set (i.e. True), the original '
                             'un-neighbourhood processed mask is applied to '
                             'mask out the neighbourhood processed dataset. '
                             'If not set, re_mask defaults to False and the '
                             'original un-neighbourhood processed mask is '
                             'not applied. Therefore, the neighbourhood '
                             'processing may result in values being present '
                             'in areas that were originally masked. ')
    parser.add_argument('--percentiles', metavar='PERCENTILES',
                        default=DEFAULT_PERCENTILES, nargs='+', type=float,
                        help='Calculate values at the specified percentiles '
                             'from the neighbourhood surrounding each grid '
                             'point.')
    parser.add_argument('input_filepath', metavar='INPUT_FILE',
                        help='A path to an input NetCDF file to be processed.')
    parser.add_argument('output_filepath', metavar='OUTPUT_FILE',
                        help='The output path for the processed NetCDF.')
    parser.add_argument('--input_mask_filepath', metavar='INPUT_MASK_FILE',
                        help='A path to an input mask NetCDF file to be '
                             'used to mask the input file. '
                             'This is currently only supported for square '
                             'neighbourhoods. The data should contain 1 for '
                             'usable points and 0 for discarded points, e.g. '
                             'a land-mask.')
    parser.add_argument('--halo_radius', metavar='HALO_RADIUS',
                        default=None, type=float,
                        help='radius in metres of excess halo to clip.'
                             ' Used where a larger'
                             ' grid was defined than the standard grid'
                             ' and we want to clip the grid back to the'
                             ' standard grid e.g. for global data'
                             ' regridded to UK area. Default=None')
    parser.add_argument('--apply-recursive-filter', action='store_true',
                        default=False,
                        help='Option to apply the recursive filter to a '
                             'square neighbourhooded output dataset, '
                             'converting it into a Gaussian-like kernel or '
                             'smoothing over short distances. '
                             'The filter uses an alpha '
                             'parameter (0 < alpha < 1) to control what '
                             'proportion of the probability is passed onto '
                             'the next grid-square in the x and y directions. '
                             'The alpha parameter can be set on a grid-square '
                             'by grid-square basis for the x and y directions '
                             'separately (using two arrays of alpha '
                             'parameters of the same dimensionality as the '
                             'domain). Alternatively a single alpha value can '
                             'be set for each of the x and y directions. These'
                             ' methods can be mixed, e.g. an array for the x '
                             'direction and a float for the y direction and '
                             'vice versa. The recursive filter cannot be '
                             'applied to a circular kernel')
    parser.add_argument('--input_filepath_alphas_x_cube',
                        metavar='ALPHAS_X_FILE',
                        help='A path to a NetCDF file describing the alpha '
                             'factors to be used for smoothing in the x '
                             'direction when applying the recursive filter')
    parser.add_argument('--input_filepath_alphas_y_cube',
                        metavar='ALPHAS_Y_FILE',
                        help='A path to a NetCDF file describing the alpha '
                             'factors to be used for smoothing in the y '
                             'direction when applying the recursive filter')
    parser.add_argument('--alpha_x', metavar='ALPHA_X',
                        default=None, type=float,
                        help='A single alpha factor (0 < alpha_x < 1) to be '
                             'applied to every grid square in the x '
                             'direction when applying the recursive filter')
    parser.add_argument('--alpha_y', metavar='ALPHA_Y',
                        default=None, type=float,
                        help='A single alpha factor (0 < alpha_y < 1) to be '
                             'applied to every grid square in the y '
                             'direction when applying the recursive filter.')
    parser.add_argument('--iterations', metavar='ITERATIONS',
                        default=1, type=int,
                        help='Number of times to apply the filter, default=1 '
                        '(typically < 5)')

    args = parser.parse_args(args=argv)

    if (args.neighbourhood_output == "percentiles" and
            args.neighbourhood_shape == "square"):
        parser.wrong_args_error('square', 'neighbourhood_shape')

    if args.neighbourhood_output == "percentiles" and args.weighted_mode:
        parser.wrong_args_error(
            'weighted_mode', 'neighbourhood_shape=percentiles')

    if (args.neighbourhood_output == "probabilities" and
            args.percentiles != DEFAULT_PERCENTILES):
        parser.wrong_args_error(
            'percentiles', 'neighbourhood_shape=probabilities')

    if args.input_mask_filepath and args.neighbourhood_shape == "circular":
        parser.wrong_args_error(
            'neighbourhood_shape=circular', 'input_mask_filepath')

    if args.degrees_as_complex:
        if args.neighbourhood_output == "percentiles":
            parser.error('Cannot generate percentiles from complex numbers')
        if args.neighbourhood_shape == "circular":
            parser.error('Cannot process complex numbers with circular '
                         'neighbourhoods')
        if args.apply_recursive_filter:
            parser.error('Cannot process complex numbers with recursive '
                         'filter')

    # Load Cube
    cube = load_cube(args.input_filepath)
    mask_cube = load_cube(args.input_mask_filepath, allow_none=True)
    alphas_x_cube = load_cube(args.input_filepath_alphas_x_cube,
                              allow_none=True)
    alphas_y_cube = load_cube(args.input_filepath_alphas_y_cube,
                              allow_none=True)

    # Process Cube
    result = process(cube, args.neighbourhood_output,
                     args.neighbourhood_shape, args.radius,
                     args.radii_by_lead_time, args.degrees_as_complex,
                     args.weighted_mode, args.sum_or_fraction, args.re_mask,
                     args.percentiles, mask_cube, args.halo_radius,
                     args.apply_recursive_filter, alphas_x_cube,
                     alphas_y_cube, args.alpha_x, args.alpha_y,
                     args.iterations)

    # Save Cube
    save_netcdf(result, args.output_filepath)


def process(cube, neighbourhood_output, neighbourhood_shape, radius=None,
            radii_by_lead_time=None, degrees_as_complex=False,
            weighted_mode=False, sum_or_fraction="fraction", re_mask=False,
            percentiles=DEFAULT_PERCENTILES, mask_cube=None,
            halo_radius=None, apply_recursive_filter=False, alphas_x_cube=None,
            alphas_y_cube=None, alpha_x=None, alpha_y=None, iterations=1):
    """Runs neighbourhood processing.

    Apply the requested neighbourhood method via the
    NeighbourhoodProcessing plugin to a Cube.

    Args:
        cube (iris.cube.Cube):
            The Cube to be processed.
        neighbourhood_output (str):
            The form of the results generated using neighbourhood processing.
            If "probabilities" is selected, the mean probability with a
            neighbourhood is calculated. If "percentiles" is selected, then
            the percentiles are calculated with a neighbourhood. Calculating
            percentiles from a neighbourhood is only supported for a circular
            neighbourhood.
            Options: "probabilities", "percentiles".
        neighbourhood_shape (str):
            Name of the neighbourhood method to use. Only a "circular"
            neighbourhood shape is applicable for calculating "percentiles"
            output.
            Options: "circular", "square".
        radius (float):
            The radius in metres of the neighbourhood to apply
        radii_by_lead_time (list):
            A list with the radius in metres at [0] and the lead_time at [1]
            Lead time is a List of lead times or forecast periods, at which
            the radii within 'radii' are defined. The lead times are expected
            in hours.
        degrees_as_complex (bool):
            If True processes angles as complex numbers.
            Not compatible with circular kernel, percentiles or recursive
            filter.
            Default is False.
        weighted_mode (bool):
            If True the weighting decreases with radius.
            If False a constant weighting is assumed.
            weighted_mode is only applicable for calculating "probability"
            neighbourhood output using the circular kernal.
            Default is False
        sum_or_fraction (str):
            Identifier for whether sum or fraction should be returned from
            neighbourhooding. The sum represents the sum of the neighbourhood.
            The fraction represents the sum of the neighbourhood divided by
            the neighbourhood area.
            Default is "fraction".
        re_mask (bool):
            If re_mask is True, the original un-neighbourhood processed mask
            is applied to mask out the neighbourhood processed cube.
            If re_mask is False, the original un-neighbourhood processed mask
            is not applied. Therefore, the neighbourhood processing may result
            in values being present in area that were originally masked.
            Default is False.
        percentiles (float or None):
            Calculates value at the specified percentiles from the
            neighbourhood surrounding each grid point.
            Default is improver.constants.DEFAULT_PERCENTILES.
        mask_cube (iris.cube.Cube):
            A cube to mask the input cube. The data should contain 1 for
            usable points and 0 for discarded points.
            Only supported with square neighbourhoods.
            Default is None.
        halo_radius (float or None):
            Radius in metres of excess halo to clip. Used where a larger grid
            was defined than the standard grid and we want to clip the grid
            back to the standard grid.
            Default is None.
        apply_recursive_filter (bool):
            Boolean to apply the recursive filter to a square neighbourhooded
            output dataset, converting it into a Gaussian-like kernel or
            smoothing over short distances.
            Default is False.
        alphas_x_cube (iris.cube.Cube):
            A Cube used for the smoothing in the x direction when applying
            the recursive filter.
            Default is None.
        alphas_y_cube (iris.cube.Cube):
            A Cube used for the smoothing in the y direction when applying
            the recursive filter.
            Default is None.
        alpha_x (float):
            A single alpha factor (0 < alpha_x < 1) to be applied to every grid
            square in the x direction when applying the recursive filter.
            Default is None.
        alpha_y (float):
            A single alpha factor (0 < alpha_x < 1) to be applied to every grid
            square in the y direction when applying the recursive filter.
            Default is None.
        iterations (int):
            The number of times to apply the filter. (typically < 5)
            Default is 1 (one).

    Returns:
        result (iris.cube.Cube):
            A processed Cube.

    Raises:
        RuntimeError:
            If neighbourhood_shape is used with the wrong neighbourhood
            output.
        RuntimeError:
            If weighted_mode is used with the wrong neighbourhood_output.
        RuntimeError:
            If neighbourhood_output='probabilities' and the default
            percentiles are used.
        RuntimeError:
            If neighbourhood_shape='circular' is used with mask cube.
        ValueError:
            If degree_as_complex is used with
            neighbourhood_output='percentiles'.
        ValueError:
            If degree_as_complex is used with neighbourhood_shape='circular'.
        ValueError:
            If degree_as_complex is used with apply_recursive_filter.
        ValueError:
            If neighbourhood_shape is 'circular' and apply_recursive_function
            is True.

    """
    if (neighbourhood_output == "percentiles" and
            neighbourhood_shape == "square"):
        raise RuntimeError('neighbourhood_shape="square" cannot be used with'
                           'neighbourhood_output="percentiles"')

    if neighbourhood_output == "percentiles" and weighted_mode:
        raise RuntimeError('weighted_mode cannot be used with'
                           'neighbourhood_output="percentiles"')

    if (neighbourhood_output == "probabilities" and
            percentiles != DEFAULT_PERCENTILES):
        raise RuntimeError('percentiles cannot be DEFAULT_PERCENTILES with'
                           'neighbourhood_output="probabilities"')

    if mask_cube and neighbourhood_shape == "circular":
        raise RuntimeError('mask_cube cannot be used with'
                           'neighbourhood_output="circular"')

    if neighbourhood_shape == 'circular' and apply_recursive_filter:
        raise ValueError('Recursive filter option is not applicable to '
                         'circular neighbourhoods. ')

    if degrees_as_complex:
        if neighbourhood_output == "percentiles":
            raise ValueError(
                'Cannot generate percentiles from complex numbers')
        if neighbourhood_shape == "circular":
            raise ValueError(
                'Cannot process complex numbers with circular neighbourhoods')
        if apply_recursive_filter:
            raise ValueError(
                'Cannot process complex numbers with recursive filter')

    if degrees_as_complex:
        # convert cube data into complex numbers
        cube.data = WindDirection.deg_to_complex(cube.data)

    radius_or_radii, lead_times = radius_or_radii_and_lead(
        radius, radii_by_lead_time)

    if neighbourhood_output == "probabilities":
        result = (
            NeighbourhoodProcessing(
                neighbourhood_shape, radius_or_radii,
                lead_times=lead_times,
                weighted_mode=weighted_mode,
                sum_or_fraction=sum_or_fraction, re_mask=re_mask
            ).process(cube, mask_cube=mask_cube))
    elif neighbourhood_output == "percentiles":
        result = (
            GeneratePercentilesFromANeighbourhood(
                neighbourhood_shape, radius_or_radii,
                lead_times=lead_times,
                percentiles=percentiles
            ).process(cube))

    # If the '--apply-recursive-filter' option has been specified in the
    # input command, pass the neighbourhooded 'result' cube obtained above
    # through the recursive-filter plugin before saving the output.
    # The recursive filter is only applicable to square neighbourhoods.
    if neighbourhood_shape == 'square' and apply_recursive_filter:
        result = RecursiveFilter(
            alpha_x=alpha_x, alpha_y=alpha_y,
            iterations=iterations, re_mask=re_mask).process(
            result, alphas_x=alphas_x_cube, alphas_y=alphas_y_cube,
            mask_cube=mask_cube)

    if degrees_as_complex:
        # convert neighbourhooded cube back to degrees
        result.data = WindDirection.complex_to_deg(result.data)
    if halo_radius is not None:
        result = remove_cube_halo(result, halo_radius)
    return result


if __name__ == "__main__":
    main()
