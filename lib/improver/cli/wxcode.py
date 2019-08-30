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
"""CLI to generate weather symbols."""

import argparse
from argparse import RawTextHelpFormatter

import numpy as np

from improver.argparser import ArgParser
from improver.utilities.load import load_cubelist
from improver.utilities.save import save_netcdf
from improver.wxcode.weather_symbols import WeatherSymbols
from improver.wxcode.wxcode_decision_tree import wxcode_decision_tree
from improver.wxcode.wxcode_decision_tree_global import (
    wxcode_decision_tree_global)
from improver.wxcode.wxcode_utilities import expand_nested_lists


def interrogate_decision_tree(wxtree):
    """
    Obtain a list of necessary inputs from the decision tree as it is currently
    defined. Return a list of the diagnostic names, the thresholds needed,
    and whether they are thresholded above or below these values. This output
    is used to create the CLI help, informing the user of the necessary inputs.

    Returns:
        output (list of str):
            Returns a list of strings, an entry for each diagnostic required,
            including threshold details.
    """

    # Get current weather symbol decision tree and populate a list of
    # required inputs for printing.
    if wxtree == 'high_resolution':
        queries = wxcode_decision_tree()
    elif wxtree == 'global':
        queries = wxcode_decision_tree_global()

    # Diagnostic names and threshold values.
    requirements = {}
    # How the data has been thresholded relative to these thresholds.
    relative = {}

    for query in queries.values():
        diagnostics = expand_nested_lists(query, 'diagnostic_fields')
        for index, diagnostic in enumerate(diagnostics):
            if diagnostic not in requirements:
                requirements[diagnostic] = []
                relative[diagnostic] = []
            requirements[diagnostic].extend(
                expand_nested_lists(query, 'diagnostic_thresholds')[index])
            relative[diagnostic].append(
                expand_nested_lists(query, 'diagnostic_conditions')[index])

    # Create a list of formatted strings that will be printed as part of the
    # CLI help.
    output = []
    for requirement in requirements:
        entries = np.array([entry for entry in requirements[requirement]])
        relations = np.array([entry for entry in relative[requirement]])
        _, thresholds = np.unique(np.array([item.points.item()
                                            for item in entries]),
                                  return_index=True)
        output.append('{}; thresholds: {}'.format(
            requirement, ', '.join([
                '{} ({})'.format(str(threshold.points.item()),
                                 str(threshold.units))
                for threshold, relation in
                zip(entries[thresholds], relations[thresholds])])))
    return output


def main(argv=None):
    """Parser to accept input data and an output destination before invoking
    the weather symbols plugin.
    """

    diagnostics = interrogate_decision_tree('high_resolution')
    n_files = len(diagnostics)
    dlist = (' - {}\n'*n_files)

    diagnostics_global = interrogate_decision_tree('global')
    n_files_global = len(diagnostics_global)
    dlist_global = (' - {}\n'*n_files_global)

    parser = ArgParser(
        description='Calculate gridded weather symbol codes.\nThis plugin '
        'requires a specific set of input diagnostics, where data\nmay be in '
        'any units to which the thresholds given below can\nbe converted:\n' +
        dlist.format(*diagnostics) +
        '\n\n or for global data\n\n' +
        dlist_global.format(*diagnostics_global),
        formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        'input_filepaths', metavar='INPUT_FILES', nargs="+",
        help='Paths to files containing the required input diagnostics.')
    parser.add_argument('output_filepath', metavar='OUTPUT_FILE',
                        help='The output path for the processed NetCDF.')
    parser.add_argument("--wxtree", metavar="WXTREE",
                        default="high_resolution",
                        choices=["high_resolution", "global"],
                        help="Weather Code tree.\n"
                        "Choices are high_resolution or global.\n"
                        "Default=high_resolution.", type=str)

    args = parser.parse_args(args=argv)

    # Load Cube
    cubes = load_cubelist(args.input_filepaths, no_lazy_load=True)
    required_number_of_inputs = n_files
    if args.wxtree == 'global':
        required_number_of_inputs = n_files_global
    if len(cubes) != required_number_of_inputs:
        msg = ('Incorrect number of inputs: files {} gave {} cubes' +
               ', {} required').format(args.input_filepaths, len(cubes),
                                       required_number_of_inputs)
        raise argparse.ArgumentTypeError(msg)

    # Process Cube
    result = process(cubes, args.wxtree)

    # Save Cube
    save_netcdf(result, args.output_filepath)


def process(cubes, wxtree='high_resolution'):
    """ Processes cube for Weather symbols.

    Args:
        cubes (iris.cube.Cubelist):
            A cubelist containing the diagnostics required for the
            weather symbols decision tree, these at co-incident times.
        wxtree (str):
            Weather Code tree.
            Choices are high_resolution or global.
            Default is 'high_resolution'.

    Returns:
        result (iris.cube.Cube):
            A cube of weather symbols.

    """
    result = WeatherSymbols(wxtree=wxtree).process(cubes)
    return result


if __name__ == "__main__":
    main()
