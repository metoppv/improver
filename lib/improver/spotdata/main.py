# -*- coding: utf-8 -*-
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
"""The main routine for site specific post-processing."""

import os
import json
import multiprocessing as mp
from iris.cube import CubeList

from improver.spotdata.read_input import (Load,
                                          get_method_prerequisites)
from improver.spotdata.neighbour_finding import PointSelection
from improver.spotdata.extract_data import ExtractData
from improver.spotdata.write_output import WriteOutput
from improver.spotdata.ancillaries import get_ancillary_data
from improver.spotdata.extrema import ExtractExtrema
from improver.spotdata.site_data import ImportSiteData
from improver.spotdata.times import get_forecast_times
from improver.spotdata.common_functions import (construct_neighbour_hash,
                                                datetime_constraint,
                                                extract_cube_at_time,
                                                extract_ad_at_time)


def run_spotdata(config_file_path, data_path, ancillary_path,
                 diagnostic_list=None,
                 site_path=None, constants_file_path=None,
                 site_properties=None, forecast_date=None,
                 forecast_time=None, forecast_length=None,
                 output_path=None, use_multiprocessing=False):
    """
    A routine that calls the components of the spotdata code. This includes
    building site data into a suitable format, finding grid neighbours to
    those sites with the chosen method, and then extracting data with the
    chosen method. The final results are written out to new irregularly
    gridded iris.cube.Cubes.

    Args:
    -----
    config_file_path : string
        Path to a json file defining the recipes for extracting diagnostics at
        SpotData sites from gridded data.

    data_path : string
        Path to diagnostic data files.

    ancillary_path : string
        Path to ancillary data files.

    diagnostic_list : list of strings
        List of diagnostic names that match those defined in the config_file
        to select which diagnostics to process.

    site_path : string
        Path to site data file if in use. If no lats/lons are specified at the
        command line, this file path is needed.

    constants_file_path : string
        Path to a json file defining constants to be used in methods that have
        tolerances that may be set. e.g. maxiumum vertical extrapolation/
        interpolation of temperatures using a temperature lapse rate method.

    site_properties : dict
        Contains:

        latitudes : list of ints/floats or None
            A list of latitudes for running on the fly for a custom set of
            sites. The order should correspond to the subsequent latitudes and
            altitudes variables to construct each site.

        longitudes : list of ints/floats or None
            A list of longitudes for running on the fly for a custom set of
            sites.

        altitudes : list of ints/floats or None
            A list of altitudes for running on the fly for a custom set of
            sites.

        site_ids : list of ints or None
            A list of site_ids to associate with the above on the fly
            constructed sites. This must be ordered the same as the latitudes/
            longitudes/altitudes lists.

    forecast_date : string (YYYYMMDD)
        A string of format YYYYMMDD defining the start date for which forecasts
        are required.

    forecast_time : integer
        An integer giving the hour on the forecast_date at which to start the
        forecast output; 24hr clock such that 17 = 17Z for example.

    forecast_length : integer
        An integer giving the desired length of the forecast output in hours
        (e.g. 48 for a two day forecast period).

    output_path : string
        Path to which output file containing processed diagnostic should be
        written.

    use_multiprocessing : boolean
        A switch determining whether to use multiprocessing in the data
        extraction step.

    Returns:
    --------
    Writes out cubes of the requested diagnostics, with data extracted to the
    sites read from a file or defined at run time.

    Raises:
    -------
    ValueError : raised if no site specifications are provided.
    IOError : if required data files are not found at given data_path.

    """
    # Establish forecast time list based upon input specifications, or if not
    # provided, use defaults.
    forecast_times = get_forecast_times(forecast_date=forecast_date,
                                        forecast_time=forecast_time,
                                        forecast_length=forecast_length)

    # Check site data has been provided.
    if site_path is None and not site_properties:
        raise ValueError("No SpotData site information has been provided "
                         "from a file or defined at runtime.")

    # If using locations set at command line, set optional information such
    # as site altitude and site_id. If a site definition file is provided it
    # will take precedence.
    if site_path is None:
        sites = ImportSiteData('runtime_list').process(site_properties)
    else:
        sites = ImportSiteData('from_file').process(site_path)

    # Read in extraction recipes for all diagnostics.
    with open(config_file_path, 'r') as input_file:
        all_diagnostics = json.load(input_file)

    # Read in constants to use; if not available, defaults will be used.
    config_constants = None
    neighbour_kwargs = {}
    if constants_file_path is not None:
        with open(constants_file_path, 'r') as input_file:
            config_constants = json.load(input_file)
        no_neighbours = config_constants.get('no_neighbours')
        if no_neighbours is not None:
            neighbour_kwargs['no_neighbours'] = no_neighbours

    # Use the diagnostic_list to establish which diagnostics are to be
    # processed; if unset, use all.
    diagnostics = all_diagnostics
    if diagnostic_list is not None:
        diagnostics = dict((diagnostic, all_diagnostics[diagnostic])
                           for diagnostic in diagnostic_list)

    # Load ancillary data files; fields that don't vary in time.
    ancillary_data = get_ancillary_data(diagnostics, ancillary_path)

    # Add configuration constants to ancillaries (may be None if unset).
    ancillary_data['config_constants'] = config_constants

    # Set up site-grid point neighbour list using default method. Other IGPS
    # methods will use this as a starting point so it must always be done.
    # Assumes orography file is on the same grid as the diagnostic data.
    neighbours = {}
    default_neighbours = {'method': 'fast_nearest_neighbour',
                          'vertical_bias': None,
                          'land_constraint': False}
    default_hash = construct_neighbour_hash(default_neighbours)
    neighbours[default_hash] = PointSelection(**default_neighbours).process(
        ancillary_data['orography'], sites,
        ancillary_data=ancillary_data, **neighbour_kwargs)

    # Set up site-grid point neighbour lists for all IGPS methods being used.
    for key in diagnostics.keys():
        neighbour_finding = diagnostics[key]['neighbour_finding']
        neighbour_hash = construct_neighbour_hash(neighbour_finding)
        # Check if defined neighbour method results already exist.
        if neighbour_hash not in neighbours.keys():
            # If not, find neighbours with new method.
            neighbours[neighbour_hash] = (
                PointSelection(**neighbour_finding).process(
                    ancillary_data['orography'], sites,
                    ancillary_data=ancillary_data,
                    default_neighbours=neighbours[default_hash],
                    **neighbour_kwargs)
                )

    if use_multiprocessing:
        # Process diagnostics on separate threads if multiprocessing is
        # selected. Determine number of diagnostics to establish
        # multiprocessing pool size.
        n_diagnostic_threads = min(len(diagnostics.keys()), mp.cpu_count())

        # Establish multiprocessing pool - each diagnostic processed on its
        # own thread.
        diagnostic_pool = mp.Pool(processes=n_diagnostic_threads)

        for key in diagnostics.keys():
            diagnostic = diagnostics[key]
            diagnostic_pool.apply_async(
                process_diagnostic,
                args=(
                    diagnostic, neighbours, sites, forecast_times,
                    ancillary_data, output_path))

        diagnostic_pool.close()
        diagnostic_pool.join()

    else:
        # Process diagnostics serially on one thread.
        for key in diagnostics.keys():
            diagnostic = diagnostics[key]
            process_diagnostic(diagnostic, neighbours, sites, forecast_times,
                               data_path, ancillary_data,
                               output_path=output_path)


def process_diagnostic(diagnostic, neighbours, sites, forecast_times,
                       data_path, ancillary_data, output_path=None):
    """
    Extract data and write output for a given diagnostic.

    Args:
    -----
    diagnostic : string
        String naming the diagnostic to be processed.

    neighbours : numpy.array
        Array of neigbouring grid points that are associated with sites
        in the SortedDictionary of sites.

    sites : dict
        A dictionary containing the properties of spotdata sites.

    forecast_times : list[datetime.datetime objects]
        A list of datetimes representing forecast times for which data is
        required.

    data_path : string
        Path to diagnostic data files.

    ancillary_data : dict
        A dictionary containing additional model data that is needed.
        e.g. {'orography': <cube of orography>}

    output_path : str
        Path to which output file containing processed diagnostic should be
        written.

    Returns:
    --------
    Nil.

    Raises:
    -------
    IOError : If no relevant data cubes are found at given path.
    Exception : No spotdata returned.

    """
    # Search directory structure for all files relevant to current diagnostic.
    files_to_read = [
        os.path.join(dirpath, filename)
        for dirpath, _, files in os.walk(data_path)
        for filename in files if diagnostic['filepath'] in filename]
    if not files_to_read:
        raise IOError('No relevant data files found in {}.'.format(
            data_path))

    # Load cubes into an iris.cube.CubeList.
    cubes = Load('multi_file').process(files_to_read,
                                       diagnostic['diagnostic_name'])

    # Grab the relevant set of grid point neighbours for the neighbour finding
    # method being used by this diagnostic.
    neighbour_hash = construct_neighbour_hash(diagnostic['neighbour_finding'])
    neighbour_list = neighbours[neighbour_hash]

    # Check if additional diagnostics are needed (e.g. multi-level data).
    # If required, load into the additional_diagnostics dictionary.
    additional_diagnostics = get_method_prerequisites(
        diagnostic['interpolation_method'], data_path)

    # Create empty iris.cube.CubeList to hold extracted data cubes.
    resulting_cubes = CubeList()

    # Get optional kwargs that may be set to override defaults.
    optionals = ['upper_level', 'lower_level', 'no_neighbours',
                 'dz_tolerance', 'dthetadz_threshold', 'dz_max_adjustment']
    kwargs = {}
    if ancillary_data.get('config_constants') is not None:
        for optional in optionals:
            constant = ancillary_data.get('config_constants').get(optional)
            if constant is not None:
                kwargs[optional] = constant

    # Loop over forecast times.
    for a_time in forecast_times:
        # Extract Cube from CubeList at current time.
        time_extract = datetime_constraint(a_time)
        cube = extract_cube_at_time(cubes, a_time, time_extract)
        if cube is None:
            # If no cube is available at given time, try the next time.
            continue

        ad = {}
        if additional_diagnostics is not None:
            # Extract additional diagnostcs at current time.
            ad = extract_ad_at_time(additional_diagnostics, a_time,
                                    time_extract)

        args = (cube, sites, neighbour_list, ancillary_data, ad)

        # Extract diagnostic data using defined method.
        resulting_cubes.append(
            ExtractData(
                diagnostic['interpolation_method']).process(*args, **kwargs)
            )

    # Concatenate CubeList into Cube, creating a time DimCoord, and write out.
    if resulting_cubes:
        cube_out, = resulting_cubes.concatenate()
        WriteOutput('as_netcdf', dir_path=output_path).process(cube_out)
    else:
        raise Exception('No data available at given forecast times.')

    # If set in the configuration, extract the diagnostic maxima and minima
    # values.
    if diagnostic['extrema']:
        extrema_cubes = ExtractExtrema(24, start_hour=9).process(cube_out)
        extrema_cubes = extrema_cubes.merge()
        for extrema_cube in extrema_cubes:
            WriteOutput('as_netcdf', dir_path=output_path).process(
                extrema_cube)
