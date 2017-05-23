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
"""The framework for site specific post-processing."""

import argparse
import multiprocessing as mp

from improver.spotdata.read_input import Load
from improver.spotdata.neighbour_finding import PointSelection
from improver.spotdata.extract_data import (ExtractData,
                                            get_method_prerequisites)
from improver.spotdata.write_output import WriteOutput
from improver.spotdata.ancillaries import get_ancillary_data
from improver.spotdata.extrema import ExtractExtrema
from improver.spotdata.site_data import ImportSiteData
from improver.spotdata.times import get_forecast_times
from improver.spotdata.configurations import define_diagnostics


def run_framework(config_name, data_path, ancillary_path, site_path=None,
                  latitudes=None, longitudes=None,
                  altitudes=None, site_ids=None, forecast_date=None,
                  forecast_time=None, forecast_length=None,
                  use_multiprocessing=False):
    '''
    A framework that calls the components of the spotdata code. This includes
    building site data into a suitable format, finding grid neighbours to
    those sites with the chosen method, and then extracting data with the
    chosen method. The final results are written out to new irregularly
    gridded iris.cube.Cubes.

    Args:
    -----
    config_name         : A string giving the chosen configuration with which
                          to run the spotdata system. e.g. pws_default which
                          will produce the required diagnostics for this
                          product.
    data_path           : String giving path to diagnostic data files.
    ancillary_path      : String giving path to ancillary data files.
    site_path           : String giving path to site data file if in use. If
                          no lats/lons are specified at the command line, this
                          file path is needed.
    latitudes           : A list of latitudes for running on the fly for a
                          custom set of sites. The order should correspond
                          to the subsequent latitudes and altitudes variables
                          to construct each site.
    longitudes          : A list of longitudes for running on the fly for a
                          custom set of sites.
    altitudes           : A list of altitudes for running on the fly for a
                          custom set of sites.
    site_ids            : A list of site_ids to associate with the above on
                          the fly constructed sites. This must be ordered the
                          same as the latitudes/longitudes/altitudes lists.
    forecast_date       : A string of format YYYYMMDD defining the start date
                          for which forecasts are required.
    forecast_time       : An integer giving the hour on the forecast_date at
                          which to start the forecast output; 24hr clock such
                          that 17 = 17Z for example.
    forecast_length     : An integer giving the desired length of the forecast
                          output in hours (e.g. 48 for a two day forecast
                          period).
    use_multiprocessing : A boolean determining whether to use multiprocessing
                          in the data extraction component of the code.

    Returns:
    --------
    Nil.

    '''

    # Establish forecast time list based upon input specifications, or if not
    # provided, use defaults.
    forecast_times = get_forecast_times(forecast_date=forecast_date,
                                        forecast_time=forecast_time,
                                        forecast_length=forecast_length)

    # If using locations set at command line, set optional information such
    # as site altitude and site_id.
    if latitudes and longitudes:
        optionals = {}
        if altitudes is not None:
            optionals.update({'altitudes': altitudes})
        if site_ids is not None:
            optionals.update({'site_ids': site_ids})

        sites = ImportSiteData('runtime_list').process(latitudes, longitudes,
                                                       **optionals)

    # Clumsy implementation of grabbing the BD pickle file if no sites are
    # specified.
    if latitudes is None or longitudes is None:
        if site_path is None:
            raise Exception('Site path required to get site data if no sites '
                            'are specified at runtime.')
        else:
            site_path = (site_path + '/bestdata2_locsDB.pkl')
            sites = ImportSiteData('pickle_file').process(site_path)

    # Use the selected config to estabilish which diagnostics are required.
    # Also gets the default method of selecting grid point neighbours for the
    # given configuration.
    neighbour_finding_default, diagnostics = define_diagnostics(config_name,
                                                                data_path)

    # Load ancillary data files; fields that don't vary in time.
    ancillary_data = get_ancillary_data(diagnostics, ancillary_path)

    # Construct a set of neighbour_finding methods to be used in this run.
    neighbour_schemes = list(
        set([diagnostics[x]['neighbour_finding']
             for x in diagnostics.keys()]))
    neighbour_schemes.remove('default')

    # Set up site-grid point neighbour list using default method. Other IGPS
    # methods will use this as a starting point so it must always be done.
    neighbours = {}
    neighbours.update(
        {'default':
         PointSelection(neighbour_finding_default).process(
             ancillary_data['orography'], sites,
             ancillary_data=ancillary_data)
         })

    # Set up site-grid point neighbour lists for all IGPS methods being used.
    for scheme in neighbour_schemes:
        neighbours.update(
            {scheme:
             PointSelection(scheme).process(
                 ancillary_data['orography'], sites,
                 ancillary_data=ancillary_data,
                 default_neighbours=neighbours['default'])
             })

    if use_multiprocessing:
        # Process diagnostics on separate threads is multiprocessing is
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
                    ancillary_data))

        diagnostic_pool.close()
        diagnostic_pool.join()

    else:
        # Process diagnostics serially on one thread.
        for key in diagnostics.keys():
            diagnostic = diagnostics[key]
            process_diagnostic(diagnostic, neighbours, sites, forecast_times,
                               ancillary_data)


def process_diagnostic(diagnostic, neighbours, sites, forecast_times,
                       ancillary_data):
    '''
    Extract data and write output for a given diagnostic.

    Args:
    -----
    diagnostic     : String naming the diagnostic to be processed.
    neighbours     : Dictionary of gridpoint neighbours to each site produced
                     by the different available neighbour finding methods that
                     have been used in the chosen configuration.
    sites          : Dictionary of spotdata sites to process.
    forecast_times : Python datetime objects specifying the times for which
                     forecast diagnostics are required.
    ancillary_data : Dictionary of time invariant fields that may be used by
                     the data extraction methods (e.g. orography).

    Returns:
    --------
    Nil.

    '''
    # print 'neighbour finding with ', diagnostic['neighbour_finding']
    # print 'using interpolation method ', diagnostic['interpolation_method']

    data = Load('multi_file').process(diagnostic['filepath'],
                                      diagnostic['diagnostic_name'])
    neighbour_list = neighbours[diagnostic['neighbour_finding']]

    additional_data = get_method_prerequisites(
        diagnostic['interpolation_method'])

    cubes_out = ExtractData(
        diagnostic['interpolation_method']
        ).process(data, sites, neighbour_list, forecast_times,
                  additional_data, ancillary_data=ancillary_data)

    cube_out, = cubes_out.concatenate()

    if diagnostic['extrema']:
        ExtractExtrema('In24hr').process(cube_out)

    WriteOutput('as_netcdf').process(cube_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSPS.')
    parser.add_argument('config_name',
                        help='Configuration to use, defining which diagnostics'
                             ' to produce.'
                        )
    parser.add_argument('data_path', type=str,
                        help='Path to diagnostic data files.'
                        )
    parser.add_argument('ancillary_path', type=str,
                        help='Path to ancillary (time invariant) data files.'
                        )
    parser.add_argument('--site_path', type=str,
                        help='Path to site data file.'
                        )
    parser.add_argument('--latitudes', type=int, choices=range(-90, 90),
                        nargs='+',
                        help='Latitude of site of interest.'
                        )
    parser.add_argument('--longitudes', type=int, choices=range(-180, 180),
                        nargs='+',
                        help='Longitude of site of interest.'
                        )
    parser.add_argument('--altitudes', type=float, nargs='+',
                        help='Altitude of site of interest.'
                        )
    parser.add_argument('--site_ids', type=float, nargs='+',
                        help='ID no. for sites can be set if desired.'
                        )
    parser.add_argument('--start_date', type=str,
                        help='Start date of forecast in format YYYYMMDD '
                             '(e.g. 20170327 = 27th March 2017).'
                        )
    parser.add_argument('--start_time', type=int,
                        help='Starting hour in 24hr clock of forecast. '
                             '(e.g. 3 = 03Z, 14 = 14Z).'
                        )
    parser.add_argument('--length', type=int,
                        help='Length of forecast in hours.'
                        )
    parser.add_argument('--multiprocess', type=bool,
                        help='Process diagnostics using multiprocessing.'
                        )

    args = parser.parse_args()

    run_framework(args.config_name, args.data_path, args.ancillary_path,
                  site_path=args.site_path, latitudes=args.latitudes,
                  longitudes=args.longitudes, altitudes=args.altitudes,
                  site_ids=args.site_ids,
                  forecast_date=args.start_date, forecast_time=args.start_time,
                  forecast_length=args.length,
                  use_multiprocessing=args.multiprocess)
