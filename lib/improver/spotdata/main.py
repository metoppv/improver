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

from functools import partial
import multiprocessing as mp

from iris.cube import CubeList

from improver.spotdata.neighbour_finding import PointSelection
from improver.spotdata.extract_data import ExtractData
from improver.spotdata.extrema import ExtractExtrema
from improver.spotdata.common_functions import (construct_neighbour_hash,
                                                datetime_constraint,
                                                extract_cube_at_time,
                                                extract_ad_at_time)


def run_spotdata(diagnostics, ancillary_data, sites, config_constants,
                 use_multiprocessing=False):
    """
    A routine that calls the components of the spotdata code. This includes
    building site data into a suitable format, finding grid neighbours to
    those sites with the chosen method, and then extracting data with the
    chosen method. The final results are written out to new irregularly
    gridded iris.cube.Cubes.

    Args:
        diagnostics (dict):
            Dictionary containing the information regarding the methods that
            will be applied for a specific diagnostic, as well as the data
            following loading in a cube, and any additional data required to
            be able to compute the methods requested.

            For example::

              {
                  "temperature": {
                      "diagnostic_name": "air_temperature",
                      "extrema": True,
                      "filepath": "temperature_at_screen_level",
                      "interpolation_method": "use_nearest",
                      "neighbour_finding": {
                          "land_constraint": False,
                          "method": "fast_nearest_neighbour",
                          "vertical_bias": None
                      "data": iris.cube.CubeList
                      "additional_data" : iris.cube.CubeList
                      }
                  }
              }

        ancillary_data (dict):
            Dictionary containing named ancillary data; the key gives the name
            and the item is the iris.cube.Cube of data.

        sites (dict):
            Contains:

            latitudes (list of ints/floats or None):
                A list of latitudes for running for a custom set of
                sites. The order should correspond to the subsequent latitudes
                and altitudes variables to construct each site.

            longitudes (list of ints/floats or None):
                A list of longitudes for running for a custom set of
                sites.

            altitudes (list of ints/floats or None):
                A list of altitudes for running for a custom set of
                sites.

            site_ids (list of ints or None):
                A list of site_ids to associate with the above
                constructed sites. This must be ordered the same as the
                latitudes/longitudes/altitudes lists.

        config_constants (dict):
            Dictionary defining constants to be used in methods that have
            tolerances that may be set. e.g. maximum vertical extrapolation/
            interpolation of temperatures using a temperature lapse rate
            method.

        use_multiprocessing (boolean):
            A switch determining whether to use multiprocessing in the data
            extraction step.

    Returns:
        (tuple): tuple containing:
            **resulting_cube** (iris.cube.Cube or None):
                Cube after extracting the diagnostic requested using the
                desired extraction method.
                None is returned if the "resulting_cubes" is an empty CubeList
                after processing.
            **extrema_cubes** (iris.cube.CubeList or None):
                CubeList containing extrema values, if the 'extrema' diagnostic
                is requested.
                None is returned if the value for diagnostic_dict["extrema"]
                is False, so that the extrema calculation is not required.
    """
    # Read in constants to use; if not available, defaults will be used.
    neighbour_kwargs = {}
    if config_constants is not None:
        no_neighbours = config_constants.get('no_neighbours')
        if no_neighbours is not None:
            neighbour_kwargs['no_neighbours'] = no_neighbours

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

        diagnostic_keys = [
            diagnostic_name for diagnostic_name in diagnostics.keys()]

        result = (
            diagnostic_pool.map_async(
                partial(
                    process_diagnostic, diagnostics, neighbours, sites,
                    ancillary_data), diagnostic_keys))
        diagnostic_pool.close()
        diagnostic_pool.join()
        resulting_cubes = CubeList()
        extrema_cubes = CubeList()
        for result in result.get():
            resulting_cubes.append(result[0])
            extrema_cubes.append(result[1:])
    else:
        # Process diagnostics serially on one thread.
        resulting_cubes = CubeList()
        extrema_cubes = CubeList()
        for key in diagnostics.keys():
            resulting_cube, extrema_cubelist = (
                process_diagnostic(
                    diagnostics, neighbours, sites,
                    ancillary_data, key))
            resulting_cubes.append(resulting_cube)
            extrema_cubes.append(extrema_cubelist)
    return resulting_cubes, extrema_cubes


def process_diagnostic(diagnostics, neighbours, sites,
                       ancillary_data, diagnostic_name):
    """
    Extract data and write output for a given diagnostic.

    Args:
        diagnostics (dict):
            Dictionary containing information regarding how the diagnostics
            are to be processed.

            For example::

              {
                  "temperature": {
                      "diagnostic_name": "air_temperature",
                      "extrema": true,
                      "filepath": "temperature_at_screen_level",
                      "interpolation_method":
                          "model_level_temperature_lapse_rate",
                      "neighbour_finding": {
                          "land_constraint": false,
                          "method": "fast_nearest_neighbour",
                          "vertical_bias": null
                      }
                  }
              }

        neighbours (numpy.array):
            Array of neigbouring grid points that are associated with sites
            in the SortedDictionary of sites.

        sites (dict):
            A dictionary containing the properties of spotdata sites.

        ancillary_data (dict):
            A dictionary containing additional model data that is needed.
            e.g. {'orography': <cube of orography>}

        diagnostic_name (string):
            A string matching the keys in the diagnostics dictionary that
            will be used to access information regarding how the diagnostic
            is to be processed.

    Returns:
        (tuple): tuple containing:
            **resulting_cube** (iris.cube.Cube or None):
                Cube after extracting the diagnostic requested using the
                desired extraction method.
                None is returned if the "resulting_cubes" is an empty CubeList
                after processing.
            **extrema_cubes** (iris.cube.CubeList or None):
                CubeList containing extrema values, if the 'extrema' diagnostic
                is requested.
                None is returned if the value for diagnostic_dict["extrema"]
                is False, so that the extrema calculation is not required.

    """
    diagnostic_dict = diagnostics[diagnostic_name]

    # Grab the relevant set of grid point neighbours for the neighbour finding
    # method being used by this diagnostic.
    neighbour_hash = (
        construct_neighbour_hash(diagnostic_dict['neighbour_finding']))
    neighbour_list = neighbours[neighbour_hash]

    # Get optional kwargs that may be set to override defaults.
    optionals = ['upper_level', 'lower_level', 'no_neighbours',
                 'dz_tolerance', 'dthetadz_threshold', 'dz_max_adjustment']
    kwargs = {}
    if ancillary_data.get('config_constants') is not None:
        for optional in optionals:
            constant = ancillary_data.get('config_constants').get(optional)
            if constant is not None:
                kwargs[optional] = constant

    # Create a list of datetimes to loop through.
    forecast_times = []
    for cube in diagnostic_dict["data"]:
        time = cube.coord("time")
        forecast_times.extend(time.units.num2date(time.points))

    # Create empty iris.cube.CubeList to hold extracted data cubes.
    resulting_cubes = CubeList()

    # Loop over forecast times.
    for a_time in forecast_times:
        # Extract Cube from CubeList at current time.
        time_extract = datetime_constraint(a_time)
        cube = extract_cube_at_time(
            diagnostic_dict["data"], a_time, time_extract)
        if cube is None:
            # If no cube is available at given time, try the next time.
            continue

        ad = {}
        if diagnostic_dict["additional_data"] is not None:
            # Extract additional diagnostics at current time.
            ad = extract_ad_at_time(diagnostic_dict["additional_data"], a_time,
                                    time_extract)

        args = (cube, sites, neighbour_list, ancillary_data, ad)

        # Extract diagnostic data using defined method.
        resulting_cubes.append(
            ExtractData(
                diagnostic_dict['interpolation_method']).process(
                    *args, **kwargs))

    if resulting_cubes:
        # Concatenate CubeList into Cube for cubes with different
        # forecast times.
        resulting_cube = resulting_cubes.concatenate_cube()
    else:
        resulting_cube = None

    if diagnostic_dict['extrema']:
        extrema_cubes = (
            ExtractExtrema(24, start_hour=9).process(resulting_cube.copy()))
        extrema_cubes = extrema_cubes.merge()
    else:
        extrema_cubes = None

    return resulting_cube, extrema_cubes
