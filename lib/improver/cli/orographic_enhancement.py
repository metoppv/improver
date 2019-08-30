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
"""Script to calculate orographic enhancement."""

import os

from improver.argparser import ArgParser
from improver.orographic_enhancement import OrographicEnhancement
from improver.utilities.cube_extraction import extract_subcube
from improver.utilities.filename import generate_file_name
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf
from improver.wind_calculations.wind_components import ResolveWindComponents


def load_and_extract(cube_filepath, height_value, units):
    """
    Function to load a cube, attempt to extract a height level.
    If no matching level is available an error is raised.

    Args:
        cube_filepath (str):
            Path to the input NetCDF file.
        height_value (float):
            The boundary height to be extracted with the input units.
        units (str):
            The units of the height level to be extracted.
    Returns:
        cube (iris.cube.Cube):
            A cube containing the extracted height level.
    Raises:
        ValueError: If height level is not found in the input cube.
    """
    cube = load_cube(cube_filepath)

    # Write constraint in this format so a constraint is constructed that
    # is suitable for floating point comparison
    height_constraint = ["height=[{}:{}]".format(height_value-0.1,
                                                 height_value+0.1)]
    cube = extract_subcube(cube, height_constraint, units=[units])

    if cube is not None:
        return cube

    raise ValueError('No data available from {} at height {}{}'.format(
            cube_filepath, height_value, units))


def main(argv=None):
    """Calculate orographic enhancement of precipitation from model pressure,
    temperature, relative humidity and wind input files"""

    parser = ArgParser(description='Calculate orographic enhancement using the'
                       ' ResolveWindComponents() and OrographicEnhancement() '
                       'plugins. Outputs data on the high resolution orography'
                       ' grid and regridded to the coarser resolution of the '
                       'input diagnostic variables.')

    parser.add_argument('temperature_filepath', metavar='TEMPERATURE_FILEPATH',
                        help='Full path to input NetCDF file of temperature on'
                        ' height levels')
    parser.add_argument('humidity_filepath', metavar='HUMIDITY_FILEPATH',
                        help='Full path to input NetCDF file of relative '
                        'humidity on height levels')
    parser.add_argument('pressure_filepath', metavar='PRESSURE_FILEPATH',
                        help='Full path to input NetCDF file of pressure on '
                        'height levels')
    parser.add_argument('windspeed_filepath', metavar='WINDSPEED_FILEPATH',
                        help='Full path to input NetCDF file of wind speed on '
                        'height levels')
    parser.add_argument('winddir_filepath', metavar='WINDDIR_FILEPATH',
                        help='Full path to input NetCDF file of wind direction'
                        ' on height levels')
    parser.add_argument('orography_filepath', metavar='OROGRAPHY_FILEPATH',
                        help='Full path to input NetCDF high resolution '
                        'orography ancillary. This should be on the same or a '
                        'finer resolution grid than the input variables, and '
                        'defines the grid on which the orographic enhancement '
                        'will be calculated.')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR', help='Directory '
                        'to write output orographic enhancement files')
    parser.add_argument('--boundary_height', type=float, default=1000.,
                        help='Model height level to extract variables for '
                        'calculating orographic enhancement, as proxy for '
                        'the boundary layer.')
    parser.add_argument('--boundary_height_units', type=str, default='m',
                        help='Units of the boundary height specified for '
                        'extracting model levels.')

    args = parser.parse_args(args=argv)

    constraint_info = (args.boundary_height, args.boundary_height_units)

    temperature = load_and_extract(args.temperature_filepath, *constraint_info)
    humidity = load_and_extract(args.humidity_filepath, *constraint_info)
    pressure = load_and_extract(args.pressure_filepath, *constraint_info)
    wind_speed = load_and_extract(args.windspeed_filepath, *constraint_info)
    wind_dir = load_and_extract(args.winddir_filepath, *constraint_info)

    # load high resolution orography
    orography = load_cube(args.orography_filepath)

    orogenh_high_res, orogenh_standard = process(
        temperature, humidity, pressure, wind_speed, wind_dir, orography)

    # generate file names
    fname_standard = os.path.join(
        args.output_dir, generate_file_name(orogenh_standard))
    fname_high_res = os.path.join(
        args.output_dir, generate_file_name(
            orogenh_high_res,
            parameter="orographic_enhancement_high_resolution"))

    # save output files
    save_netcdf(orogenh_standard, fname_standard)
    save_netcdf(orogenh_high_res, fname_high_res)


def process(temperature, humidity, pressure, wind_speed, wind_dir, orography):
    """Calculate orograhpic enhancement

    Uses the ResolveWindComponents() and OrographicEnhancement() plugins.
    Outputs data on the high resolution orography grid and regrided to the
    coarser resolution of the input diagnostic variables.

    Args:
        temperature (iris.cube.Cube):
             Cube containing temperature at top of boundary layer.
        humidity (iris.cube.Cube):
            Cube containing relative humidity at top of boundary layer.
        pressure (iris.cube.Cube):
            Cube containing pressure at top of boundary layer.
        wind_speed (iris.cube.Cube):
            Cube containing wind speed values.
        wind_dir (iris.cube.Cube):
            Cube containing wind direction values relative to true north.
        orography (iris.cube.Cube):
            Cube containing height of orography above sea level on high
            resolution (1 km) UKPP domain grid.

    Returns:
        (tuple): tuple containing:
                **orogenh_high_res** (iris.cube.Cube):
                    Precipitation enhancement due to orography in mm/h on the
                    UK standard grid, padded with masked up np.nans where
                    outside the UKPP domain.
                **orogenh_standard** (iris.cube.Cube):
                    Precipitation enhancement due to orography in mm/h on
                    the 1km Transverse Mercator UKPP grid domain.
    """
    # resolve u and v wind components
    u_wind, v_wind = ResolveWindComponents().process(wind_speed, wind_dir)
    # calculate orographic enhancement
    orogenh_high_res, orogenh_standard = OrographicEnhancement().process(
        temperature, humidity, pressure, u_wind, v_wind, orography)
    return orogenh_high_res, orogenh_standard


if __name__ == "__main__":
    main()
