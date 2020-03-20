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
"""Script to run wind downscaling."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(wind_speed: cli.inputcube,
            sigma: cli.inputcube,
            target_orography: cli.inputcube,
            standard_orography: cli.inputcube,
            silhouette_roughness: cli.inputcube,
            vegetative_roughness: cli.inputcube = None,
            *,
            model_resolution: float,
            output_height_level: float = None,
            output_height_level_units='m'):
    """Wind downscaling.

    Run wind downscaling to apply roughness correction and height correction
    to wind fields as described in Howard and Clark (2007). All inputs must
    be on the same standard grid.

    Args:
        wind_speed (iris.cube.Cube):
            Cube of wind speed on standard grid.
            Any units can be supplied.
        sigma (iris.cube.Cube):
            Cube of standard deviation of model orography height.
            Units of field: m.
        target_orography (iris.cube.Cube):
            Cube of orography to downscale fields to.
            Units of field: m.
        standard_orography (iris.cube.Cube):
            Cube of orography on standard grid. (interpolated model orography).
            Units of field: m.
        silhouette_roughness (iris.cube.Cube):
            Cube of model silhouette roughness.
            Units of field: dimensionless.
        vegetative_roughness (iris.cube.Cube):
            Cube of vegetative roughness length.
            Units of field: m.
        model_resolution (float):
            Original resolution of model orography (before interpolation to
            standard grid)
            Units of field: m.
        output_height_level (float):
            If only a single height level is desired as output from
            wind-downscaling, this option can be used to select the height
            level. If no units are provided with 'output_height_level_units',
            metres are assumed.
        output_height_level_units (str):
            If a single height level is selected as output using
            'output_height_level', this additional argument may be used to
            specify the units of the value entered to select the level.
            e.g hPa.

    Returns:
        iris.cube.Cube:
            The processed Cube.

    Rises:
        ValueError:
            If the requested height value is not found.

    """
    import warnings

    import iris
    import numpy as np
    from iris.exceptions import CoordinateNotFoundError

    from improver.utilities.cube_extraction import apply_extraction
    from improver.wind_calculations import wind_downscaling

    if output_height_level_units and output_height_level is None:
        warnings.warn('output_height_level_units has been set but no '
                      'associated height level has been provided. These units '
                      'will have no effect.')
    try:
        wind_speed_iterator = wind_speed.slices_over('realization')
    except CoordinateNotFoundError:
        wind_speed_iterator = [wind_speed]
    wind_speed_list = iris.cube.CubeList()
    for wind_speed_slice in wind_speed_iterator:
        result = (
            wind_downscaling.RoughnessCorrection(
                silhouette_roughness, sigma, target_orography,
                standard_orography, model_resolution,
                z0_cube=vegetative_roughness,
                height_levels_cube=None)(wind_speed_slice))
        wind_speed_list.append(result)
    # TODO: Remove temporary fix for chunking problems when merging cubes
    max_npoints = max([np.prod(cube.data.shape) for cube in wind_speed_list])
    while iris._lazy_data._MAX_CHUNK_SIZE < max_npoints:
        iris._lazy_data._MAX_CHUNK_SIZE *= 2
    wind_speed = wind_speed_list.merge_cube()
    non_dim_coords = [x.name() for x in wind_speed.coords(dim_coords=False)]
    if 'realization' in non_dim_coords:
        wind_speed = iris.util.new_axis(wind_speed, 'realization')
    if output_height_level is not None:
        constraints = {'height': output_height_level}
        units = {'height': output_height_level_units}
        single_level = apply_extraction(
            wind_speed, iris.Constraint(**constraints), units)
        if not single_level:
            raise ValueError(
                'Requested height level not found, no cube '
                'returned. Available height levels are:\n'
                '{0:}\nin units of {1:}'.format(
                    wind_speed.coord('height').points,
                    wind_speed.coord('height').units))
        wind_speed = single_level
    return wind_speed
