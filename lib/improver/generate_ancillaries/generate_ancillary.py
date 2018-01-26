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
"""Module containing ancillary generation utilities for Improver"""

from cf_units import Unit
import iris
import numpy as np


def _make_mask_cube(
        mask_data, key, coords, topographic_bounds, topographic_units):
    """
    Makes cube from numpy masked array generated from orography fields.

    Args:
        mask_data (numpy masked array):
            The numpy array to make a cube from.
        key (string):
            Key from THRESHOLD_DICT which describes type of topography band.
        coords (dictionary):
            Dictionary of coordinate on the model ancillary file.
        topographic_bounds(list):
            List containing the lower and upper thresholds defining the mask
        topographic_units (string):
            Name of the units of the topographic zone coordinate of the output
            cube.

    Returns:
        mask_cube (cube):
            Cube containing the mask_data array, with appropriate coordinate
            and attribute information.
    """
    mask_cube = iris.cube.Cube(mask_data, long_name='Topography mask')
    if any([item is None for item in topographic_bounds]):
        msg = ("The topographic bounds variable should have both an "
               "upper and lower limit: "
               "Your topographic_bounds are {}")
        raise TypeError(msg.format(topographic_bounds))
    elif len(topographic_bounds) != 2:
        msg = ("The topographic bounds variable should have only an "
               "upper and lower limit: "
               "Your topographic_bounds variable has length {}")
        raise TypeError(msg.format(len(topographic_bounds)))
    else:
        coord_name = 'topographic_zone'
        central_point = np.mean(topographic_bounds)
        threshold_coord = iris.coords.AuxCoord(central_point,
                                               bounds=topographic_bounds,
                                               long_name=coord_name,
                                               units=Unit(topographic_units))
        mask_cube.add_aux_coord(threshold_coord)
    mask_cube.attributes['Topographical Type'] = key.title()
    for coord in coords:
        if coord.name() in ['projection_y_coordinate', 'latitude']:
            mask_cube.add_dim_coord(coord, 0)
        elif coord.name() in ['projection_x_coordinate', 'longitude']:
            mask_cube.add_dim_coord(coord, 1)
        else:
            mask_cube.add_aux_coord(coord)
    mask_cube = iris.util.new_axis(mask_cube, scalar_coord=coord_name)
    return mask_cube


class CorrectLandSeaMask(object):
    """
    Round landsea mask to binary values

    Corrects interpolated land sea masks to boolean values of
    False [sea] and True [land].
    """
    def __init__(self):
        pass

    def __repr__(self):
        """Represent the configured plugin instance as a string"""
        result = ('<CorrectLandSeaMask>')
        return result

    @staticmethod
    def process(standard_landmask):
        """Read in the interpolated landmask and round values < 0.5 to False
             and values >=0.5 to True.

        Args:
            standard_landmask:
                input landmask on standard grid.

        Returns:
            standard_landmask (cube):
                output landmask of boolean values.
        """
        mask_sea = np.ma.masked_less(standard_landmask.data, 0.5).mask
        standard_landmask.data[mask_sea] = False
        mask_land = np.ma.masked_greater(standard_landmask.data, 0.).mask
        standard_landmask.data[mask_land] = True
        return standard_landmask


class GenerateOrographyBandAncils(object):
    """
    Generate topographic band ancillaries for the standard grids.

    Reads orography files, then generates binary mask
    of land points within the orography band specified.
    """
    def __init__(self):
        pass

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<GenerateOrographyBandAncils>')
        return result

    @staticmethod
    def sea_mask(landmask, orog_band, sea_fill_value=None):
        """
        Function to mask sea points and substitute the default numpy
        fill value behind this mask_cube.

        Args:
            landmask (numpy array):
                The landmask generated by gen_landmask.
            orog_band (numpy array):
                The binary array to which the landmask will be applied.

        Keyword Args:
            sea_fill_value (float):
                A fill value to set sea points to and leave the output
                unmasked, rather than the default behaviour of returning a
                masked array with a default fill value.

        Returns:
            mask_data (numpy array):
                An array where the sea points have been masked out and filled
                with a default fill value, or just filled with the given
                sea_fill_value and not masked.
        """
        points_to_mask = np.logical_not(landmask)

        if sea_fill_value is None:
            mask_data = np.ma.masked_where(points_to_mask, orog_band)
            sea_fill_value = np.ma.default_fill_value(mask_data.data)
            mask_data.data[points_to_mask] = sea_fill_value
        else:
            mask_data = orog_band
            mask_data[points_to_mask] = sea_fill_value
        return mask_data

    def gen_orography_masks(
            self, standard_orography, standard_landmask, key, thresholds,
            units='m'):
        """
        Function to generate topographical band masks.

        For each threshold defined in 'thresholds', a cube with 0 over sea
        points and 1 for land points within the topography band will be
        generated.
        The lower threshold is exclusive to the band whilst the upper
        threshold is inclusive i.e:
        lower_threshold < band <= upper_threshold


        For example, for threshold pair [1,3] with orography::

                     [[0 0 2]    and      sea mask: [[-- -- 2]
                      [0 2 3]                        [0  2  3]
                      [0 1 4]]                       [-- 1  4]]

        the resultant array will be::

                     [[0 0 1]
                      [0 1 1]
                      [0 0 0]]

        Args:
            standard_orography (iris.cube.Cube):
                The standard orography.
            standard_landmask (iris.cube.Cube):
                The landmask generated by gen_landmask.
            key (string):
                Key from THRESHOLD_DICT which describes type of topography
                band. The key may currently be:

                    - land
                    - any_surface_type

            thresholds(list):
                Upper and/or lower thresholds of the current topographical
                band.

        Keyword Args:
            units (string):
                Units to be fed to CF_units to create a unit for the cube.
                The unit must be convertable to meters. If no unit is given
                this will default to meters.

        Returns:
            mask_cube (cube):
                Cube containing topographical band mask.

        Raises:
            KeyError: if the key does not match any in THRESHOLD_DICT.
        """
        thresholds = Unit(units).convert(
            np.array(thresholds), standard_orography.units)
        coords = standard_orography.coords()

        lower_threshold, upper_threshold = thresholds
        orog_band = np.ma.masked_where(
            np.ma.logical_and(
                (standard_orography.data > lower_threshold),
                (standard_orography.data <= upper_threshold)),
            standard_orography.data).mask.astype(int)

        if key == 'land':  # topographical bands above land only
            if standard_landmask is None:
                msg = ('To generate topography bands only over land, a land '
                       'mask must be provided as input')
                raise IOError(msg)

            if not isinstance(orog_band, np.ndarray):
                orog_band = np.zeros(standard_orography.data.shape).astype(int)

            mask_data = self.sea_mask(standard_landmask.data, orog_band,
                                      sea_fill_value=0)
            mask_cube = _make_mask_cube(
                mask_data, key, coords, topographic_bounds=thresholds,
                topographic_units=standard_orography.units)
        elif key == 'any_surface_type':
            mask_cube = _make_mask_cube(
                orog_band, key, coords, topographic_bounds=thresholds,
                topographic_units=standard_orography.units)
        else:
            msg = 'Unknown threshold_dict key: {}'
            raise KeyError(msg.format(key))
        mask_cube.units = Unit('1')
        return mask_cube

    def process(self, orography, thresholds_dict, landmask=None):
        """Loops over the supplied orographic bands, adding a cube
           for each band to the mask cubelist.

        Args:
            orography (iris.cube.Cube):
                orography on standard grid.

            threshold_dict (dictionary):
                definition of orography bands required.

        Keyword Args:
            landmask (iris.cube.Cube):
                land mask on standard grid. If provided and threshold_dict
                key is "land", sea points are masked.

        Returns:
            cubelist (iris.cube.CubeList):
              list of orographic band mask cubes.
        """
        cubelist = iris.cube.CubeList()
        for dict_key, bounds_dict in thresholds_dict.iteritems():
            if len(bounds_dict) == 0:
                msg = 'No threshold(s) found for topographic type: {}'
                raise ValueError(msg.format(dict_key))

            for limits in bounds_dict['bounds']:
                oro_band = self.gen_orography_masks(
                    orography, landmask, dict_key,
                    limits, bounds_dict['units'])
                cubelist.append(oro_band)
        return cubelist
