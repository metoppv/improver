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
"""Module containing lapse rate calculation plugins."""

import iris
import numpy as np
from iris.analysis.maths import multiply
from iris.exceptions import CoordinateNotFoundError
from numpy.linalg import lstsq

from improver import BasePlugin
from improver.constants import DALR
from improver.metadata.utilities import (
    create_new_diagnostic_cube, generate_mandatory_attributes)
from improver.utilities.cube_checker import spatial_coords_match
from improver.utilities.cube_manipulation import (
    enforce_coordinate_ordering, get_dim_coord_names)


def apply_gridded_lapse_rate(temperature, lapse_rate, source_orog, dest_orog):
    """
    Function to apply a lapse rate adjustment to temperature data forecast
    at "source_orog" heights, to be applicable at "dest_orog" heights.

    Args:
        temperature (iris.cube.Cube):
            Input temperature field to be adjusted
        lapse_rate (iris.cube.Cube):
            Cube of pre-calculated lapse rates (units modified in place), which
            must match the temperature cube
        source_orog (iris.cube.Cube):
            2D cube of source orography heights (units modified in place)
        dest_orog (iris.cube.Cube):
            2D cube of destination orography heights (units modified in place)

    Returns:
        iris.cube.Cube:
            Lapse-rate adjusted temperature field
    """
    # check dimensions and coordinates match on input cubes
    for crd in temperature.coords(dim_coords=True):
        try:
            if crd != lapse_rate.coord(crd.name()):
                raise ValueError(
                    'Lapse rate cube coordinate "{}" does not match '
                    'temperature cube coordinate'.format(crd.name()))
        except CoordinateNotFoundError:
            raise ValueError('Lapse rate cube has no coordinate '
                             '"{}"'.format(crd.name()))

    if not spatial_coords_match(temperature, source_orog):
        raise ValueError(
            'Source orography spatial coordinates do not match '
            'temperature grid')
    if not spatial_coords_match(temperature, dest_orog):
        raise ValueError(
            'Destination orography spatial coordinates do not match '
            'temperature grid')

    # calculate height difference (in m) on which to adjust
    source_orog.convert_units('m')
    dest_orog.convert_units('m')
    orog_diff = (
        next(dest_orog.slices([dest_orog.coord(axis='y'),
                               dest_orog.coord(axis='x')])) -
        next(source_orog.slices([source_orog.coord(axis='y'),
                                 source_orog.coord(axis='x')])))

    # convert lapse rate cube to K m-1
    lapse_rate.convert_units('K m-1')

    # adjust temperatures
    adjusted_temperature = []
    for lrsubcube, tempsubcube in zip(
            lapse_rate.slices([lapse_rate.coord(axis='y'),
                               lapse_rate.coord(axis='x')]),
            temperature.slices([temperature.coord(axis='y'),
                                temperature.coord(axis='x')])):

        # calculate temperature adjustment in K
        adjustment = multiply(orog_diff, lrsubcube)

        # apply adjustment to each spatial slice of the temperature cube
        newcube = tempsubcube.copy()
        newcube.convert_units('K')
        newcube.data += adjustment.data
        adjusted_temperature.append(newcube)

    return iris.cube.CubeList(adjusted_temperature).merge_cube()


class LapseRate(BasePlugin):
    """
    Plugin to calculate the lapse rate from orography and temperature
    cubes.

    References:
        The method applied here is based on the method used in the 2010 paper:
        https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/met.177

    Code methodology:

    1) Apply land/sea mask to temperature and orography datasets. Mask sea
       points as NaN since image processing module does not recognise Numpy
       masks.
    2) Creates "views" of both datasets, where each view represents a
       neighbourhood of points. To do this, each array is padded with
       NaN values to a width of half the neighbourhood size.
    3) For all the stored orography neighbourhoods - take the neighbours around
       the central point and create a mask where the height difference from
       the central point is greater than 35m.
    4) Loop through array of neighbourhoods and take the height and temperature
       of all grid points and calculate the
       temperature/height gradient = lapse rate
    5) Constrain the returned lapse rates between min_lapse_rate and
       max_lapse_rate. These default to > DALR and < -3.0*DALR but are user
       configurable
    """

    def __init__(self, max_height_diff=35, nbhood_radius=7,
                 max_lapse_rate=-3*DALR, min_lapse_rate=DALR):
        """
        The class is called with the default constraints for the processing
        code.

        Args:
            max_height_diff (float):
                Maximum allowable height difference between the central point
                and points in the neighbourhood over which the lapse rate will
                be calculated (metres).
                The default value of 35m is from the referenced paper.

            nbhood_radius (int):
                Radius of neighbourhood around each point. The neighbourhood
                will be a square array with side length 2*nbhood_radius + 1.
                The default value of 7 is from the referenced paper.

            max_lapse_rate (float):
                Maximum lapse rate allowed.

            min_lapse_rate (float):
                Minimum lapse rate allowed.

        """

        self.max_height_diff = max_height_diff
        self.nbhood_radius = nbhood_radius
        self.max_lapse_rate = max_lapse_rate
        self.min_lapse_rate = min_lapse_rate

        if self.max_lapse_rate < self.min_lapse_rate:
            msg = "Maximum lapse rate is less than minimum lapse rate"
            raise ValueError(msg)

        if self.nbhood_radius < 0:
            msg = "Neighbourhood radius is less than zero"
            raise ValueError(msg)

        if self.max_height_diff < 0:
            msg = "Maximum height difference is less than zero"
            raise ValueError(msg)

        # nbhood_size=3 corresponds to a 3x3 array centred on the
        # central point.
        self.nbhood_size = int((2*nbhood_radius) + 1)

        # alinfit uses the central point to ensure that the center
        # of the neighbourhood is not nan.
        self.nbhoodarray_size = self.nbhood_size**2
        self.ind_central_point = int(self.nbhood_size/2)

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        desc = ('<LapseRate: max_height_diff: {}, nbhood_radius: {},'
                'max_lapse_rate: {}, min_lapse_rate: {}>'
                .format(self.max_height_diff, self.nbhood_radius,
                        self.max_lapse_rate, self.min_lapse_rate))
        return desc

    def _rolling_window(self, A, temp_shape, shape):
        """Creates views of the array A, this avoids
        creating a massive matrix of points in a neighbourhood
        calculation

        Args:
            A:
                The input array padded with nans for half the
                neighbourhood size (2D).
            temp_shape:
                The shape of the original temperature data.
            shape:
                The neighbourhood shape e.g. is the neighbourhood
                size is 3, the shape would be (3, 3) to create a
                3x3 array.

        Returns:
            ndarray of "views" into the data, each view represents
            a neighbourhood of points.
        """
        adjshp = (*temp_shape, *shape)
        strides = A.strides + A.strides
        return np.lib.stride_tricks.as_strided(
            A, shape=adjshp, strides=strides, writeable=False)

    def _create_windows(self, temp, orog):
        """Pads the input arrays and passes them to _rolling_window
        to create windows of the data.

        Args:
            temp:
                The surface temperature numpy array (2D).
            orog:
                The orography height numpy array (2D)

        Returns:
            2 ndarrays, both relating to "views" of the data,
            first:
                views into the temperature data
            second:
                views into the orography data.
        """
        padding_size = self.nbhood_size // 2
        shape = (self.nbhood_size, self.nbhood_size)
        temp_orig_shape = temp.shape

        temp = np.pad(temp, padding_size,
                      mode='constant', constant_values=np.nan)
        orog = np.pad(orog, padding_size,
                      mode='constant', constant_values=np.nan)

        return (self._rolling_window(temp, temp_orig_shape, shape),
                self._rolling_window(orog, temp_orig_shape, shape))

    def alinfit(self, orog, temp, mask=None, axis=-1):
        """Uses a simple linear fit approach to calculate the
        gradient (i.e. lapse rate)  between the orography height
        and the surface temperature. Although equivalent, this
        approach is much faster than using scipy lstsq to fit the
        data. Checks the data to make sure that the standard
        deviation isn't 0 for either arrays and that the central
        point in the neighbourhood isn't nan, if either of these
        are true, the dry adiabatic lapse rate is used (DALR).

        Args:
            orog:
                Views of the orography data (3D array). The final 2
                dimensions represent each point surrounded by its
                neighbourhood.
            temp:
                Views of the temperature data (similar to the
                orography above).
            mask:
                Optional argument, must match the shape of orog and
                temp (3D array), specifies which points to exclude
                from the linear fit by using a boolean to represent
                each point, True is used to include the data, False
                ise used to exclude it.
            axis:
                Optional argument, specifies the axis to operate on.

        Returns:
            1d numpy array of gradients.
        """
        if not isinstance(axis, tuple):
            axis = (axis,)
        if mask is None:
            N = np.sum(~np.isnan(orog))
        else:
            N = np.nansum(mask, axis=axis)
            orog = np.where(mask, orog, np.nan)
            temp = np.where(mask, temp, np.nan)

        orog = np.where(np.isnan(temp), np.nan, orog)

        # Finds a compatible shape for the means to be reshaped into
        assert orog.shape == temp.shape
        shape = list(orog.shape)
        for ax in axis:
            shape[ax] = 1
        
        X_diff = orog - np.nanmean(orog, axis=axis).reshape(shape)
        Y_diff = temp - np.nanmean(temp, axis=axis).reshape(shape)

        XY_cov = np.nansum(X_diff * Y_diff, axis=axis)
        X_var = np.nansum(X_diff * X_diff, axis=axis)

        grad = XY_cov / X_var

        ycheck = np.isclose(np.nanstd(temp, axis=axis), 0)
        xcheck = np.isclose(np.nanstd(orog, axis=axis), 0)
        grad = np.where(ycheck | xcheck, DALR, grad)

        y_nan_check = np.isnan(temp[..., self.ind_central_point,
                                    self.ind_central_point])
        grad = np.where(y_nan_check, DALR, grad)

        return grad

    def _create_height_diff_mask(self, orog_subsections):
        """Create a mask for any neighbouring points where the
        height difference from the central point is greater than
        max_height_diff.

        Args:
            orog_subsections:
                3D numpy array where the final 2 axes represent the
                orography neighbourhood data.

        Returns:
            3D numpy array containing boolean values the same shape
            as the orog_subsections. True is the orography height is
            lower than max_height_diff, False if not.
        """
        central_points = orog_subsections[...,
                                          self.ind_central_point,
                                          self.ind_central_point]
        central_points = central_points[np.newaxis, np.newaxis].T

        height_diff = np.absolute(np.subtract(orog_subsections,
                                              central_points))
        return np.where(height_diff < self.max_height_diff, True, False)

    def _generate_lapse_rate_array(
            self, temperature_data, orography_data, land_sea_mask_data):
        """
        Calculate lapse rates and apply filters

        Args:
            temperature_data (numpy.ndarray)
                2D array (single realization) of temperature data, in Kelvin
            orography_data (numpy.ndarray)
                2D array of orographies, in metres
            land_sea_mask_data (numpy.ndarray)
                2D land-sea mask

        Returns:
            numpy.ndarray
                Lapse rate values
        """
        # Fill sea points with NaN values.
        temperature_data = np.where(
            land_sea_mask_data, temperature_data, np.nan)

        lapse_rate_array = []
        # Pads the data with nans and generates windows representing
        # a neighbourhood for each point.
        temp_nbhood_window, orog_nbhood_window = self._create_windows(
            temperature_data, orography_data)

        # Zips together the windows for temperature and orography
        # then finds the gradient of the surface temperature with
        # orography height - i.e. lapse rate.
        for temp, orog in zip(temp_nbhood_window, orog_nbhood_window):
            # height_diff is True for points where the height
            # difference between the central points and its
            # neighbours is < max_height_diff.
            height_diff_mask = self._create_height_diff_mask(orog)
            grad = self.alinfit(orog, temp, mask=height_diff_mask,
                                axis=(-2, -1))
            lapse_rate_array.append(grad)
        lapse_rate_array = np.array(lapse_rate_array, dtype=np.float32)

        # Enforce upper and lower limits on lapse rate values.
        lapse_rate_array = np.where(lapse_rate_array < self.min_lapse_rate,
                                    self.min_lapse_rate, lapse_rate_array)
        lapse_rate_array = np.where(lapse_rate_array > self.max_lapse_rate,
                                    self.max_lapse_rate, lapse_rate_array)
        return lapse_rate_array

    def process(self, temperature, orography, land_sea_mask,
                model_id_attr=None):
        """Calculates the lapse rate from the temperature and orography cubes.

        Args:
            temperature (iris.cube.Cube):
                Cube of air temperatures (K).
            orography (iris.cube.Cube):
                Cube containing orography data (metres)
            land_sea_mask (iris.cube.Cube):
                Cube containing a binary land-sea mask. True for land-points
                and False for Sea.
            model_id_attr (str):
                Name of the attribute used to identify the source model for
                blending. This is inherited from the input temperature cube.

        Returns:
            iris.cube.Cube:
                Cube containing lapse rate (K m-1)

        Raises
        ------
        TypeError: If input cubes are not cubes
        ValueError: If input cubes are the wrong units.

        """
        if not isinstance(temperature, iris.cube.Cube):
            msg = "Temperature input is not a cube, but {}"
            raise TypeError(msg.format(type(temperature)))

        if not isinstance(orography, iris.cube.Cube):
            msg = "Orography input is not a cube, but {}"
            raise TypeError(msg.format(type(orography)))

        if not isinstance(land_sea_mask, iris.cube.Cube):
            msg = "Land/Sea mask input is not a cube, but {}"
            raise TypeError(msg.format(type(land_sea_mask)))

        # Converts cube units.
        temperature_cube = temperature.copy()
        temperature_cube.convert_units('K')
        orography.convert_units('metres')

        # Extract x/y co-ordinates.
        x_coord = temperature_cube.coord(axis='x').name()
        y_coord = temperature_cube.coord(axis='y').name()

        # Extract orography and land/sea mask data.
        orography_data = next(orography.slices([y_coord,
                                                x_coord])).data
        land_sea_mask_data = next(land_sea_mask.slices([y_coord,
                                                        x_coord])).data
        # Fill sea points with NaN values.
        orography_data = np.where(land_sea_mask_data, orography_data, np.nan)

        # Create list of arrays over "realization" coordinate
        has_realization_dimension = False
        original_dimension_order = None
        if temperature_cube.coords("realization", dim_coords=True):
            original_dimension_order = get_dim_coord_names(temperature_cube)
            enforce_coordinate_ordering(temperature_cube, "realization")
            temp_data_slices = temperature_cube.data
            has_realization_dimension = True
        else:
            temp_data_slices = [temperature_cube.data]

        # Calculate lapse rate for each realization
        lapse_rate_data = []
        for temperature_data in temp_data_slices:
            lapse_rate_array = self._generate_lapse_rate_array(
                temperature_data, orography_data, land_sea_mask_data)
            lapse_rate_data.append(lapse_rate_array)
        lapse_rate_data = np.array(lapse_rate_data)
        if not has_realization_dimension:
            lapse_rate_data = np.squeeze(lapse_rate_data)

        attributes = generate_mandatory_attributes(
            [temperature], model_id_attr=model_id_attr)
        lapse_rate_cube = create_new_diagnostic_cube(
            'air_temperature_lapse_rate', 'K m-1', temperature_cube,
            attributes, data=lapse_rate_data)

        if original_dimension_order:
            enforce_coordinate_ordering(
                lapse_rate_cube, original_dimension_order)

        return lapse_rate_cube
