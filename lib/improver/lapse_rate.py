# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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

import numpy as np
from numpy.linalg import lstsq
from scipy.ndimage import generic_filter

import iris
from iris.analysis.maths import multiply
from iris.exceptions import CoordinateNotFoundError

from improver.utilities.cube_checker import spatial_coords_match
from improver.utilities.cube_manipulation import enforce_float32_precision
from improver.constants import DALR


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
        adjusted_temperature (iris.cube.Cube):
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


class SaveNeighbourhood(object):
    """Saves the neighbourhood around each central point.

    The "generic_filter" module extracts the neighbourhood around each
    point as "buffer". This buffer is passed to the "SaveNeighbourhood" class.
    The "filter" function then saves this buffer into the "allbuffers" array.

    """
    def __init__(self, allbuffers):
        """Initialise the class.

        Create the global variables that allows the "filter" function
        to save each extracted buffer into "allbuffers".

        """
        # Initialises the iterator.
        self.i = 0
        # Saves all the buffers into this array.
        self.allbuffers = allbuffers

    def filter(self, buffer):
        """Defines filter function to be applied to extracted buffers.

        Saves the contents of the buffer into "allbuffers" array. Therefore
        a return value isn't required. However "generic_filter"
        requires a return value - so use zero.

        Args:
            buffer (array):
                Array containing neighourbood points.

        Returns:
            zero (float)
                Blank return value required by "generic_filter".

        """
        self.allbuffers[self.i, :] = buffer
        self.i += 1
        return 0.0


class LapseRate(object):
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
    2) Extracts neighbourhoods from both datasets:
       Apply "generic_filter" image processing module to temperature data
       to extract a neighbourhood for each single point and save this
       neighbourhood into a larger array.
       Repeat for orography data.
       generic_filter mode="constant" ensures that points beyond edges of
       dataset will be filled with "cval". (NaN in this case)
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

        # generic_filter extracts the neighbourhood and returns a 1D array.
        # ind_central_point indicates where the central point would be on
        # this array
        self.nbhoodarray_size = self.nbhood_size**2
        self.ind_central_point = int(self.nbhoodarray_size/2)

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        desc = ('<LapseRate: max_height_diff: {}, nbhood_radius: {},'
                'max_lapse_rate: {}, min_lapse_rate: {}>'
                .format(self.max_height_diff, self.nbhood_radius,
                        self.max_lapse_rate, self.min_lapse_rate))
        return desc

    def _calc_lapse_rate(self, temperature, orography):
        """Function to calculate the lapse rate.

        This holds the function to determine the local lapse rate at a point by
        calculating a least-squares fit to local temperature and altitude data
        to find the local lapse rate.

        Args:
            temperature(1D np.array):
                Contains the temperature values for the central point and its
                neighbours.

            orography(1D np.array):
                Contains the height values for the central point and its
                neighbours.

        Returns:
            gradient (float):
                The gradient of the temperature/orography values. This
                represents the lapse rate.

        """

        # If central point NaN then return blank value.
        if np.isnan(temperature[self.ind_central_point]):
            return DALR

        # Remove points where there are NaN temperature values from both arrays
        # before calculation.
        y_data = temperature[~np.isnan(temperature)]
        x_data = orography[~np.isnan(temperature)]

        # Return DALR if standard deviation of both datasets = 0 (where all
        # points are the same value).
        if np.isclose(np.std(x_data), 0.0) and np.isclose(np.std(y_data), 0.0):
            return DALR

        matrix = np.stack([x_data, np.ones(len(x_data))], axis=0).T
        gradient, _ = lstsq(matrix, y_data, rcond=None)[0]

        return gradient

    def _create_heightdiff_mask(self, all_orog_subsections):
        """
        Function to create a mask for any neighbouring points where the height
        difference from the central point is greater than max_height_diff.

        Slice through the orography subsection array to remove central
        points.
        Extracts the height value of each central point and masks out the
        neighbouring points where their height difference is greater than
        the maximum.

        Args:
            all_orog_subsections(2D np.array):
               Each row contains the height values of each neighbourhood.

        Returns:
            height_diff_mask (np.ndarray):
                A 2D array of boolean values.

        """

        self.all_orog_subsections = all_orog_subsections

        central_points = self.all_orog_subsections[:, self.ind_central_point]
        central_points = np.swapaxes([central_points], 0, 1)

        height_diff = np.subtract(self.all_orog_subsections, central_points)
        height_diff = np.absolute(height_diff)

        height_diff_mask = np.where(height_diff >= self.max_height_diff, True,
                                    False)

        return height_diff_mask

    def process(self, temperature_cube, orography_cube, land_sea_mask_cube):
        """Calculates the lapse rate from the temperature and orography cubes.

        Args:
            temperature_cube (iris.cube.Cube):
                Cube of air temperatures (K).

            orography_cube (Iris.cube.Cube):
                Cube containing orography data (metres)

            land_sea_mask_cube (iris.cube.Cube):
                Cube containing a binary land-sea mask. True for land-points
                and False for Sea.

        Returns:
            lapse_rate_cube (iris.cube.Cube):
                Cube containing lapse rate (Km-1)

        Raises
        ------
        TypeError: If input cubes are not cubes
        ValueError: If input cubes are the wrong units.

        """

        if not isinstance(temperature_cube, iris.cube.Cube):
            msg = "Temperature input is not a cube, but {}"
            raise TypeError(msg.format(type(temperature_cube)))

        if not isinstance(orography_cube, iris.cube.Cube):
            msg = "Orography input is not a cube, but {}"
            raise TypeError(msg.format(type(orography_cube)))

        if not isinstance(land_sea_mask_cube, iris.cube.Cube):
            msg = "Land/Sea mask input is not a cube, but {}"
            raise TypeError(msg.format(type(land_sea_mask_cube)))

        # Converts cube units.
        temperature_cube.convert_units('K')
        orography_cube.convert_units('metres')

        enforce_float32_precision([temperature_cube])

        # Extract x/y co-ordinates.
        x_coord = temperature_cube.coord(axis='x').name()
        y_coord = temperature_cube.coord(axis='y').name()

        # Extract orography and land/sea mask data.
        orography_data = next(orography_cube.slices([y_coord,
                                                     x_coord])).data
        land_sea_mask = next(land_sea_mask_cube.slices([y_coord,
                                                        x_coord])).data
        # Fill sea points with NaN values.
        orography_data = np.where(land_sea_mask, orography_data, np.nan)

        # Extract data array dimensions to define output arrays.
        dataarray_shape = next(temperature_cube.slices([y_coord,
                                                        x_coord])).shape
        dataarray_size = dataarray_shape[0] * dataarray_shape[1]

        # Array containing all of the subsections extracted from data array.
        # Also enforce single precision to speed up calculations.
        all_temp_subsections = np.zeros(
            (dataarray_size, self.nbhoodarray_size), dtype=np.float32)
        all_orog_subsections = np.zeros(
            (dataarray_size, self.nbhoodarray_size), dtype=np.float32)

        # Attempts to extract realizations. If cube doesn't contain the
        # dimension then place within list.
        try:
            slices_over_realization = temperature_cube.slices_over(
                "realization")
        except iris.exceptions.CoordinateNotFoundError:
            slices_over_realization = [temperature_cube]

        # Creates cube list to hold lapse rate data.
        lapse_rate_cube_list = iris.cube.CubeList([])

        for temp_slice in slices_over_realization:

            # Create slice to store lapse rate values.
            lapse_rate_slice = temp_slice

            temperature_data = temp_slice.data

            # Fill sea points with NaN values. Can't use Numpy mask since not
            # recognised by "generic_filter" function.
            temperature_data = np.where(land_sea_mask, temperature_data,
                                        np.nan)

            # Saves all neighbourhoods into "all_temp_subsections".
            # cval is value given to points outside the array.
            fnc = SaveNeighbourhood(allbuffers=all_temp_subsections)
            generic_filter(temperature_data, fnc.filter,
                           size=self.nbhood_size,
                           mode='constant', cval=np.nan)

            fnc = SaveNeighbourhood(allbuffers=all_orog_subsections)
            generic_filter(orography_data, fnc.filter,
                           size=self.nbhood_size,
                           mode='constant', cval=np.nan)

            # height_diff_mask is True for points where the height
            # difference between the central point and its neighbours
            # is > max_height_diff.
            height_diff_mask = self._create_heightdiff_mask(
                all_orog_subsections)

            # Mask points with extreme height differences as NaN.
            all_orog_subsections = np.where(height_diff_mask, np.nan,
                                            all_orog_subsections)
            all_temp_subsections = np.where(height_diff_mask, np.nan,
                                            all_temp_subsections)

            # Loop through both arrays and find gradient of each subsection.
            # The gradient indicates lapse rate - save into another array.
            # TODO: This for loop is the bottleneck in the code and needs to
            # be parallelised.
            lapse_rate_array = [self._calc_lapse_rate(temp, orog)
                                for temp, orog in zip(all_temp_subsections,
                                                      all_orog_subsections)]

            lapse_rate_array = np.array(
                lapse_rate_array, dtype=np.float32).reshape(dataarray_shape)

            # Enforces upper and lower limits on lapse rate values.
            lapse_rate_array = np.where(lapse_rate_array < self.min_lapse_rate,
                                        self.min_lapse_rate, lapse_rate_array)
            lapse_rate_array = np.where(lapse_rate_array > self.max_lapse_rate,
                                        self.max_lapse_rate, lapse_rate_array)

            lapse_rate_slice.data = lapse_rate_array
            lapse_rate_cube_list.append(lapse_rate_slice)

        lapse_rate_cube = lapse_rate_cube_list.merge_cube()
        lapse_rate_cube.rename('air_temperature_lapse_rate')
        lapse_rate_cube.units = 'K m-1'

        return lapse_rate_cube
