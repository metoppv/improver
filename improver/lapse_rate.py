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
from scipy.ndimage import generic_filter

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


class SaveNeighbourhood:
    """Saves the neighbourhood around each central point.

    The "generic_filter" module extracts the neighbourhood around each
    point as "buffer". This buffer is passed to the "SaveNeighbourhood" class.
    The "filter" function then saves this buffer into the "allbuffers" array.

    """
    def __init__(self, allbuffers):
        """Initialise the class.

        Create the global variables that allows the "filter" function
        to save each extracted buffer into "allbuffers".

        Args:
            allbuffers (numpy.ndarray):
                Where to save each extracted buffer.

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
            buffer (numpy.ndarray):
                Array containing neighourbood points.

        Returns:
            zero (float)
                Blank return value required by "generic_filter".

        """
        self.allbuffers[self.i, :] = buffer
        self.i += 1
        return 0.0


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
            temperature (1D numpy.ndarray):
                Contains the temperature values for the central point and its
                neighbours.

            orography (1D numpy.ndarray):
                Contains the height values for the central point and its
                neighbours.

        Returns:
            float:
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
            all_orog_subsections(2D numpy.ndarray):
               Each row contains the height values of each neighbourhood.

        Returns:
            numpy.ndarray:
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
        # Fill sea points with NaN values. Can't use Numpy mask since not
        # recognised by "generic_filter" function.
        temperature_data = np.where(
            land_sea_mask_data, temperature_data, np.nan)

        # Generate data neighbourhoods on which to calculate lapse rates
        # pylint: disable=unsubscriptable-object
        dataarray_size = temperature_data.shape[0] * temperature_data.shape[1]

        temp_nbhoods = np.zeros(
            (dataarray_size, self.nbhoodarray_size), dtype=np.float32)
        fnc = SaveNeighbourhood(allbuffers=temp_nbhoods)
        generic_filter(temperature_data, fnc.filter,
                       size=self.nbhood_size,
                       mode='constant', cval=np.nan)

        orog_nbhoods = np.zeros(
            (dataarray_size, self.nbhoodarray_size), dtype=np.float32)
        fnc = SaveNeighbourhood(allbuffers=orog_nbhoods)
        generic_filter(orography_data, fnc.filter,
                       size=self.nbhood_size,
                       mode='constant', cval=np.nan)

        # height_diff_mask is True for points where the height
        # difference between the central point and its neighbours
        # is > max_height_diff.
        height_diff_mask = self._create_heightdiff_mask(orog_nbhoods)

        # Mask points with extreme height differences as NaN.
        temp_nbhoods = np.where(height_diff_mask, np.nan, temp_nbhoods)
        orog_nbhoods = np.where(height_diff_mask, np.nan, orog_nbhoods)

        # Loop through both arrays and find gradient of surface temperature
        # with orography height - ie lapse rate.
        # TODO: This for loop is the bottleneck in the code and needs to
        # be parallelised.
        lapse_rate_array = [self._calc_lapse_rate(temp, orog)
                            for temp, orog in zip(temp_nbhoods, orog_nbhoods)]

        lapse_rate_array = np.array(
            lapse_rate_array, dtype=np.float32).reshape(
                (temperature_data.shape))

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
