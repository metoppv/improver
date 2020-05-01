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
import numpy.ma as ma
from iris.exceptions import CoordinateNotFoundError

from improver import BasePlugin, PostProcessingPlugin
from improver.constants import DALR
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities import mathematical_operations, neighbourhood_tools
from improver.utilities.cube_checker import spatial_coords_match
from improver.utilities.cube_manipulation import (
    enforce_coordinate_ordering,
    get_dim_coord_names,
)


class ApplyGriddedLapseRate(PostProcessingPlugin):
    """Class to apply a lapse rate adjustment to a temperature data forecast"""

    @staticmethod
    def _check_dim_coords(temperature, lapse_rate):
        """Throw an error if the dimension coordinates are not the same for
        temperature and lapse rate cubes

        Args:
            temperature (iris.cube.Cube)
            lapse_rate (iris.cube.Cube)
        """
        for crd in temperature.coords(dim_coords=True):
            try:
                if crd != lapse_rate.coord(crd.name()):
                    raise ValueError(
                        'Lapse rate cube coordinate "{}" does not match '
                        "temperature cube coordinate".format(crd.name())
                    )
            except CoordinateNotFoundError:
                raise ValueError(
                    "Lapse rate cube has no coordinate " '"{}"'.format(crd.name())
                )

    def _calc_orog_diff(self, source_orog, dest_orog):
        """Get difference in orography heights, in metres

        Args:
            source_orog (iris.cube.Cube):
                2D cube of source orography heights (units modified in place)
            dest_orog (iris.cube.Cube):
                2D cube of destination orography heights (units modified in
                place)

        Returns:
            iris.cube.Cube
        """
        source_orog.convert_units("m")
        dest_orog.convert_units("m")
        orog_diff = next(dest_orog.slices(self.xy_coords)) - next(
            source_orog.slices(self.xy_coords)
        )
        return orog_diff

    def process(self, temperature, lapse_rate, source_orog, dest_orog):
        """Applies lapse rate correction to temperature forecast.  All cubes'
        units are modified in place.

        Args:
            temperature (iris.cube.Cube):
                Input temperature field to be adjusted
            lapse_rate (iris.cube.Cube):
                Cube of pre-calculated lapse rates
            source_orog (iris.cube.Cube):
                2D cube of source orography heights
            dest_orog (iris.cube.Cube):
                2D cube of destination orography heights

        Returns:
            iris.cube.Cube:
                Lapse-rate adjusted temperature field, in Kelvin
        """
        lapse_rate.convert_units("K m-1")
        self.xy_coords = [lapse_rate.coord(axis="y"), lapse_rate.coord(axis="x")]

        self._check_dim_coords(temperature, lapse_rate)

        if not spatial_coords_match(temperature, source_orog):
            raise ValueError(
                "Source orography spatial coordinates do not match " "temperature grid"
            )

        if not spatial_coords_match(temperature, dest_orog):
            raise ValueError(
                "Destination orography spatial coordinates do not match "
                "temperature grid"
            )

        orog_diff = self._calc_orog_diff(source_orog, dest_orog)

        adjusted_temperature = []
        for lr_slice, t_slice in zip(
            lapse_rate.slices(self.xy_coords), temperature.slices(self.xy_coords)
        ):
            newcube = t_slice.copy()
            newcube.convert_units("K")
            newcube.data += np.multiply(orog_diff.data, lr_slice.data)
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

    def __init__(
        self,
        max_height_diff=35,
        nbhood_radius=7,
        max_lapse_rate=-3 * DALR,
        min_lapse_rate=DALR,
    ):
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
        self.nbhood_size = int((2 * nbhood_radius) + 1)

        # Used in the neighbourhood checks, ensures that the center
        # of the array is non NaN.
        self.ind_central_point = self.nbhood_size // 2

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        desc = (
            "<LapseRate: max_height_diff: {}, nbhood_radius: {},"
            "max_lapse_rate: {}, min_lapse_rate: {}>".format(
                self.max_height_diff,
                self.nbhood_radius,
                self.max_lapse_rate,
                self.min_lapse_rate,
            )
        )
        return desc

    def _create_windows(self, temp, orog, mask):
        """Uses neighbourhood tools to pad and generate rolling windows
        of the temp and orog datasets.

        Args:
            temp (numpy.ndarray):
                2D array (single realization) of temperature data, in Kelvin
            orog (numpy.ndarray):
                2D array of orographies, in metres
            mask (numpy.ndarray):
                2D array with land-sea mask

        Returns:
            (tuple): tuple_containing:
                **views of temp** (numpy.ndarray):
                    Rolling windows of the padded temperature dataset.
                **views of orog** (numpy.ndarray):
                    Rolling windows of the padded orography dataset.
        """
        window_shape = (self.nbhood_size, self.nbhood_size)
        mask = ~mask.astype(np.bool) | np.isnan(temp)
        # Note: neither 'pad' nor 'as_strided' support masked arrays, so
        #       'pad_and_roll' mask and data separately as a workaround
        mask_windows = neighbourhood_tools.pad_and_roll(
            mask, window_shape, mode="constant", constant_values=True
        )
        orog_windows = neighbourhood_tools.pad_and_roll(
            orog, window_shape, mode="constant", constant_values=np.nan
        )
        temp_windows = neighbourhood_tools.pad_and_roll(
            temp, window_shape, mode="constant", constant_values=np.nan
        )
        orog_windows = ma.masked_array(orog_windows, mask=mask_windows, copy=False)
        temp_windows = ma.masked_array(temp_windows, mask=mask_windows, copy=False)
        return temp_windows, orog_windows

    def _generate_lapse_rate_array(
        self, temperature_data, orography_data, land_sea_mask_data
    ):
        """
        Calculate lapse rates and apply filters

        Args:
            temperature_data (numpy.ndarray):
                2D array (single realization) of temperature data, in Kelvin
            orography_data (numpy.ndarray):
                2D array of orographies, in metres
            land_sea_mask_data (numpy.ndarray):
                2D land-sea mask

        Returns:
            numpy.ndarray:
                Lapse rate values
        """
        # Preallocate output array
        lapse_rate_array = np.empty_like(temperature_data, dtype=np.float32)

        # Pads the data with nans and generates masked windows representing
        # a neighbourhood for each point.
        temp_nbhood_window, orog_nbhood_window = self._create_windows(
            temperature_data, orography_data, land_sea_mask_data
        )

        # Zips together the windows for temperature and orography
        # then finds the gradient of the surface temperature with
        # orography height - i.e. lapse rate.
        cnpt = self.ind_central_point
        axis = (-2, -1)
        for lapse, temp, orog in zip(
            lapse_rate_array, temp_nbhood_window, orog_nbhood_window
        ):
            # height_diff_mask is True for points where the height
            # difference between the central points and its
            # neighbours is greater then max_height_diff.
            orog_centre = orog[..., cnpt : cnpt + 1, cnpt : cnpt + 1]
            height_diff_mask = np.abs(orog - orog_centre) > self.max_height_diff

            temp = ma.masked_array(temp, mask=height_diff_mask, copy=False)

            # Masks orog to match temp
            orog = ma.masked_array(orog, mask=temp.mask, copy=False)

            grad = mathematical_operations.fast_linear_fit(
                orog, temp, axis=axis, gradient_only=True
            )

            # Checks that the standard deviations are not 0
            # i.e. there is some variance to fit a gradient to.
            tempcheck = np.isclose(np.std(temp, axis=axis), 0)
            orogcheck = np.isclose(np.std(orog, axis=axis), 0)
            # checks that our central point in the neighbourhood
            # is not masked.
            temp_mask_check = temp.mask[..., cnpt, cnpt]

            # Mask out combined checks
            grad[tempcheck | orogcheck | temp_mask_check] = ma.masked

            # Fill out the mask with DALR.
            grad = grad.filled(DALR)

            lapse[...] = grad

        # Enforce upper and lower limits on lapse rate values.
        lapse_rate_array = lapse_rate_array.clip(
            self.min_lapse_rate, self.max_lapse_rate
        )
        return lapse_rate_array

    def process(self, temperature, orography, land_sea_mask, model_id_attr=None):
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
        temperature_cube.convert_units("K")
        orography.convert_units("metres")

        # Extract x/y co-ordinates.
        x_coord = temperature_cube.coord(axis="x").name()
        y_coord = temperature_cube.coord(axis="y").name()

        # Extract orography and land/sea mask data.
        orography_data = next(orography.slices([y_coord, x_coord])).data
        land_sea_mask_data = next(land_sea_mask.slices([y_coord, x_coord])).data
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
                temperature_data, orography_data, land_sea_mask_data
            )
            lapse_rate_data.append(lapse_rate_array)
        lapse_rate_data = np.array(lapse_rate_data)
        if not has_realization_dimension:
            lapse_rate_data = np.squeeze(lapse_rate_data)

        attributes = generate_mandatory_attributes(
            [temperature], model_id_attr=model_id_attr
        )
        lapse_rate_cube = create_new_diagnostic_cube(
            "air_temperature_lapse_rate",
            "K m-1",
            temperature_cube,
            attributes,
            data=lapse_rate_data,
        )

        if original_dimension_order:
            enforce_coordinate_ordering(lapse_rate_cube, original_dimension_order)

        return lapse_rate_cube
