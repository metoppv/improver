# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""
This module defines the utilities required for Ensemble Copula Coupling
plugins.

"""

from typing import List, Optional, Union

import cf_units as unit
import iris
import numpy as np
from cf_units import Unit
from iris.cube import Cube
from numpy import ndarray

from improver.ensemble_copula_coupling.constants import BOUNDS_FOR_ECDF


def concatenate_array_with_array_endpoints(
    data: ndarray, low_endpoint: float, high_endpoint: float,
) -> ndarray:
    """
    For an array, add lower and upper endpoints to the last dimension, which
    increases that dimension's length by 2.

    Args:
        data:
            Array of values
        low_endpoint:
            Number used to create an array of a constant value
            as the lower endpoint.
        high_endpoint:
            Number of used to create an array of a constant value
            as the upper endpoint.

    Returns:
        Array of values after padding with the low_endpoint and
        high_endpoint.
    """
    lower_array = np.full((*data.shape[0:-1], 1), low_endpoint, dtype=data.dtype)
    upper_array = np.full((*data.shape[0:-1], 1), high_endpoint, dtype=data.dtype)
    data = np.concatenate((lower_array, data, upper_array), axis=-1)
    return data


def choose_set_of_percentiles(
    no_of_percentiles: int, sampling: str = "quantile"
) -> List[float]:
    """
    Function to create percentiles.

    Args:
        no_of_percentiles:
            Number of percentiles.
        sampling:
            Type of sampling of the distribution to produce a set of
            percentiles e.g. quantile or random.

            Accepted options for sampling are:

            * Quantile: A regular set of equally-spaced percentiles aimed
                        at dividing a Cumulative Distribution Function into
                        blocks of equal probability.
            * Random: A random set of ordered percentiles.

    Returns:
        Percentiles calculated using the sampling technique specified.

    Raises:
        ValueError: if the sampling option is not one of the accepted options.

    References:
        For further details, Flowerdew, J., 2014.
        Calibrating ensemble reliability whilst preserving spatial structure.
        Tellus, Series A: Dynamic Meteorology and Oceanography, 66(1), pp.1-20.
        Schefzik, R., Thorarinsdottir, T.L. & Gneiting, T., 2013.
        Uncertainty Quantification in Complex Simulation Models Using Ensemble
        Copula Coupling.
        Statistical Science, 28(4), pp.616-640.
    """
    if sampling in ["quantile"]:
        # Generate percentiles from 1/N+1 to N/N+1.
        percentiles = np.linspace(
            1 / float(1 + no_of_percentiles),
            no_of_percentiles / float(1 + no_of_percentiles),
            no_of_percentiles,
        ).tolist()
    elif sampling in ["random"]:
        # Generate percentiles from 1/N+1 to N/N+1.
        # Random sampling doesn't currently sample the ends of the
        # distribution i.e. 0 to 1/N+1 and N/N+1 to 1.
        percentiles = np.random.uniform(
            1 / float(1 + no_of_percentiles),
            no_of_percentiles / float(1 + no_of_percentiles),
            no_of_percentiles,
        )
        percentiles = sorted(list(percentiles))
    else:
        msg = "Unrecognised sampling option '{}'".format(sampling)
        raise ValueError(msg)
    return [item * 100 for item in percentiles]


def create_cube_with_percentiles(
    percentiles: Union[List[float], ndarray],
    template_cube: Cube,
    cube_data: ndarray,
    cube_unit: Optional[Union[Unit, str]] = None,
) -> Cube:
    """
    Create a cube with a percentile coordinate based on a template cube.
    The resulting cube will have an extra percentile coordinate compared with
    the template cube. The shape of the cube_data should be the shape of the
    desired output cube.

    Args:
        percentiles:
            Ensemble percentiles. There should be the same number of
            percentiles as the first dimension of cube_data.
        template_cube:
            Cube to copy metadata from.
        cube_data:
            Data to insert into the template cube.
            The shape of the cube_data, excluding the dimension associated with
            the percentile coordinate, should be the same as the shape of
            template_cube.
            For example, template_cube shape is (3, 3, 3), whilst the cube_data
            is (10, 3, 3, 3), where there are 10 percentiles.
        cube_unit:
            The units of the data within the cube, if different from those of
            the template_cube.

    Returns:
        Cube containing a percentile coordinate as the leading dimension (or
        scalar percentile coordinate if single-valued)
    """
    # create cube with new percentile dimension
    cubes = iris.cube.CubeList([])
    for point in percentiles:
        cube = template_cube.copy()
        cube.add_aux_coord(
            iris.coords.AuxCoord(
                np.float32(point), long_name="percentile", units=unit.Unit("%")
            )
        )
        cubes.append(cube)
    result = cubes.merge_cube()

    # replace data and units
    result.data = cube_data
    if cube_unit is not None:
        result.units = cube_unit

    return result


def get_bounds_of_distribution(bounds_pairing_key: str, desired_units: Unit) -> ndarray:
    """
    Gets the bounds of the distribution and converts the units of the
    bounds_pairing to the desired_units.

    This method gets the bounds values and units from the imported
    dictionaries: BOUNDS_FOR_ECDF and units_of_BOUNDS_FOR_ECDF.
    The units of the bounds are converted to be the desired units.

    Args:
        bounds_pairing_key:
            Name of key to be used for the BOUNDS_FOR_ECDF dictionary, in order
            to get the desired bounds_pairing.
        desired_units:
            Units to which the bounds_pairing will be converted.

    Returns:
        Lower and upper bound to be used as the ends of the
        empirical cumulative distribution function, converted to have
        the desired units.

    Raises:
        KeyError: If the bounds_pairing_key is not within the BOUNDS_FOR_ECDF
            dictionary.
    """
    # Extract bounds from dictionary of constants.
    try:
        bounds_pairing = BOUNDS_FOR_ECDF[bounds_pairing_key].value
        bounds_pairing_units = BOUNDS_FOR_ECDF[bounds_pairing_key].units
    except KeyError as err:
        msg = (
            "The bounds_pairing_key: {} is not recognised "
            "within BOUNDS_FOR_ECDF {}. \n"
            "Error: {}".format(bounds_pairing_key, BOUNDS_FOR_ECDF, err)
        )
        raise KeyError(msg)
    bounds_pairing_units = unit.Unit(bounds_pairing_units)
    bounds_pairing = bounds_pairing_units.convert(
        np.array(bounds_pairing), desired_units
    )
    return bounds_pairing


def restore_non_percentile_dimensions(
    array_to_reshape: ndarray, original_cube: Cube, n_percentiles: int
) -> ndarray:
    """
    Reshape a 2d array, so that it has the dimensions of the original cube,
    whilst ensuring that the probabilistic dimension is the first dimension.

    Args:
        array_to_reshape:
            The array that requires reshaping.  This has dimensions "percentiles"
            by "points", where "points" is a flattened array of all the other
            original dimensions that needs reshaping.
        original_cube:
            Cube slice containing the desired shape to be reshaped to, apart from
            the probabilistic dimension.  This would typically be expected to be
            either [time, y, x] or [y, x].
        n_percentiles:
            Length of the required probabilistic dimension ("percentiles").

    Returns:
        The array after reshaping.

    Raises:
        ValueError: If the probabilistic dimension is not the first on the
            original_cube.
        CoordinateNotFoundError: If the input_probabilistic_dimension_name is
            not a coordinate on the original_cube.
    """
    shape_to_reshape_to = list(original_cube.shape)
    if n_percentiles > 1:
        shape_to_reshape_to = [n_percentiles] + shape_to_reshape_to
    return array_to_reshape.reshape(shape_to_reshape_to)


def flatten_and_interpolate(
    desired_percentiles: Union[list, ndarray],
    original_percentiles: ndarray,
    data: ndarray,
    max_value: float = 101,
) -> ndarray:
    """
    Interpolates percentiles or probabilities in a semi-vectorised manner to
    provide greater speed. The interpolation is performed on a cube with
    3-dimensions. The last dimension is equivalent to xp in usual interpolation
    syntax. The leading dimension is looped over as a compromise between
    vectorisation and memory requirements given typical array sizes. Between
    these the dimensions are collapsed and the interpolation operation is
    vectorised.

    The method works by applying the numpy interpolation functionality to large
    1-dimensional arrays. These are constructed by repeating the original and
    desired percentile sequences N times in 1-dimensional arrays, where N is the
    length of the grid dimension being vectorised. The data taken from different
    grid points along the dimension being vectorised that corresponds to the
    original percentiles is also placed end-to-end in a single 1-dimensional
    array. Finally, each repeated sequence of percentiles in both the original
    and desired sets is offset by some value (max_value). Interpolation is then
    used with the three long 1-dimensional arrays in a single step.

    For a typical spatial grid the vectorisation applied along say x, but a loop
    is used over y. This reduces the amount of memory used whilst providing a
    speed up.

    This method will become slower for increasingly large percentile coordinates
    but provides a significant speed up for typical numbers of percentiles
    (i.e. up to around 20).

    Args:
        desired_percentiles:
            The target percentiles.
        original_percentiles:
            The percentiles at which the data is provided.
        data:
            The data to be interpolated.
        max_value:
            A value beyond the maximum potential value of the percentile
            coordinate, defaults to 101. Used to calculate an offset for
            vectorisation. The number should not be equivalent to the maximum
            percentile value (e.g. 100) as this will not give sufficiently
            large offsets.

    Returns:
        The forecast values interpolated to the desired_percentiles.
    """

    def calculate_offset(percentiles_size: int, dim_size: int):
        """Calculate offsets, each step has a length determined by the variable
        percentile_size. The number of steps is set by dim_size. The increment
        is set by max_value. An 1-dimensional array describing the offsets is
        returned."""
        steps = np.ones((percentiles_size * dim_size), dtype=np.int) * max_value
        for i in range(0, dim_size):
            steps[i * percentiles_size : (i + 1) * percentiles_size] *= i
        return steps

    desired_percentiles = np.array(desired_percentiles)
    loop_size = data.shape[0]
    # Calculate the remaining array size that is not part of the looping (first)
    # dimension or the percentile (last) dimension.
    vectorised_dim_size = data[..., -1].size // loop_size

    # Flatten input and target percentiles and tile. Float64 to ensure
    # interpolation gives correct results.
    orig_flat = np.tile(original_percentiles, vectorised_dim_size).astype(np.float64)
    desired_flat = np.tile(desired_percentiles, vectorised_dim_size).astype(np.float64)
    # Offset the repeated sets of percentiles to make them distinct
    orig_flat += calculate_offset(original_percentiles.size, vectorised_dim_size)
    desired_flat += calculate_offset(desired_percentiles.size, vectorised_dim_size)

    # Interpolate along one entire span of the looping (first) dimension in
    # each iteration of the loop before reshaping the output.
    result = np.empty(
        (loop_size, vectorised_dim_size, desired_percentiles.size), dtype=np.float32
    )
    for index in range(loop_size):
        result[index] = np.interp(
            desired_flat, orig_flat, data[index].reshape(-1),
        ).reshape(vectorised_dim_size, -1)

    # Move the percentile dimension to be the leading dimension.
    return np.moveaxis(result, -1, 0)
