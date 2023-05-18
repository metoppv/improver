# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
import warnings
from itertools import product
from typing import List, Optional, Union

import cf_units as unit
import iris
import numpy as np
from cf_units import Unit
from iris.cube import Cube
from numpy import ndarray

from improver.ensemble_copula_coupling.constants import BOUNDS_FOR_ECDF


def concatenate_2d_array_with_2d_array_endpoints(
    array_2d: ndarray, low_endpoint: float, high_endpoint: float,
) -> ndarray:
    """
    For a 2d array, add a 2d array as the lower and upper endpoints.
    The concatenation to add the lower and upper endpoints to the 2d array
    are performed along the second (index 1) dimension.

    Args:
        array_2d:
            2d array of values
        low_endpoint:
            Number used to create a 2d array of a constant value
            as the lower endpoint.
        high_endpoint:
            Number of used to create a 2d array of a constant value
            as the upper endpoint.

    Returns:
        2d array of values after padding with the low_endpoint and
        high_endpoint.
    """
    if array_2d.ndim != 2:
        raise ValueError("Expected 2D input, got {}D input".format(array_2d.ndim))
    lower_array = np.full((array_2d.shape[0], 1), low_endpoint, dtype=array_2d.dtype)
    upper_array = np.full((array_2d.shape[0], 1), high_endpoint, dtype=array_2d.dtype)
    array_2d = np.concatenate((lower_array, array_2d, upper_array), axis=1)
    return array_2d


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


def insert_lower_and_upper_endpoint_to_1d_array(
    array_1d: ndarray, low_endpoint: float, high_endpoint: float
) -> ndarray:
    """
    For a 1d array, add a lower and upper endpoint.

    Args:
        array_1d:
            1d array of values
        low_endpoint:
            Number of use as the lower endpoint.
        high_endpoint:
            Number of use as the upper endpoint.

    Returns:
        1d array of values padded with the low_endpoint and high_endpoint.
    """
    if array_1d.ndim != 1:
        raise ValueError("Expected 1D input, got {}D input".format(array_1d.ndim))
    lower_array = np.array([low_endpoint])
    upper_array = np.array([high_endpoint])
    array_1d = np.concatenate((lower_array, array_1d, upper_array))
    if array_1d.dtype == np.float64:
        array_1d = array_1d.astype(np.float32)
    return array_1d


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


def slow_interp_same_x(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """For each row i of fp, calculate np.interp(x, xp, fp[i, :]).
    Args:
        x: 1-D array
        xp: 1-D array, sorted in non-decreasing order
        fp: 2-D array with len(xp) columns
    Returns:
        2-D array with shape (len(fp), len(x)), with each row i equal to
            np.interp(x, xp, fp[i, :])
    """

    result = np.empty((fp.shape[0], len(x)), np.float32)
    for i in range(fp.shape[0]):
        result[i, :] = np.interp(x, xp, fp[i, :])
    return result


def interpolate_multiple_rows_same_x(*args):
    """For each row i of fp, do the equivalent of np.interp(x, xp, fp[i, :]).

    Calls a fast numba implementation where numba is available (see
    `improver.ensemble_copula_coupling.numba_utilities.fast_interp_same_y`) and calls a
    the native python implementation otherwise (see :func:`slow_interp_same_y`).

    Args:
        x: 1-D array
        xp: 1-D array, sorted in non-decreasing order
        fp: 2-D array with len(xp) columns
    Returns:
        2-D array with shape (len(fp), len(x)), with each row i equal to
            np.interp(x, xp, fp[i, :])
    """
    try:
        import numba  # noqa: F401

        from improver.ensemble_copula_coupling.numba_utilities import fast_interp_same_x

        return fast_interp_same_x(*args)
    except ImportError:
        warnings.warn("Module numba unavailable. ResamplePercentiles will be slower.")
        return slow_interp_same_x(*args)


def slow_interp_same_y(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """For each row i of xp, do the equivalent of np.interp(x, xp[i], fp).

    Args:
        x: 1-d array
        xp: n * m array, each row must be in non-decreasing order
        fp: 1-d array with length m
    Returns:
        n * len(x) array where each row i is equal to np.interp(x, xp[i], fp)
    """
    result = np.empty((xp.shape[0], len(x)), dtype=np.float32)
    for i in range(xp.shape[0]):
        result[i] = np.interp(x, xp[i, :], fp)
    return result


def interpolate_multiple_rows_same_y(*args):
    """For each row i of xp, do the equivalent of np.interp(x, xp[i], fp).

    Calls a fast numba implementation where numba is available (see
    `improver.ensemble_copula_coupling.numba_utilities.fast_interp_same_y`) and calls a
    the native python implementation otherwise (see :func:`slow_interp_same_y`).

    Args:
        x: 1-d array
        xp: n * m array, each row must be in non-decreasing order
        fp: 1-d array with length m
    Returns:
        n * len(x) array where each row i is equal to np.interp(x, xp[i], fp)
    """
    try:
        import numba  # noqa: F401

        from improver.ensemble_copula_coupling.numba_utilities import fast_interp_same_y

        return fast_interp_same_y(*args)
    except ImportError:
        warnings.warn(
            "Module numba unavailable. ConvertProbabilitiesToPercentiles will be slower."
        )
        return slow_interp_same_y(*args)


def slow_interpolate_pointwise(x, xp, fp):
    """
    Given 2 arrays xp and fp with the same dimensions, interpolate along first dimensions
    at values given by x.

    Args:
        x: 1-d array, points at which to evaluate interpolation function
        xp: array with at least 2 dimensions. First dimension is the interpolation dimension.
        fp: array with at least 2 dimensions. First dimension is the interpolation dimension.

    Returns:
        array with dimensions (len(x), ) + xp.shape[1:]
    """

    if xp.shape != fp.shape:
        raise ValueError("xp and fp must have the same shape")
    if len(xp.shape) < 2:
        raise ValueError("xp and fp must have at least 2 dimensions")
    result = np.empty((len(x),) + tuple(xp.shape[1:]))
    for coords in product(*[range(i) for i in xp.shape[1:]]):
        coord_slice = (slice(None),) + coords
        result[coord_slice] = np.interp(x, xp[coord_slice], fp[coord_slice])
    return result


def interpolate_pointwise(*args):
    """
    Given 2 arrays xp and fp with the same dimensions, interpolate along first dimensions
    at values given by x. Specifically, if xp and fp have dimensions d0, ..., dn,
    then the output has dimensions len(x), d1, ..., dn, and the value of coordinate
    (i_0, ..., i_n) is equal to np.interp(x, xp[:, i_1, ..., i_n], fp[:, i_1, ..., i_n]).

    Calls a fast numba implementation where numba is available (see
    `improver.ensemble_copula_coupling.numba_utilities.fast_interpolate_pointwise`) and calls
    a native python implementation otherwise (see :func:`slow_interpolate_pointwise`).

    Args:
        x: 1-d array, points at which to evaluate interpolation function
        xp: array with at least 2 dimensions. First dimension is the interpolation dimension.
        fp: array with at least 2 dimensions. First dimension is the interpolation dimension.

    Returns:
        array with dimensions (len(x), ) + xp.shape[1:]
    """
    try:
        import numba  # noqa: F401

        from improver.ensemble_copula_coupling.numba_utilities import (
            fast_interpolate_pointwise,
        )

        return fast_interpolate_pointwise(*args)
    except ImportError:
        warnings.warn("Module numba unavailable. interpolate_pointwise will be slower.")
        return slow_interpolate_pointwise(*args)
