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
"""
This module defines all the utilities used by the "plugins"
specific for ensemble calibration.

"""
import iris
import numpy as np

from improver.utilities.temporal import iris_time_to_datetime


def convert_cube_data_to_2d(
        forecast, coord="realization", transpose=True):
    """
    Function to convert data from a N-dimensional cube into a 2d
    numpy array. The result can be transposed, if required.

    Args:
        forecast (iris.cube.Cube):
            N-dimensional cube to be reshaped.
        coord (str):
            The data will be flattened along this coordinate.
        transpose (bool):
            If True, the resulting flattened data is transposed.
            This will transpose a 2d array of the format [:, coord]
            to [coord, :].
            If False, the resulting flattened data is not transposed.
            This will result in a 2d array of format [:, coord].

    Returns:
        numpy.ndarray:
            Reshaped 2d array.

    """
    forecast_data = []
    if np.ma.is_masked(forecast.data):
        forecast.data = np.ma.filled(forecast.data, np.nan)

    for coord_slice in forecast.slices_over(coord):
        forecast_data.append(coord_slice.data.flatten())
    if transpose:
        forecast_data = np.asarray(forecast_data).T
    return np.array(forecast_data)


def flatten_ignoring_masked_data(data_array, preserve_leading_dimension=False):
    """
    Flatten an array, selecting only valid data if the array is masked. There
    is also the option to reshape the resulting array so it has the same
    leading dimension as the input array, but the other dimensions of the
    array are flattened. It is assumed that each of the slices
    along the leading dimension are masked in the same way. This functionality
    is used in EstimateCoefficientsForEnsembleCalibration when realizations
    are used as predictors.

    Args:
        data_array (numpy.ndarray or numpy.ma.MaskedArray):
            An array or masked array to be flattened. If it is masked and the
            leading dimension is preserved the mask must be the same for every
            slice along the leading dimension.
        preserve_leading_dimension (bool):
            Default False.
            If True the flattened array is reshaped so it has the same leading
            dimension as the input array. If False the returned array is 1D.
    Returns:
        numpy.ndarray:
            A flattened array containing only valid data. Either 1D or, if
            preserving the leading dimension 2D. In the latter case the
            leading dimension is the same as the input data_array.
    Raises:
        ValueError: If preserving the leading dimension and the mask on the
                    input array is not the same for every slice along the
                    leading dimension.
    """
    if np.ma.is_masked(data_array):
        # If we have multiple 2D x-y slices check that the mask is the same for
        # each slice along the leading dimension.
        if data_array.ndim > 2:
            first_slice_mask = data_array[0].mask

            for i in range(1, data_array.shape[0]):
                if not np.all(first_slice_mask == data_array[i].mask):
                    msg = ("The mask on the input array is not the same for "
                           "every slice along the leading dimension.")
                    raise ValueError(msg)
        # If the mask is ok, select the unmasked data, flattening it at
        # the same time.
        result = data_array[~data_array.mask]
    else:
        result = data_array.flatten()
    if preserve_leading_dimension:
        # Reshape back to give the same leading dimension in the array. The 2nd
        # dimension is inferred through the use of -1.
        final_shape = (data_array.shape[0], -1)
        result = result.reshape(final_shape)
    return result


def check_predictor(predictor):
    """
    Check the predictor at the start of the process methods in relevant
    ensemble calibration plugins, to avoid having to check and raise an error
    later.

    Args:
        predictor (str):
            String to specify the form of the predictor used to calculate
            the location parameter when estimating the EMOS coefficients.
            Currently the ensemble mean ("mean") and the ensemble
            realizations ("realizations") are supported as the predictors.

    Raises:
        ValueError: If the predictor is not valid.
    """
    if predictor.lower() not in ["mean", "realizations"]:
        msg = ("The requested value for the predictor {} is not an accepted "
               "value. Accepted values are 'mean' or 'realizations'").format(
                   predictor.lower())
        raise ValueError(msg)


def filter_non_matching_cubes(historic_forecast, truth):
    """
    Provide filtering for the historic forecast and truth to make sure
    that these contain matching validity times. This ensures that any
    mismatch between the historic forecasts and truth is dealt with.

    Args:
        historic_forecast (iris.cube.Cube):
            Cube of historic forecasts that potentially contains
            a mismatch compared to the truth.
        truth (iris.cube.Cube):
            Cube of truth that potentially contains a mismatch
            compared to the historic forecasts.

    Returns:
        (tuple): tuple containing:
            **matching_historic_forecasts** (iris.cube.Cube):
                Cube of historic forecasts where any mismatches with
                the truth cube have been removed.
            **matching_truths** (iris.cube.Cube):
                Cube of truths where any mismatches with
                the historic_forecasts cube have been removed.

    Raises:
        ValueError: The filtering has found no matches in validity time
            between the historic forecasts and the truths.

    """
    matching_historic_forecasts = iris.cube.CubeList([])
    matching_truths = iris.cube.CubeList([])
    for hf_slice in historic_forecast.slices_over("time"):
        if hf_slice.coord("time").has_bounds():
            point = iris_time_to_datetime(hf_slice.coord("time"),
                                          point_or_bound="point")
            bounds, = iris_time_to_datetime(
                hf_slice.coord("time"), point_or_bound="bound")
            coord_values = (
                {"time": lambda cell: point[0] == cell.point and
                    bounds[0] == cell.bound[0] and
                    bounds[1] == cell.bound[1]})
        else:
            coord_values = (
                {"time": iris_time_to_datetime(
                    hf_slice.coord("time"), point_or_bound="point")})

        constr = iris.Constraint(coord_values=coord_values)
        truth_slice = truth.extract(constr)

        if truth_slice:
            matching_historic_forecasts.append(hf_slice)
            matching_truths.append(truth_slice)
    if not matching_historic_forecasts and not matching_truths:
        msg = ("The filtering has found no matches in validity time "
               "between the historic forecasts and the truths.")
        raise ValueError(msg)
    return (matching_historic_forecasts.merge_cube(),
            matching_truths.merge_cube())


def create_unified_frt_coord(forecast_reference_time):
    """
    Constructs a single forecast reference time coordinate from a multi-valued
    coordinate. The new coordinate records the maximum range of bounds of
    the input forecast reference times, with the point value set to the latest
    of those in the inputs.

    Args:
        forecast_reference_time (iris.coord.DimCoord):
            The forecast_reference_time coordinate to be used in the
            coordinate creation.
    Returns:
        iris.coord.DimCoord:
            A dimension coordinate containing the forecast reference time
            coordinate with suitable bounds. The coordinate point is that
            of the latest contributing forecast.
    """
    frt_point = forecast_reference_time.points.max()
    frt_bounds_min = forecast_reference_time.points.min()
    frt_bounds_max = frt_point
    if forecast_reference_time.has_bounds():
        frt_bounds_min = min(frt_bounds_min,
                             forecast_reference_time.bounds.min())
        frt_bounds_max = max(frt_bounds_max,
                             forecast_reference_time.bounds.max())
    frt_bounds = (frt_bounds_min, frt_bounds_max)
    return forecast_reference_time[0].copy(points=frt_point,
                                           bounds=frt_bounds)


def merge_land_and_sea(calibrated_land_only, uncalibrated):
    """
    Merge data that has been calibrated over the land with uncalibrated data.
    Calibrated data will have masked data over the sea which will need to be
    filled with the uncalibrated data.

    Args:
        calibrated_land_only (iris.cube.Cube):
            A cube that has been calibrated over the land, with sea points
            masked out. Either realizations, probabilities or percentiles.
            Data is modified in place.
        uncalibrated (iris.cube.Cube):
            A cube of uncalibrated data with valid data over the sea. Either
            realizations, probabilities or percentiles. Dimension coordinates
            must be the same as the calibrated_land_only cube.

    Raises:
        ValueError: If input cubes do not have the same input dimensions.
    """
    # Check dimensions the same on both cubes.
    if calibrated_land_only.dim_coords != uncalibrated.dim_coords:
        message = "Input cubes do not have the same dimension coordinates"
        raise ValueError(message)
    # Merge data if calibrated_land_only data is masked.
    if np.ma.is_masked(calibrated_land_only.data):
        new_data = calibrated_land_only.data.data
        mask = calibrated_land_only.data.mask
        new_data[mask] = uncalibrated.data[mask]
        calibrated_land_only.data = new_data
