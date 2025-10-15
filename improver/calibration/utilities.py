# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
This module defines all the utilities used by the "plugins"
specific for ensemble calibration.

"""

import warnings
from pathlib import Path
from typing import List, Set, Tuple, Union

import iris
import numpy as np
import pandas as pd
from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from numpy import ndarray
from numpy.ma.core import MaskedArray

from improver.utilities.cube_manipulation import enforce_coordinate_ordering
from improver.utilities.temporal import iris_time_to_datetime


def convert_cube_data_to_2d(
    forecast: Cube, coord: str = "realization", transpose: bool = True
) -> ndarray:
    """
    Function to convert data from a N-dimensional cube into a 2d
    numpy array. The result can be transposed, if required.

    Args:
        forecast:
            N-dimensional cube to be reshaped.
        coord:
            This dimension is retained as the second dimension by default,
            and the leading dimension if "transpose" is set to False.
        transpose:
            If True, the resulting flattened data is transposed.
            This will transpose a 2d array of the format [coord, :]
            to [:, coord].  If coord is not a dimension on the input cube,
            the resulting array will be 2d with items of length 1.

    Returns:
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


def flatten_ignoring_masked_data(
    data_array: Union[MaskedArray, ndarray], preserve_leading_dimension: bool = False
) -> ndarray:
    """
    Flatten an array, selecting only valid data if the array is masked. There
    is also the option to reshape the resulting array so it has the same
    leading dimension as the input array, but the other dimensions of the
    array are flattened. It is assumed that each of the slices
    along the leading dimension are masked in the same way. This functionality
    is used in EstimateCoefficientsForEnsembleCalibration when realizations
    are used as predictors.

    Args:
        data_array:
            An array or masked array to be flattened. If it is masked and the
            leading dimension is preserved the mask must be the same for every
            slice along the leading dimension.
        preserve_leading_dimension:
            Default False.
            If True the flattened array is reshaped so it has the same leading
            dimension as the input array. If False the returned array is 1D.

    Returns:
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
                    msg = (
                        "The mask on the input array is not the same for "
                        "every slice along the leading dimension."
                    )
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


def check_predictor(predictor: str) -> str:
    """
    Check the predictor at the start of the process methods in relevant
    ensemble calibration plugins, to avoid having to check and raise an error
    later. Also, lowercase the string.

    Args:
        predictor:
            String to specify the form of the predictor used to calculate
            the location parameter when estimating the EMOS coefficients.
            Currently the ensemble mean ("mean") and the ensemble
            realizations ("realizations") are supported as the predictors.

    Returns:
        The predictor string in lowercase.

    Raises:
        ValueError: If the predictor is not valid.
    """
    if predictor.lower() not in ["mean", "realizations"]:
        msg = (
            "The requested value for the predictor {} is not an accepted "
            "value. Accepted values are 'mean' or 'realizations'"
        ).format(predictor.lower())
        raise ValueError(msg)
    return predictor.lower()


def filter_non_matching_cubes(
    historic_forecast: Cube, truth: Cube
) -> Tuple[Cube, Cube]:
    """
    Provide filtering for the historic forecast and truth to make sure
    that these contain matching validity times. This ensures that any
    mismatch between the historic forecasts and truth is dealt with.
    If multiple time slices of the historic forecast match with the
    same truth slice, only the first truth slice is kept to avoid
    duplicate truth slices, which prevent the truth cubes being merged.
    This can occur when processing a cube with a multi-dimensional time
    coordinate. If a historic forecast time slice contains only NaNs,
    then this time slice is also skipped. This can occur when processing
    a multi-dimensional time coordinate where some of the forecast reference
    time and forecast period combinations do not typically occur, so may
    be filled with NaNs.

    Args:
        historic_forecast:
            Cube of historic forecasts that potentially contains
            a mismatch compared to the truth.
        truth:
            Cube of truth that potentially contains a mismatch
            compared to the historic forecasts.

    Returns:
        - Cube of historic forecasts where any mismatches with
          the truth cube have been removed.
        - Cube of truths where any mismatches with
          the historic_forecasts cube have been removed.

    Raises:
        ValueError: The filtering has found no matches in validity time
            between the historic forecasts and the truths.
    """
    matching_historic_forecasts = iris.cube.CubeList([])
    matching_truths = iris.cube.CubeList([])
    truth_times = []
    for hf_slice in historic_forecast.slices_over("time"):
        if hf_slice.coord("time").has_bounds():
            point = iris_time_to_datetime(
                hf_slice.coord("time"), point_or_bound="point"
            )
            (bounds,) = iris_time_to_datetime(
                hf_slice.coord("time"), point_or_bound="bound"
            )
            coord_values = {
                "time": lambda cell: point[0] == cell.point
                and bounds[0] == cell.bound[0]
                and bounds[1] == cell.bound[1]
            }
        else:
            coord_values = {
                "time": iris_time_to_datetime(
                    hf_slice.coord("time"), point_or_bound="point"
                )
            }

        constr = iris.Constraint(coord_values=coord_values)
        truth_slice = truth.extract(constr)

        if (
            truth_slice
            and not np.isnan(hf_slice.data).all()
            and truth_slice.coord("time").cell(0) not in truth_times
        ):
            truth_times.append(truth_slice.coord("time").cell(0))
            matching_historic_forecasts.append(hf_slice)
            matching_truths.append(truth_slice)
    if not matching_historic_forecasts and not matching_truths:
        msg = (
            "The filtering has found no matches in validity time "
            "between the historic forecasts and the truths."
        )
        raise ValueError(msg)

    hf_coord_names = [c.name() for c in historic_forecast.coords(dim_coords=True)]
    truth_coord_names = [c.name() for c in truth.coords(dim_coords=True)]
    hf_cube = matching_historic_forecasts.merge_cube()
    truth_cube = matching_truths.merge_cube()
    enforce_coordinate_ordering(hf_cube, hf_coord_names)
    enforce_coordinate_ordering(truth_cube, truth_coord_names)
    return (hf_cube, truth_cube)


def create_unified_frt_coord(forecast_reference_time: DimCoord) -> DimCoord:
    """
    Constructs a single forecast reference time coordinate from a multi-valued
    coordinate. The new coordinate records the maximum range of bounds of
    the input forecast reference times, with the point value set to the latest
    of those in the inputs.

    Args:
        forecast_reference_time:
            The forecast_reference_time coordinate to be used in the
            coordinate creation.

    Returns:
        A dimension coordinate containing the forecast reference time
        coordinate with suitable bounds. The coordinate point is that
        of the latest contributing forecast.
    """
    frt_point = forecast_reference_time.points.max()
    frt_bounds_min = forecast_reference_time.points.min()
    frt_bounds_max = frt_point
    if forecast_reference_time.has_bounds():
        frt_bounds_min = min(frt_bounds_min, forecast_reference_time.bounds.min())
        frt_bounds_max = max(frt_bounds_max, forecast_reference_time.bounds.max())
    frt_bounds = (frt_bounds_min, frt_bounds_max)
    return forecast_reference_time[0].copy(points=frt_point, bounds=frt_bounds)


def merge_land_and_sea(calibrated_land_only: Cube, uncalibrated: Cube) -> None:
    """
    Merge data that has been calibrated over the land with uncalibrated data.
    Calibrated data will have masked data over the sea which will need to be
    filled with the uncalibrated data.

    Args:
        calibrated_land_only:
            A cube that has been calibrated over the land, with sea points
            masked out. Either realizations, probabilities or percentiles.
            Data is modified in place.
        uncalibrated:
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


def _ceiling_fp(cube: Cube) -> np.ndarray:
    """Find the forecast period points rounded up to the next hour.

    Args:
        cube:
            Cube with a forecast_period coordinate.

    Returns:
        The forecast period points in units of hours after
        rounding the points up to the next hour.
    """
    coord = cube.coord("forecast_period").copy()
    coord.convert_units("hours")
    return np.ceil(coord.points)


def forecast_coords_match(first_cube: Cube, second_cube: Cube) -> None:
    """
    Determine if two cubes have equivalent forecast_periods and
    forecast_reference_time coordinates with an accepted leniency.
    The forecast period is rounded up to the next hour to
    support calibrating subhourly forecasts with coefficients taken from on
    the hour. For forecast reference time, only the hour is checked.

    Args:
        first_cube:
            First cube to compare.
        second_cube:
            Second cube to compare.

    Raises:
        ValueError: The two cubes are not equivalent.
    """
    mismatches = []
    if _ceiling_fp(first_cube) != _ceiling_fp(second_cube):
        mismatches.append("rounded forecast_period hours")

    if get_frt_hours(first_cube.coord("forecast_reference_time")) != get_frt_hours(
        second_cube.coord("forecast_reference_time")
    ):
        mismatches.append("forecast_reference_time hours")
    if mismatches:
        msg = "The following coordinates of the two cubes do not match: {}"
        raise ValueError(msg.format(", ".join(mismatches)))


def get_frt_hours(forecast_reference_time: DimCoord) -> Set[int]:
    """
    Returns a set of integer representations of the hour of the
    forecast reference time.

    Args:
        forecast_reference_time:
            The forecast_reference_time coordinate to extract the hours from.

    Returns:
        A set of integer representations of the forecast reference time
        hours.
    """
    frt_hours = []
    for frt in forecast_reference_time.cells():
        frt_hours.append(np.int32(frt.point.hour))
    return set(frt_hours)


def check_forecast_consistency(forecasts: Cube) -> None:
    """
    Checks that the forecast cubes have a consistent forecast reference time
    hour and a consistent forecast period.

    Args:
        forecasts:

    Raises:
        ValueError: Forecast cubes have differing forecast reference time hours
        ValueError: Forecast cubes have differing forecast periods
    """
    frt_hours = get_frt_hours(forecasts.coord("forecast_reference_time"))

    if len(frt_hours) != 1:
        frt_hours = set(map(int, frt_hours))
        msg = (
            "Forecasts have been provided with differing hours for the "
            "forecast reference time {}"
        )
        raise ValueError(msg.format(frt_hours))
    if len(forecasts.coord("forecast_period").points) != 1:
        msg = "Forecasts have been provided with differing forecast periods {}"
        raise ValueError(msg.format(forecasts.coord("forecast_period").points))


def broadcast_data_to_time_coord(cubelist: CubeList) -> List[ndarray]:
    """Ensure that the data from all cubes within a cubelist is of the required shape
    by broadcasting the data from cubes without a time coordinate along the time
    dimension taken from other input cubes that do have a time coordinate. In the
    case where none of the input cubes have a time coordinate that is a dimension
    coordinate, which may occur when using a very small training dataset, the
    data is returned without being broadcast.

    Args:
        cubelist:
            The cubelist from which the data will be extracted and broadcast along
            the time dimension as required.

    Returns:
       The data taken from cubes within a cubelist where cubes without a
       time coordinate have had their data broadcast along the time dimension
       (with this time dimension provided by other input cubes with a time
       dimension) to ensure that the data within each numpy array within the
       output list has the same shape. If a time dimension coordinate is not
       present on any of the cubes, no broadcasting occurs.
    """
    broadcasted_data = []
    num_times = [
        len(cube.coord("time").points)
        for cube in cubelist
        if cube.coords("time", dim_coords=True)
    ]
    for cube in cubelist:
        data = cube.data
        if not cube.coords("time") and num_times:
            # Broadcast data from cube along a time dimension.
            data = np.broadcast_to(data, (num_times[0],) + data.shape)

        broadcasted_data.append(data)
    return broadcasted_data


def check_data_sufficiency(
    historic_forecasts: Cube,
    truths: Cube,
    point_by_point: bool,
    proportion_of_nans: float,
):
    """Check whether there is sufficient valid data (i.e. values that are not NaN)
    within the historic forecasts and truths, in order to robustly compute EMOS
    coefficients.

    Args:
        historic_forecasts:
            Cube containing historic forcasts.
        truths:
            Cube containing truths.
        point_by_point:
            If True, coefficients are calculated independently for each
            point within the input cube by creating an initial guess and
            minimising each grid point independently.
        proportion_of_nans:
            The proportion of the matching historic forecast-truth pairs that
            are allowed to be NaN.

    Raises:
        ValueError: If the proportion of NaNs is higher than allowable for a site,
            if using point_by_point.
        ValueError: If the proportion of NaNs is higher than allowable when
            considering all sites.
    """
    if not historic_forecasts.coords("wmo_id"):
        return

    truths_data = np.broadcast_to(truths.data, historic_forecasts.shape)
    index = np.isnan(historic_forecasts.data) & np.isnan(truths_data)

    if point_by_point:
        wmo_id_axis = historic_forecasts.coord_dims("wmo_id")[0]
        non_wmo_id_axes = list(range(len(historic_forecasts.shape)))
        non_wmo_id_axes.pop(wmo_id_axis)
        detected_proportion = np.count_nonzero(
            index, axis=tuple(non_wmo_id_axes)
        ) / np.prod(np.array(index.shape)[non_wmo_id_axes])
        if np.any(detected_proportion > proportion_of_nans):
            number_of_sites = np.sum(detected_proportion > proportion_of_nans)
            msg = (
                f"{number_of_sites} sites have a proportion of NaNs that is "
                f"higher than the allowable proportion of NaNs within the "
                "historic forecasts and truth pairs. The allowable proportion is "
                f"{proportion_of_nans}. The maximum proportion of NaNs is "
                f"{np.amax(detected_proportion)}."
            )
            raise ValueError(msg)
    else:
        detected_proportion = np.count_nonzero(index) / index.size
        if detected_proportion > proportion_of_nans:
            msg = (
                f"The proportion of NaNs detected is {detected_proportion}. "
                f"This is higher than the allowable proportion of NaNs within the "
                f"historic forecasts and truth pairs: {proportion_of_nans}."
            )
            raise ValueError(msg)


def prepare_cube_no_calibration(
    forecast: Cube,
    emos_coefficients: CubeList,
    ignore_ecc_bounds_exceedance: bool = False,
    validity_times: List[str] = None,
    percentiles: List[float] = None,
    prob_template: Cube = None,
) -> Cube | None:
    """
    Function to add appropriate metadata to cubes that cannot be calibrated. If the
    forecast can be calibrated then nothing is returned.

    Args:
        forecast (iris.cube.Cube):
            The forecast to be calibrated. The input format could be either
            realizations, probabilities or percentiles.
        emos_coefficients (iris.cube.CubeList):
            The EMOS coefficients to be applied to the forecast.
        ignore_ecc_bounds_exceedance (bool):
            If True, where the percentiles exceed the ECC bounds range,
            raises a warning rather than an exception. This occurs when the
            current forecast is in the form of probabilities and is
            converted to percentiles, as part of converting the input
            probabilities into realizations.
        validity_times (List[str]):
            Times at which the forecast must be valid. This must be provided
            as a four digit string (HHMM) where the first two digits represent the hour
            and the last two digits represent the minutes e.g. 0300 or 0315. If the
            forecast provided is at a different validity time then no coefficients
            will be applied.
        percentiles (List[float]):
            The set of percentiles used to create the calibrated forecast.
        prob_template (iris.cube.Cube):
            Optionally, a cube containing a probability forecast that will be
            used as a template when generating probability output when the input
            format of the forecast cube is not probabilities i.e. realizations
            or percentiles. If no coefficients are provided and a probability
            template is provided, the probability template forecast will be
            returned as the uncalibrated probability forecast.

    Returns:
        The prepared forecast cube or None.
    """
    from improver.calibration import add_warning_comment, validity_time_check
    from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
        ResamplePercentiles,
    )

    if validity_times is not None and not validity_time_check(forecast, validity_times):
        if percentiles:
            # Ensure that a consistent set of percentiles are returned,
            # regardless of whether SAMOS is successfully applied.
            percentiles = [np.float32(p) for p in percentiles]
            forecast = ResamplePercentiles(
                ecc_bounds_warning=ignore_ecc_bounds_exceedance
            )(forecast, percentiles=percentiles)
        elif prob_template:
            forecast = prob_template
        forecast = add_warning_comment(forecast)
        return forecast

    if emos_coefficients is None:
        if prob_template:
            msg = (
                "There are no coefficients provided for calibration. As a "
                "probability template has been provided with the aim of "
                "creating a calibrated probability forecast, the probability "
                "template will be returned as the uncalibrated probability "
                "forecast."
            )
            warnings.warn(msg)
            prob_template = add_warning_comment(prob_template)
            return prob_template

        if percentiles:
            # Ensure that a consistent set of percentiles are returned,
            # regardless of whether SAMOS is successfully applied.
            percentiles = [np.float32(p) for p in percentiles]
            forecast = ResamplePercentiles(
                ecc_bounds_warning=ignore_ecc_bounds_exceedance
            )(forecast, percentiles=percentiles)

        msg = (
            "There are no coefficients provided for calibration. The "
            "uncalibrated forecast will be returned."
        )
        warnings.warn(msg)

        forecast = add_warning_comment(forecast)
        return forecast


def convert_parquet_to_cube(
    forecast: Path,
    truth: Path,
    forecast_period: int,
    cycletime: str,
    training_length: int,
    diagnostic: str,
    percentiles: List[float],
    experiment: str,
) -> iris.cube.CubeList:
    """Function to convert a parquet file containing forecast and truth data
    into a CubeList for use in calibration.

    Args:
        forecast (pathlib.Path):
            The path to a Parquet file containing the historical forecasts
            to be used for calibration. The expected columns within the
            Parquet file are: forecast, blend_time, forecast_period,
            forecast_reference_time, time, wmo_id, percentile, diagnostic,
            latitude, longitude, period, height, cf_name, units.
        truth (pathlib.Path):
            The path to a Parquet file containing the truths to be used
            for calibration. The expected columns within the
            Parquet file are: ob_value, time, wmo_id, diagnostic, latitude,
            longitude and altitude.
        forecast_period (int):
            Forecast period to be calibrated in seconds.
        cycletime (str):
            Cycletime of a format similar to 20170109T0000Z.
        training_length (int):
            Number of days within the training period.
        diagnostic (str):
            The name of the diagnostic to be calibrated within the forecast
            and truth tables. This name is used to filter the Parquet file
            when reading from disk.
        percentiles (List[float]):
            The set of percentiles to be used for estimating coefficients.
            These should be a set of equally spaced quantiles.
        experiment (str):
            A value within the experiment column to select from the forecast
            table.

    Returns:
        A CubeList containing the forecast and truth cubes, with the
        forecast cube containing the percentiles as an auxiliary coordinate.
    """
    from improver.calibration.dataframe_utilities import (
        forecast_and_truth_dataframes_to_cubes,
    )

    # Load forecasts from parquet file filtering by diagnostic and blend_time.
    forecast_period_td = pd.Timedelta(int(forecast_period), unit="seconds")

    cycletimes = pd.date_range(
        end=pd.Timestamp(cycletime)
        - pd.Timedelta(1, unit="days")
        - forecast_period_td.floor("D"),
        periods=int(training_length),
        freq="D",
    )
    filters = [[("diagnostic", "==", diagnostic), ("blend_time", "in", cycletimes)]]
    forecast_df = pd.read_parquet(forecast, filters=filters)

    # Load truths from parquet file filtering by diagnostic.
    filters = [[("diagnostic", "==", diagnostic)]]
    truth_df = pd.read_parquet(truth, filters=filters)
    if truth_df.empty:
        msg = (
            f"The requested filepath {truth} does not contain the "
            f"requested contents: {filters}"
        )
        raise IOError(msg)

    forecast_cube, truth_cube = forecast_and_truth_dataframes_to_cubes(
        forecast_df,
        truth_df,
        cycletime,
        forecast_period,
        training_length,
        percentiles=percentiles,
        experiment=experiment,
    )
    if not forecast_cube or not truth_cube:
        return [None, None]
    else:
        return CubeList([forecast_cube, truth_cube])
