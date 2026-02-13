# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing lapse rate calculation plugins."""

from typing import Iterable, Optional, Tuple

import iris
import numpy as np
import scipy.stats
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError
from numpy import ndarray
from scipy.ndimage import maximum_filter, minimum_filter

from improver import BasePlugin, PostProcessingPlugin
from improver.constants import DALR, ELR
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.nbhood.nbhood import circular_kernel
from improver.utilities import mathematical_operations, neighbourhood_tools
from improver.utilities.cube_checker import spatial_coords_match
from improver.utilities.cube_manipulation import (
    enforce_coordinate_ordering,
    get_dim_coord_names,
)


def compute_lapse_rate_adjustment(
    lapse_rate: np.ndarray, orog_diff: np.ndarray, max_orog_diff_limit: float = 50
) -> np.ndarray:
    """Compute the lapse rate adjustment i.e. the lapse rate multiplied by the
    relevant orographic difference. The lapse rate is assumed to be appropriate for a
    fixed vertical displacement between the source and destination orographies.
    If the the vertical displacement is greater than the limit specified,
    further vertical ascent or descent is assumed to follow the environmental
    lapse rate (also known as standard atmosphere lapse rate). Note that this is an
    extension of Sheridan et al., 2018, which applies this vertical displacement limit
    for positive lapse rates only.

    For the specific case of a deep unresolved valley with a positive lapse rate at
    the altitude of the source orography, the atmosphere can be imagined to be
    compartmentalised into two regions. Firstly, a cold pool section that extends below
    the altitude of the source orography by the value given by the max orography
    difference specified (70 m in Sheridan et al., 2018) and secondly, a standard
    atmosphere section for the remaining orographic difference. The total lapse rate
    adjustment is the sum of the contributions from these two vertical sections.

    References:
        Sheridan, P., S. Vosper, and S. Smith, 2018: A Physically Based Algorithm for
        Downscaling Temperature in Complex Terrain. J. Appl. Meteor. Climatol.,
        57, 1907–1929, https://doi.org/10.1175/JAMC-D-17-0140.1.

    Args:
        lapse_rate: Array containing lapse rate in units of K/m.
        orog_diff: Array containing the difference in orography
            (destination orography minus source orography) in metres.
        max_orog_diff_limit: Maximum vertical displacement in metres to be corrected
            using the lapse rate provided. Vertical displacement in excess of this
            value will be corrected using the environmental lapse rate (also known
            as standard atmosphere lapse rate). This defaults to 50.
            Sheridan et al. use 70 m. As lapse rate adjustment could be performed
            both for gridded data and for site data in sequence, the adjustments
            could accumulate. To limit the possible cumulative effect from multiple
            lapse rate corrections, a default lower than 70m has been chosen.

    Returns:
        The vertical lapse rate adjustment to be applied to correct a
        diagnostic forecast in SI units.
    """
    orog_diff = np.broadcast_to(orog_diff, lapse_rate.shape).copy()
    orig_orog_diff = orog_diff.copy()

    # Constraint if the orographic difference is either greater than the max allowed
    # orographic difference (e.g. an unresolved hilltop) or less than the negative of
    # the max allowed orographic difference (e.g. an unresolved valley).
    condition1 = orog_diff > max_orog_diff_limit
    condition2 = orog_diff < -max_orog_diff_limit
    orog_diff[condition1] = max_orog_diff_limit
    orog_diff[condition2] = -max_orog_diff_limit
    vertical_adjustment = np.multiply(orog_diff, lapse_rate)

    # Compute an additional lapse rate adjustment for points with an absolute
    # orographic difference greater than the maximum allowed.
    orig_orog_diff[condition1] = np.clip(
        orig_orog_diff[condition1] - max_orog_diff_limit, 0, None
    )
    orig_orog_diff[condition2] = np.clip(
        orig_orog_diff[condition2] + max_orog_diff_limit, None, 0
    )

    # Assume the Environmental Lapse Rate (also known as Standard Atmosphere
    # Lapse Rate).
    vertical_adjustment[condition1] += np.multiply(orig_orog_diff[condition1], ELR)
    vertical_adjustment[condition2] += np.multiply(orig_orog_diff[condition2], ELR)
    return vertical_adjustment


def compute_from_slope_and_intercept(
    position: np.ndarray, slope: np.ndarray, intercept: np.ndarray
) -> np.ndarray:
    """
    Solves y for a linear equation y = mx + c

    Args:
        position:
            Array of values corresponding to x in the linear equation
        slope:
            Array of values corresponding to m in the linear equation
        intercept:
            Array of values corresponding to c in the linear equation
    """
    return position * slope + intercept


class ApplyGriddedLapseRate(PostProcessingPlugin):
    """Class to apply a lapse rate adjustment to a forecast diagnostic"""

    def __init__(
        self,
        data_limits: Iterable[float] = (None, None),
        data_limits_from_nbhood: int = None,
        max_from_nbhood_uplift: float = 1.1,
        min_from_nbhood_uplift: float = 1.0,
    ):
        """
        Initialise the class

        Args:
            data_limits:
                (Minimum value, Maximum value). If one or both are not None, these values are used to truncate
                the data after all calculations are complete.
            data_limits_from_nbhood:
                If not None, this value is used to calculate data limits from the neighbourhood of each point.
                The value is the number of grid points in a radius in grid-lengths from which to sample the local minima and maxima.
            max_from_nbhood_uplift:
                If data_limits_from_nbhood is not None, the local maximum is multiplied by this value
                to give the upper limit for the data. This allows the local maximum to be exceeded by
                a small amount, which may be desirable for modelling unresolved hills and valleys.
            min_from_nbhood_uplift:
                If data_limits_from_nbhood is not None, the local minimum is multiplied by this value
                to give the lower limit for the data. This allows the local minimum to be exceeded by
                a small amount, which may be desirable for modelling unresolved hills and valleys.
                This multiplier must be smaller than or equal to the max_from_nbhood_uplift multiplier,
                otherwise the limits could be inverted.
        """
        if data_limits_from_nbhood is None:
            self.data_limits_from_nbhood = None
            self.local_min, self.local_max = data_limits
            if not np.isclose(max_from_nbhood_uplift, 1.1) or not np.isclose(
                min_from_nbhood_uplift, 1.0
            ):
                raise ValueError(
                    "max_from_nbhood_uplift and min_from_nbhood_uplift should not be set if data_limits_from_nbhood is None"
                )
        else:
            self.data_limits_from_nbhood = data_limits_from_nbhood
            self.local_min, self.local_max = (None, None)
            self.max_from_nbhood_uplift = max_from_nbhood_uplift
            self.min_from_nbhood_uplift = min_from_nbhood_uplift
            if not data_limits_from_nbhood >= 1:
                raise ValueError(
                    f"Neighbourhood radius must be at least 1 to ensure that the central point is not the only point in the neighbourhood. Got {data_limits_from_nbhood}."
                )
            if self.max_from_nbhood_uplift <= 0.0 or self.min_from_nbhood_uplift <= 0.0:
                raise ValueError(
                    f"max_from_nbhood_uplift and min_from_nbhood_uplift should be greater than 0. Got max_from_nbhood_uplift = {max_from_nbhood_uplift}, min_from_nbhood_uplift = {min_from_nbhood_uplift}."
                )
            if self.max_from_nbhood_uplift < self.min_from_nbhood_uplift:
                raise ValueError(
                    f"max_from_nbhood_uplift should be greater than or equal to min_from_nbhood_uplift. Got max_from_nbhood_uplift = {max_from_nbhood_uplift}, min_from_nbhood_uplift = {min_from_nbhood_uplift}."
                )

    @staticmethod
    def _check_dim_coords(diagnostic: Cube, lapse_rate: Cube) -> None:
        """Throw an error if the dimension coordinates are not the same for
        diagnostic and lapse rate cubes

        Args:
            diagnostic
            lapse_rate
        """
        for crd in diagnostic.coords(dim_coords=True):
            try:
                if crd != lapse_rate.coord(crd.name()):
                    raise ValueError(
                        'Lapse rate cube coordinate "{}" does not match '
                        "diagnostic cube coordinate".format(crd.name())
                    )
            except CoordinateNotFoundError:
                raise ValueError(
                    "Lapse rate cube has no coordinate " '"{}"'.format(crd.name())
                )

    def _calc_orog_diff(self, source_orog: Cube, dest_orog: Cube) -> Cube:
        """Get difference in orography heights, in metres

        Args:
            source_orog:
                2D cube of source orography heights (units modified in place)
            dest_orog:
                2D cube of destination orography heights (units modified in
                place)

        Returns:
            The difference cube
        """
        source_orog.convert_units("m")
        dest_orog.convert_units("m")
        orog_diff = next(dest_orog.slices(self.xy_coords)) - next(
            source_orog.slices(self.xy_coords)
        )
        return orog_diff

    def _calc_local_limits(self, cube: Cube):
        """Uses data_limits_from_nbhood to calculate the local minima and maxima for each point in the cube, and sets these as the data limits."""

        # Make a local maxima and minima for capping
        kernel = circular_kernel(self.data_limits_from_nbhood, weighted_mode=False)
        self.local_max = (
            maximum_filter(cube.data, footprint=kernel) * self.max_from_nbhood_uplift
        )
        self.local_min = (
            minimum_filter(cube.data, footprint=kernel) * self.min_from_nbhood_uplift
        )

    def _apply_limits(self, cube: Cube):
        """Apply defined limits to the data in the cube"""
        # If there are no data limits, return
        if self.local_min is None and self.local_max is None:
            return
        cube.data = np.clip(cube.data, self.local_min, self.local_max)

    def process(
        self,
        diagnostic: Cube,
        lapse_rate: Cube,
        source_orog: Cube,
        dest_orog: Cube,
        intercept: Cube = None,
    ) -> Cube:
        """Applies lapse rate correction to diagnostic forecast.  All cubes'
        units are modified in place.

        Args:
            diagnostic:
                Input diagnostic field to be adjusted
            lapse_rate:
                Cube of pre-calculated lapse rates
            source_orog:
                2D cube of source orography heights
            dest_orog:
                2D cube of destination orography heights
            intercept:
                Cube of pre-calculated zero-intercept.
                If provided, the data in diagnostic are ignored.

        Returns:
            Lapse-rate adjusted diagnostic field
        """
        lapse_rate.convert_units(f"{diagnostic.units} m-1")
        self.xy_coords = [lapse_rate.coord(axis="y"), lapse_rate.coord(axis="x")]

        self._check_dim_coords(diagnostic, lapse_rate)

        if not spatial_coords_match([diagnostic, source_orog]):
            raise ValueError(
                "Source orography spatial coordinates do not match diagnostic grid"
            )

        if not spatial_coords_match([diagnostic, dest_orog]):
            raise ValueError(
                "Destination orography spatial coordinates do not match "
                "diagnostic grid"
            )

        orog_diff = self._calc_orog_diff(source_orog, dest_orog)

        if self.data_limits_from_nbhood:
            self._calc_local_limits(diagnostic)

        adjusted_diagnostic = []
        if intercept:
            for lr_slice, diagnostic_slice, intercept_slice in zip(
                lapse_rate.slices(self.xy_coords),
                diagnostic.slices(self.xy_coords),
                intercept,
            ):
                newcube = diagnostic_slice.copy()
                newcube.data += compute_from_slope_and_intercept(
                    dest_orog.data, lr_slice.data, intercept_slice.data
                )
                adjusted_diagnostic.append(newcube)
        else:
            for lr_slice, diagnostic_slice in zip(
                lapse_rate.slices(self.xy_coords), diagnostic.slices(self.xy_coords)
            ):
                newcube = diagnostic_slice.copy()
                newcube.data += compute_lapse_rate_adjustment(
                    lr_slice.data, orog_diff.data
                )
                adjusted_diagnostic.append(newcube)

        merged_cube = iris.cube.CubeList(adjusted_diagnostic).merge_cube()
        self._apply_limits(merged_cube)
        return merged_cube


class LapseRate(BasePlugin):
    """
    Plugin to calculate the lapse rate from orography and diagnostic
    cubes.

    References:
        The method applied here is based on the method used in the 2010 paper:
        https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/met.177

    Code methodology:

    1) Apply land/sea mask to diagnostic and orography datasets. Mask sea
       points as NaN since image processing module does not recognise Numpy
       masks.
    2) Creates "views" of both datasets, where each view represents a
       neighbourhood of points. To do this, each array is padded with
       NaN values to a width of half the neighbourhood size.
    3) For all the stored orography neighbourhoods - take the neighbours around
       the central point and create a mask where the height difference from
       the central point is greater than 35m.
    4) Loop through array of neighbourhoods and take the height and diagnostic
       of all grid points and calculate the
       diagnostic/height gradient = lapse rate
    5) Constrain the returned lapse rates between min_lapse_rate and
       max_lapse_rate. These default to > DALR and < -3.0*DALR, suitable for air_temperature
       but are user configurable
    """

    def __init__(
        self,
        max_height_diff: float = 35,
        nbhood_radius: int = 7,
        max_lapse_rate: float = -3 * DALR,
        min_lapse_rate: float = DALR,
        min_data_value: float = None,
        default: float = DALR,
    ) -> None:
        """
        The class is called with the default constraints for the processing
        code.

        Args:
            max_height_diff:
                Maximum allowable height difference between the central point
                and points in the neighbourhood over which the lapse rate will
                be calculated (metres).
                The default value of 35m is from the referenced paper.
            nbhood_radius:
                Radius of neighbourhood around each point. The neighbourhood
                will be a square array with side length 2*nbhood_radius + 1.
                The default value of 7 is from the referenced paper.
            max_lapse_rate:
                Maximum lapse rate allowed.
            min_lapse_rate:
                Minimum lapse rate allowed.
            min_data_value:
                Data values at or below this value are not included when calculating the lapse rate.
                Useful for truncated data distributions such as snow depth.
            default:
                Lapse rate to use where no lapse rate can be calculated
                (e.g. over the sea)
        """

        self.max_height_diff = max_height_diff
        self.nbhood_radius = nbhood_radius
        self.max_lapse_rate = max_lapse_rate
        self.min_lapse_rate = min_lapse_rate
        self.original_min_data_value = min_data_value
        self.min_data_value = min_data_value
        self.default = default
        self.intercept = None
        self.error_margin = None

        if self.max_lapse_rate < self.min_lapse_rate:
            msg = "Maximum lapse rate is less than minimum lapse rate"
            raise ValueError(msg)

        if not (self.min_lapse_rate <= self.default <= self.max_lapse_rate):
            msg = (
                "Default lapse rate is not between the minimum and maximum lapse rates"
            )
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

    def __repr__(self) -> str:
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

    def _create_windows(
        self, diagnostic: ndarray, orog: ndarray
    ) -> Tuple[ndarray, ndarray]:
        """Uses neighbourhood tools to pad and generate rolling windows
        of the diagnostic and orog datasets.

        Args:
            diagnostic:
                2D array (single realization) of any diagnostic data, in SI units.
            orog:
                2D array of orographies, in metres

        Returns:
            - Rolling windows of the padded diagnostic dataset.
            - Rolling windows of the padded orography dataset.
        """
        window_shape = (self.nbhood_size, self.nbhood_size)
        orog_windows = neighbourhood_tools.pad_and_roll(
            orog, window_shape, mode="constant", constant_values=np.nan
        )
        diagnostic_windows = neighbourhood_tools.pad_and_roll(
            diagnostic, window_shape, mode="constant", constant_values=np.nan
        )
        return diagnostic_windows, orog_windows

    def _generate_lapse_rate_array(
        self,
        diagnostic_data: ndarray,
        orography_data: ndarray,
        land_sea_mask_data: ndarray,
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Calculate lapse rates and apply filters

        Args:
            diagnostic_data:
                2D array (single realization) of a diagnostic data, in SI units
            orography_data:
                2D array of orographies, in metres
            land_sea_mask_data:
                2D land-sea mask

        Returns:
            Lapse rate values, zero-intercept values and error values
        """
        # Fill sea points with NaN values.
        diagnostic_data = np.where(land_sea_mask_data, diagnostic_data, np.nan)

        # Preallocate output array
        lapse_rate_array = np.empty_like(diagnostic_data, dtype=np.float32)
        intercept_array = np.empty_like(diagnostic_data, dtype=np.float32)
        error_array = np.empty_like(diagnostic_data, dtype=np.float32)

        # Pads the data with nans and generates masked windows representing
        # a neighbourhood for each point.
        diagnostic_nbhood_window, orog_nbhood_window = self._create_windows(
            diagnostic_data, orography_data
        )

        # Zips together the windows for diagnostic and orography
        # then finds the gradient of the surface diagnostic with
        # orography height - i.e. lapse rate.
        cnpt = self.ind_central_point
        axis = (-2, -1)
        for lapse, intercept, error, diag, orog in zip(
            lapse_rate_array,
            intercept_array,
            error_array,
            diagnostic_nbhood_window,
            orog_nbhood_window,
        ):
            # height_diff_mask is True for points where the height
            # difference between the central points and its
            # neighbours is greater then max_height_diff.
            orog_centre = orog[..., cnpt : cnpt + 1, cnpt : cnpt + 1]
            height_diff_mask = np.abs(orog - orog_centre) > self.max_height_diff

            diag = np.where(height_diff_mask, np.nan, diag)
            if self.min_data_value is not None:
                diag = np.where(diag <= self.min_data_value, np.nan, diag)

            # Places NaNs in orog to match diag
            orog = np.where(np.isnan(diag), np.nan, orog)

            grad, zero_point = mathematical_operations.fast_linear_fit(
                orog, diag, axis=axis, with_nan=True
            )

            margin_of_error = self._margin_of_error(
                orog, diag, grad, zero_point, axis=axis
            )

            # Checks that the standard deviations are not 0
            # i.e. there is some variance to fit a gradient to.
            diagcheck = np.isclose(np.nanstd(diag, axis=axis), 0)
            orogcheck = np.isclose(np.nanstd(orog, axis=axis), 0)
            # checks that our central point in the neighbourhood
            # is not nan
            diag_nan_check = np.isnan(diag[..., cnpt, cnpt])

            dalr_mask = diagcheck | orogcheck | diag_nan_check | np.isnan(grad)
            grad[dalr_mask] = self.default
            zero_point[dalr_mask] = np.nan

            lapse[...] = grad
            intercept[...] = zero_point
            error[...] = margin_of_error

        # Enforce upper and lower limits on lapse rate values.
        lapse_rate_array = lapse_rate_array.clip(
            self.min_lapse_rate, self.max_lapse_rate
        )
        return lapse_rate_array, intercept_array, error_array

    def _margin_of_error(
        self,
        x: np.ndarray,
        y: np.ndarray,
        slope: np.ndarray,
        intercept: np.ndarray,
        axis: tuple = None,
    ) -> np.ndarray:
        """Calculate the margin of error for the regression line.
        Args:
            x:
                The independent variable data (e.g. orography).
            y:
                The dependent variable data (e.g. diagnostic).
            slope:
                The slope of the regression line.
            intercept:
                The intercept of the regression line.
            axis:
                Axis or axes of x and y that relate to the neighbourhood used in the regression calculation.

            Returns:
                The margin of error for the regression line.
        """
        # slope and intercept require two extra trailing scalar dimensions for broadcasting to x and y
        expanded_slope = np.expand_dims(slope, axis=axis)
        expanded_intercept = np.expand_dims(intercept, axis=axis)
        # find number of elements in x in the trailing two axes that are not NaN
        valid_elements = np.count_nonzero(~np.isnan(x), axis=axis)
        se = self._standard_error(
            x, y, expanded_slope, expanded_intercept, valid_elements, axis=axis
        )
        t = self._t_score(valid_elements - 2)
        margin = t * se
        return margin

    @staticmethod
    def _standard_error(
        source_x: np.ndarray,
        source_y: np.ndarray,
        slope: np.ndarray,
        intercept: np.ndarray,
        valid_elements: np.ndarray,
        axis: tuple = None,
    ) -> np.ndarray:
        """Calculate the standard error of the regression line."""
        mean_x = np.expand_dims(np.nanmean(source_x, axis=axis), axis=axis)
        predicted_y = source_x * slope + intercept
        standard_error = (
            (np.nansum((source_y - predicted_y) ** 2, axis=axis) / (valid_elements - 2))
            ** 0.5
        ) / np.nansum((source_x - mean_x) ** 2, axis=axis) ** 0.5
        return standard_error

    @staticmethod
    def _t_score(
        degrees_of_freedom: np.ndarray, confidence_level: float = 95.0
    ) -> np.ndarray:
        """Calculate the t-score for a given confidence level and degrees of freedom.
        Args:
            degrees_of_freedom:
                The degrees of freedom for the t-distribution. This should be the number of usable data points minus 2.
            confidence_level:
                The desired confidence level (default is 95%).
        Returns:
            The t-score corresponding to the specified confidence level and degrees of freedom.
        """
        alpha = 1 - confidence_level * 0.01
        critical_probability = alpha / 2
        return scipy.stats.t.isf(critical_probability, degrees_of_freedom)

    def process(
        self,
        diagnostic: Cube,
        orography: Cube,
        land_sea_mask: Cube,
        model_id_attr: Optional[str] = None,
    ) -> Cube:
        """Calculates the lapse rate from any diagnostic and an orography cube.

        Args:
            diagnostic:
                Cube of data from which a lapse rate will be calculated. The units of the diagnostic
                should be SI units.
            orography:
                Cube containing orography data (metres)
            land_sea_mask:
                Cube containing a binary land-sea mask. True for land-points
                and False for Sea.
            model_id_attr:
                Name of the attribute used to identify the source model for
                blending. This is inherited from the input diagnostic cube.

        Returns:
            Cube containing lapse rate (K m-1)

        Raises
        ------
        TypeError: If input cubes are not cubes
        ValueError: If input cubes are the wrong units.

        """
        if not isinstance(diagnostic, iris.cube.Cube):
            msg = "Diagnostic input is not a cube, but {}"
            raise TypeError(msg.format(type(diagnostic)))

        if not isinstance(orography, iris.cube.Cube):
            msg = "Orography input is not a cube, but {}"
            raise TypeError(msg.format(type(orography)))

        if not isinstance(land_sea_mask, iris.cube.Cube):
            msg = "Land/Sea mask input is not a cube, but {}"
            raise TypeError(msg.format(type(land_sea_mask)))

        # Converts cube units.
        diagnostic_cube = diagnostic.copy()
        orography.convert_units("metres")

        # Adjusts tolerance using least-significant-digit attribute if present
        tolerance = (
            10.0 ** -diagnostic_cube.attributes.get("least_significant_digit", 7)
        ) / 2.0
        self.min_data_value = (
            self.original_min_data_value + tolerance
            if self.original_min_data_value is not None
            else None
        )

        # Extract x/y co-ordinates.
        x_coord = diagnostic_cube.coord(axis="x").name()
        y_coord = diagnostic_cube.coord(axis="y").name()

        # Extract orography and land/sea mask data.
        orography_data = next(orography.slices([y_coord, x_coord])).data
        land_sea_mask_data = next(land_sea_mask.slices([y_coord, x_coord])).data
        # Fill sea points with NaN values.
        orography_data = np.where(land_sea_mask_data, orography_data, np.nan)

        # Create list of arrays over "realization" coordinate
        has_realization_dimension = False
        original_dimension_order = None
        if diagnostic_cube.coords("realization", dim_coords=True):
            original_dimension_order = get_dim_coord_names(diagnostic_cube)
            enforce_coordinate_ordering(diagnostic_cube, "realization")
            data_slices = diagnostic_cube.data
            has_realization_dimension = True
        else:
            data_slices = [diagnostic_cube.data]

        # Calculate lapse rate for each realization
        lapse_rate_data = []
        intercept_data = []
        error_data = []
        for data_slice in data_slices:
            lapse_rate_array, intercept_array, error_array = (
                self._generate_lapse_rate_array(
                    data_slice, orography_data, land_sea_mask_data
                )
            )
            lapse_rate_data.append(lapse_rate_array)
            intercept_data.append(intercept_array)
            error_data.append(error_array)
        lapse_rate_data = np.array(lapse_rate_data)
        intercept_data = np.array(intercept_data)
        error_data = np.array(error_data)
        if not has_realization_dimension:
            lapse_rate_data = np.squeeze(lapse_rate_data)
            intercept_data = np.squeeze(intercept_data)
            error_data = np.squeeze(error_data)

        attributes = generate_mandatory_attributes(
            [diagnostic], model_id_attr=model_id_attr
        )
        lapse_rate_cube = create_new_diagnostic_cube(
            f"{diagnostic.name()}_lapse_rate",
            f"{diagnostic.units} m-1",
            diagnostic_cube,
            attributes,
            data=lapse_rate_data,
        )
        self.intercept = create_new_diagnostic_cube(
            f"{diagnostic.name()}_zero_intercept",
            diagnostic.units,
            diagnostic_cube,
            attributes,
            data=intercept_data,
        )
        self.error_margin = create_new_diagnostic_cube(
            f"{diagnostic.name()}_margin_of_error",
            diagnostic.units,
            diagnostic_cube,
            attributes,
            data=error_data,
        )

        if original_dimension_order:
            enforce_coordinate_ordering(lapse_rate_cube, original_dimension_order)

        return lapse_rate_cube
