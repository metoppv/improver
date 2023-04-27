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
This module defines the plugins required for Ensemble Copula Coupling.

"""
import warnings
from typing import List, Optional, Tuple

import iris
import numpy as np
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError, InvalidCubeError
from numpy import ndarray
from scipy import stats

import improver.ensemble_copula_coupling._scipy_continuous_distns as scipy_cont_distns
from improver import BasePlugin
from improver.calibration.utilities import convert_cube_data_to_2d
from improver.ensemble_copula_coupling.utilities import (
    choose_set_of_percentiles,
    concatenate_2d_array_with_2d_array_endpoints,
    create_cube_with_percentiles,
    get_bounds_of_distribution,
    insert_lower_and_upper_endpoint_to_1d_array,
    interpolate_multiple_rows_same_x,
    interpolate_multiple_rows_same_y,
    restore_non_percentile_dimensions,
)
from improver.metadata.probabilistic import (
    find_percentile_coordinate,
    find_threshold_coordinate,
    format_cell_methods_for_diagnostic,
    get_diagnostic_cube_name_from_probability_name,
    get_threshold_coord_name_from_probability_name,
    probability_is_above_or_below,
)
from improver.utilities.cube_checker import (
    check_cube_coordinates,
    check_for_x_and_y_axes,
)
from improver.utilities.cube_manipulation import (
    MergeCubes,
    enforce_coordinate_ordering,
    get_dim_coord_names,
)
from improver.utilities.indexing_operations import choose


class RebadgeRealizationsAsPercentiles(BasePlugin):
    """Class to rebadge realizations as percentiles."""

    def __init__(self, optimal_crps_percentiles: Optional[bool] = False):
        """Initialise the class.

        Args:
            optimal_crps_percentiles:
                If True, percentiles are computed following the
                recommendation of Bröcker, 2012 for optimising the CRPS using
                the equation: q = (i-0.5)/N, i=1,...,N, where N is the number
                of realizations. If False, percentiles are computed as equally
                spaced following the equation: q = i/(1+N), i=1,...,N.
                Defaults to False.

        References:
            Bröcker, J. (2012), Evaluating raw ensembles with the continuous
            ranked probability score. Q.J.R. Meteorol. Soc., 138: 1611-1617.
            https://doi.org/10.1002/qj.1891
            Hamill, T. M., and S. J. Colucci, 1997: Verification of Eta–RSM
            Short-Range Ensemble Forecasts. Mon. Wea. Rev., 125, 1312–1327,
            https://doi.org/10.1175/1520-0493(1997)125<1312:VOERSR>2.0.CO;2.
        """
        self.optimal_crps_percentiles = optimal_crps_percentiles

    def process(self, cube: Cube) -> Cube:
        """Convert a cube of realizations into percentiles by sorting the cube along
        the realization dimension and rebadging the realization coordinate as a
        percentile coordinate.

        Args:
            cube:
                Cube containing realizations.
        Returns:
            Cube containing percentiles.
        """
        if not cube.coords("realization"):
            coord_names = [c.name() for c in cube.coords(dim_coords=True)]
            msg = (
                "No realization coordinate is present within the input. "
                "A realization coordinate must be present, as this will be "
                "rebadged to be a percentile coordinate. The coordinates "
                f"present were: {coord_names}"
            )
            raise CoordinateNotFoundError(msg)

        result = cube.copy()
        if cube.coords("realization", dim_coords=True):
            (axis,) = cube.coord_dims("realization")
            result.data = np.sort(cube.data, axis=axis)

        result.coord("realization").rename("percentile")
        result.coord("percentile").units = "%"
        lenp = len(result.coord("percentile").points)
        if self.optimal_crps_percentiles:
            result.coord("percentile").points = np.array(
                [100 * (k + 0.5) / lenp for k in range(lenp)], dtype=np.float32
            )
        else:
            result.coord("percentile").points = np.array(
                [100 * (k + 1) / (lenp + 1) for k in range(lenp)], dtype=np.float32
            )
        return result


class RebadgePercentilesAsRealizations(BasePlugin):
    """
    Class to rebadge percentiles as ensemble realizations.
    This will allow the quantisation to percentiles to be completed, without
    a subsequent EnsembleReordering step to restore spatial correlations,
    if required.
    """

    @staticmethod
    def process(
        cube: Cube, ensemble_realization_numbers: Optional[ndarray] = None
    ) -> Cube:
        """
        Rebadge percentiles as ensemble realizations. The ensemble
        realization numbering will depend upon the number of percentiles in
        the input cube i.e. 0, 1, 2, 3, ..., n-1, if there are n percentiles.

        Args:
            cube:
                Cube containing a percentile coordinate, which will be
                rebadged as ensemble realization.
            ensemble_realization_numbers:
                An array containing the ensemble numbers required in the output
                realization coordinate. Default is None, meaning the
                realization coordinate will be numbered 0, 1, 2 ... n-1 for n
                percentiles on the input cube.

        Returns:
            Processed cube

        Raises:
            InvalidCubeError:
                If the realization coordinate already exists on the cube.
        """
        percentile_coord_name = find_percentile_coordinate(cube).name()

        # create array of percentiles from cube metadata, add in fake
        # 0th and 100th percentiles if not already included
        percentile_coords = np.sort(
            np.unique(np.append(cube.coord(percentile_coord_name).points, [0, 100]))
        )
        percentile_diffs = np.diff(percentile_coords)

        # percentiles cannot be rebadged unless they are evenly spaced,
        # centred on 50th percentile, and equally partition percentile
        # space
        if not np.isclose(np.max(percentile_diffs), np.min(percentile_diffs)):
            msg = (
                "The percentile cube provided cannot be rebadged as ensemble "
                "realizations. The input percentiles need to be equally spaced, "
                "be centred on the 50th percentile, and to equally partition percentile "
                "space. The percentiles provided were "
                f"{cube.coord(percentile_coord_name).points}"
            )
            raise ValueError(msg)

        if ensemble_realization_numbers is None:
            ensemble_realization_numbers = np.arange(
                len(cube.coord(percentile_coord_name).points), dtype=np.int32
            )

        cube.coord(percentile_coord_name).points = ensemble_realization_numbers

        # we can't rebadge if the realization coordinate already exists:
        try:
            realization_coord = cube.coord("realization")
        except CoordinateNotFoundError:
            realization_coord = None

        if realization_coord:
            raise InvalidCubeError(
                "Cannot rebadge percentile coordinate to realization "
                "coordinate because a realization coordinate already exists."
            )

        cube.coord(percentile_coord_name).rename("realization")
        cube.coord("realization").units = "1"
        cube.coord("realization").points = cube.coord("realization").points.astype(
            np.int32
        )

        return cube


class ResamplePercentiles(BasePlugin):
    """
    Class for resampling percentiles from an existing set of percentiles.
    In combination with the Ensemble Reordering plugin, this is a variant of
    Ensemble Copula Coupling.

    This class includes the ability to linearly interpolate from an
    input set of percentiles to a different output set of percentiles.

    """

    def __init__(
        self, ecc_bounds_warning: bool = False, skip_ecc_bounds: bool = False
    ) -> None:
        """
        Initialise the class.

        Args:
            ecc_bounds_warning:
                If true and ECC bounds are exceeded by the percentile values,
                a warning will be generated rather than an exception.
                Default value is FALSE.
            skip_ecc_bounds:
                If true, the usage of the ECC bounds is skipped. This has the
                effect that percentiles outside of the range given by the input
                percentiles will be computed by nearest neighbour interpolation from
                the nearest available percentile, rather than using linear interpolation
                between the nearest available percentile and the ECC bound.
        """
        self.ecc_bounds_warning = ecc_bounds_warning
        self.skip_ecc_bounds = skip_ecc_bounds

    def _add_bounds_to_percentiles_and_forecast_at_percentiles(
        self,
        percentiles: ndarray,
        forecast_at_percentiles: ndarray,
        bounds_pairing: Tuple[int, int],
    ) -> Tuple[ndarray, ndarray]:
        """
        Padding of the lower and upper bounds of the percentiles for a
        given phenomenon, and padding of forecast values using the
        constant lower and upper bounds.

        Args:
            percentiles:
                Array of percentiles from a Cumulative Distribution Function.
            forecast_at_percentiles:
                Array containing the underlying forecast values at each
                percentile.
            bounds_pairing:
                Lower and upper bound to be used as the ends of the
                cumulative distribution function.

        Returns:
            - Percentiles
            - Forecast at percentiles with endpoints

        Raises:
            ValueError: If the percentile points are outside the ECC bounds
                and self.ecc_bounds_warning is False.
            ValueError: If the percentiles are not in ascending order.

        Warns:
            Warning:  If the percentile points are outside the ECC bounds
                and self.ecc_bounds_warning is True.
        """
        lower_bound, upper_bound = bounds_pairing
        percentiles = insert_lower_and_upper_endpoint_to_1d_array(percentiles, 0, 100)
        forecast = concatenate_2d_array_with_2d_array_endpoints(
            forecast_at_percentiles, lower_bound, upper_bound
        )

        if np.any(np.diff(forecast) < 0):
            out_of_bounds_vals = forecast[np.where(np.diff(forecast) < 0)]
            msg = (
                "Forecast values exist that fall outside the expected extrema "
                "values that are defined as bounds in "
                "ensemble_copula_coupling/constants.py. "
                "Applying the extrema values as end points to the distribution "
                "would result in non-monotonically increasing values. "
                "The defined extremes are {}, whilst the following forecast "
                "values exist outside this range: {}.".format(
                    bounds_pairing, out_of_bounds_vals
                )
            )

            if self.ecc_bounds_warning:
                warn_msg = msg + (
                    " The percentile values that have "
                    "exceeded the existing bounds will be used "
                    "as new bounds."
                )
                warnings.warn(warn_msg)
                if upper_bound < forecast.max():
                    upper_bound = forecast.max()
                if lower_bound > forecast.min():
                    lower_bound = forecast.min()
                forecast = concatenate_2d_array_with_2d_array_endpoints(
                    forecast_at_percentiles, lower_bound, upper_bound
                )
            else:
                raise ValueError(msg)
        if np.any(np.diff(percentiles) < 0):
            msg = (
                "The percentiles must be in ascending order."
                "The input percentiles were {}".format(percentiles)
            )
            raise ValueError(msg)
        return percentiles, forecast

    def _interpolate_percentiles(
        self,
        forecast_at_percentiles: Cube,
        desired_percentiles: ndarray,
        percentile_coord_name: str,
    ) -> Cube:
        """
        Interpolation of forecast for a set of percentiles from an initial
        set of percentiles to a new set of percentiles. This is constructed
        by linearly interpolating between the original set of percentiles
        to a new set of percentiles.

        Args:
            forecast_at_percentiles:
                Cube containing a percentile coordinate.
            desired_percentiles:
                Array of the desired percentiles.
            percentile_coord_name:
                Name of required percentile coordinate.

        Returns:
            Cube containing values for the required diagnostic e.g.
            air_temperature at the required percentiles.
        """
        original_percentiles = forecast_at_percentiles.coord(
            percentile_coord_name
        ).points

        original_mask = None
        if np.ma.is_masked(forecast_at_percentiles.data):
            original_mask = forecast_at_percentiles.data.mask[0]

        # Ensure that the percentile dimension is first, so that the
        # conversion to a 2d array produces data in the desired order.
        enforce_coordinate_ordering(forecast_at_percentiles, percentile_coord_name)
        forecast_at_reshaped_percentiles = convert_cube_data_to_2d(
            forecast_at_percentiles, coord=percentile_coord_name
        )

        if not self.skip_ecc_bounds:
            cube_units = forecast_at_percentiles.units
            bounds_pairing = get_bounds_of_distribution(
                forecast_at_percentiles.name(), cube_units
            )
            (
                original_percentiles,
                forecast_at_reshaped_percentiles,
            ) = self._add_bounds_to_percentiles_and_forecast_at_percentiles(
                original_percentiles, forecast_at_reshaped_percentiles, bounds_pairing
            )

        forecast_at_interpolated_percentiles = interpolate_multiple_rows_same_x(
            np.array(desired_percentiles, dtype=np.float64),
            original_percentiles.astype(np.float64),
            forecast_at_reshaped_percentiles.astype(np.float64),
        )
        forecast_at_interpolated_percentiles = np.transpose(
            forecast_at_interpolated_percentiles
        )

        # Reshape forecast_at_percentiles, so the percentiles dimension is
        # first, and any other dimension coordinates follow.
        forecast_at_percentiles_data = restore_non_percentile_dimensions(
            forecast_at_interpolated_percentiles,
            next(forecast_at_percentiles.slices_over(percentile_coord_name)),
            len(desired_percentiles),
        )

        template_cube = next(forecast_at_percentiles.slices_over(percentile_coord_name))
        template_cube.remove_coord(percentile_coord_name)
        percentile_cube = create_cube_with_percentiles(
            desired_percentiles, template_cube, forecast_at_percentiles_data,
        )
        if original_mask is not None:
            original_mask = np.broadcast_to(original_mask, percentile_cube.shape)
            percentile_cube.data = np.ma.MaskedArray(
                percentile_cube.data, mask=original_mask
            )
        return percentile_cube

    def process(
        self,
        forecast_at_percentiles: Cube,
        no_of_percentiles: Optional[int] = None,
        sampling: Optional[str] = "quantile",
        percentiles: Optional[List] = None,
    ) -> Cube:
        """
        1. Creates a list of percentiles, if not provided.
        2. Accesses the lower and upper bound pair of the forecast values,
           in order to specify lower and upper bounds for the percentiles.
        3. Interpolate the percentile coordinate into an alternative
           set of percentiles using linear interpolation.

        Args:
            forecast_at_percentiles:
                Cube expected to contain a percentile coordinate.
            no_of_percentiles:
                Number of percentiles
                If None, the number of percentiles within the input
                forecast_at_percentiles cube is used as the
                number of percentiles.
            sampling:
                Type of sampling of the distribution to produce a set of
                percentiles e.g. quantile or random.

                Accepted options for sampling are:

                * Quantile: A regular set of equally-spaced percentiles aimed
                     at dividing a Cumulative Distribution Function into
                     blocks of equal probability.
                * Random: A random set of ordered percentiles.
            percentiles:
                List of the desired output percentiles.

        Returns:
            Cube with forecast values at the desired set of percentiles.
            The percentile coordinate is always the zeroth dimension.

        Raises:
            ValueError: The percentiles supplied must be between 0 and 100.
        """
        percentile_coord = find_percentile_coordinate(forecast_at_percentiles)

        if percentiles:
            if any(p < 0 or p > 100 for p in percentiles):
                msg = (
                    "The percentiles supplied must be between 0 and 100. "
                    f"Percentiles supplied: {percentiles}"
                )
                raise ValueError(msg)
        else:
            if no_of_percentiles is None:
                no_of_percentiles = len(
                    forecast_at_percentiles.coord(percentile_coord).points
                )
            percentiles = choose_set_of_percentiles(
                no_of_percentiles, sampling=sampling
            )

        forecast_at_percentiles = self._interpolate_percentiles(
            forecast_at_percentiles, percentiles, percentile_coord.name(),
        )
        return forecast_at_percentiles


class ConvertProbabilitiesToPercentiles(BasePlugin):
    """
    Class for generating percentiles from probabilities.
    In combination with the Ensemble Reordering plugin, this is a variant
    Ensemble Copula Coupling.

    This class includes the ability to interpolate between probabilities
    specified using multiple thresholds in order to generate the percentiles,
    see Figure 1 from Flowerdew, 2014.

    Scientific Reference:
    Flowerdew, J., 2014.
    Calibrated ensemble reliability whilst preserving spatial structure.
    Tellus Series A, Dynamic Meteorology and Oceanography, 66, 22662.

    """

    def __init__(self, ecc_bounds_warning: bool = False) -> None:
        """
        Initialise the class.

        Args:
            ecc_bounds_warning:
                If true and ECC bounds are exceeded by the percentile values,
                a warning will be generated rather than an exception.
                Default value is FALSE.
        """
        self.ecc_bounds_warning = ecc_bounds_warning

    def _add_bounds_to_thresholds_and_probabilities(
        self,
        threshold_points: ndarray,
        probabilities_for_cdf: ndarray,
        bounds_pairing: Tuple[int, int],
    ) -> Tuple[ndarray, ndarray]:
        """
        Padding of the lower and upper bounds of the distribution for a
        given phenomenon for the threshold_points, and padding of
        probabilities of 0 and 1 to the forecast probabilities.

        Args:
            threshold_points:
                Array of threshold values used to calculate the probabilities.
            probabilities_for_cdf:
                Array containing the probabilities used for constructing an
                cumulative distribution function i.e. probabilities
                below threshold.
            bounds_pairing:
                Lower and upper bound to be used as the ends of the
                cumulative distribution function.

        Returns:
            - Array of threshold values padded with the lower and upper
              bound of the distribution.
            - Array containing the probabilities padded with 0 and 1 at
              each end.

        Raises:
            ValueError: If the thresholds exceed the ECC bounds for
                the diagnostic and self.ecc_bounds_warning is False.

        Warns:
            Warning: If the thresholds exceed the ECC bounds for
                the diagnostic and self.ecc_bounds_warning is True.
        """
        lower_bound, upper_bound = bounds_pairing
        threshold_points_with_endpoints = insert_lower_and_upper_endpoint_to_1d_array(
            threshold_points, lower_bound, upper_bound
        )
        probabilities_for_cdf = concatenate_2d_array_with_2d_array_endpoints(
            probabilities_for_cdf, 0, 1
        )

        if np.any(np.diff(threshold_points_with_endpoints) < 0):
            msg = (
                "The calculated threshold values {} are not in ascending "
                "order as required for the cumulative distribution "
                "function (CDF). This is due to the threshold values "
                "exceeding the range given by the ECC bounds {}.".format(
                    threshold_points_with_endpoints, bounds_pairing
                )
            )
            # If ecc_bounds_warning has been set, generate a warning message
            # rather than raising an exception so that subsequent processing
            # can continue. Then apply the new bounds as necessary to
            # ensure the threshold values and endpoints are in ascending
            # order and avoid problems further along the processing chain.
            if self.ecc_bounds_warning:
                warn_msg = msg + (
                    " The threshold points that have "
                    "exceeded the existing bounds will be used "
                    "as new bounds."
                )
                warnings.warn(warn_msg)
                if upper_bound < max(threshold_points_with_endpoints):
                    upper_bound = max(threshold_points_with_endpoints)
                if lower_bound > min(threshold_points_with_endpoints):
                    lower_bound = min(threshold_points_with_endpoints)
                threshold_points_with_endpoints = insert_lower_and_upper_endpoint_to_1d_array(
                    threshold_points, lower_bound, upper_bound
                )
            else:
                raise ValueError(msg)
        return threshold_points_with_endpoints, probabilities_for_cdf

    def _probabilities_to_percentiles(
        self,
        forecast_probabilities: Cube,
        percentiles: ndarray,
        bounds_pairing: Tuple[int, int],
    ) -> Cube:
        """
        Conversion of probabilities to percentiles through the construction
        of an cumulative distribution function. This is effectively
        constructed by linear interpolation from the probabilities associated
        with each threshold to a set of percentiles.

        Args:
            forecast_probabilities:
                Cube with a threshold coordinate.
            percentiles:
                Array of percentiles, at which the corresponding values will be
                calculated.
            bounds_pairing:
                Lower and upper bound to be used as the ends of the
                cumulative distribution function.

        Returns:
            Cube containing values for the required diagnostic e.g.
            air_temperature at the required percentiles.

        Raises:
            NotImplementedError: If the threshold coordinate has an
                spp__relative_to_threshold attribute that is not either
                "above" or "below".

        Warns:
            Warning: If the probability values are not ascending, so the
                resulting cdf is not monotonically increasing.
        """
        threshold_coord = find_threshold_coordinate(forecast_probabilities)
        threshold_unit = threshold_coord.units
        threshold_points = threshold_coord.points

        original_mask = None
        if np.ma.is_masked(forecast_probabilities.data):
            original_mask = forecast_probabilities.data.mask[0]

        # Ensure that the percentile dimension is first, so that the
        # conversion to a 2d array produces data in the desired order.
        enforce_coordinate_ordering(forecast_probabilities, threshold_coord.name())
        prob_slices = convert_cube_data_to_2d(
            forecast_probabilities, coord=threshold_coord.name()
        )

        # The requirement below for a monotonically changing probability
        # across thresholds can be thwarted by precision errors of order 1E-10,
        # as such, here we round to a precision of 9 decimal places.
        prob_slices = np.around(prob_slices, 9)

        # Invert probabilities for data thresholded above thresholds.
        relation = probability_is_above_or_below(forecast_probabilities)
        if relation == "above":
            probabilities_for_cdf = 1 - prob_slices
        elif relation == "below":
            probabilities_for_cdf = prob_slices
        else:
            msg = (
                "Probabilities to percentiles only implemented for "
                "thresholds above or below a given value."
                "The relation to threshold is given as {}".format(relation)
            )
            raise NotImplementedError(msg)

        (
            threshold_points,
            probabilities_for_cdf,
        ) = self._add_bounds_to_thresholds_and_probabilities(
            threshold_points, probabilities_for_cdf, bounds_pairing
        )

        if np.any(np.diff(probabilities_for_cdf) < 0):
            msg = (
                "The probability values used to construct the "
                "Cumulative Distribution Function (CDF) "
                "must be ascending i.e. in order to yield "
                "a monotonically increasing CDF."
                "The probabilities are {}".format(probabilities_for_cdf)
            )
            warnings.warn(msg)

        # Convert percentiles into fractions.
        percentiles_as_fractions = np.array(
            [x / 100.0 for x in percentiles], dtype=np.float32
        )

        forecast_at_percentiles = interpolate_multiple_rows_same_y(
            percentiles_as_fractions.astype(np.float64),
            probabilities_for_cdf.astype(np.float64),
            threshold_points.astype(np.float64),
        )
        forecast_at_percentiles = forecast_at_percentiles.transpose()

        # Reshape forecast_at_percentiles, so the percentiles dimension is
        # first, and any other dimension coordinates follow.
        forecast_at_percentiles = restore_non_percentile_dimensions(
            forecast_at_percentiles,
            next(forecast_probabilities.slices_over(threshold_coord)),
            len(percentiles),
        )

        template_cube = next(forecast_probabilities.slices_over(threshold_coord.name()))
        template_cube.rename(
            get_diagnostic_cube_name_from_probability_name(template_cube.name())
        )
        template_cube.remove_coord(threshold_coord.name())

        percentile_cube = create_cube_with_percentiles(
            percentiles,
            template_cube,
            forecast_at_percentiles,
            cube_unit=threshold_unit,
        )

        if original_mask is not None:
            original_mask = np.broadcast_to(original_mask, percentile_cube.shape)
            percentile_cube.data = np.ma.MaskedArray(
                percentile_cube.data, mask=original_mask
            )
        return percentile_cube

    def process(
        self,
        forecast_probabilities: Cube,
        no_of_percentiles: Optional[int] = None,
        percentiles: Optional[List[float]] = None,
        sampling: str = "quantile",
    ) -> Cube:
        """
        1. Concatenates cubes with a threshold coordinate.
        2. Creates a list of percentiles.
        3. Accesses the lower and upper bound pair to find the ends of the
           cumulative distribution function.
        4. Convert the threshold coordinate into
           values at a set of percentiles using linear interpolation,
           see Figure 1 from Flowerdew, 2014.

        Args:
            forecast_probabilities:
                Cube containing a threshold coordinate.
            no_of_percentiles:
                Number of percentiles. If None and percentiles is not set,
                the number of thresholds within the input
                forecast_probabilities cube is used as the number of
                percentiles. This argument is mutually exclusive with
                percentiles.
            percentiles:
                The desired percentile values in the interval [0, 100].
                This argument is mutually exclusive with no_of_percentiles.
            sampling:
                Type of sampling of the distribution to produce a set of
                percentiles e.g. quantile or random.

                Accepted options for sampling are:

                * Quantile: A regular set of equally-spaced percentiles aimed
                          at dividing a Cumulative Distribution Function into
                          blocks of equal probability.
                * Random: A random set of ordered percentiles.

        Returns:
            Cube with forecast values at the desired set of percentiles.
            The threshold coordinate is always the zeroth dimension.

        Raises:
            ValueError: If both no_of_percentiles and percentiles are provided
        """
        if no_of_percentiles is not None and percentiles is not None:
            raise ValueError(
                "Cannot specify both no_of_percentiles and percentiles to "
                "{}".format(self.__class__.__name__)
            )

        threshold_coord = find_threshold_coordinate(forecast_probabilities)
        phenom_name = get_threshold_coord_name_from_probability_name(
            forecast_probabilities.name()
        )

        if no_of_percentiles is None:
            no_of_percentiles = len(
                forecast_probabilities.coord(threshold_coord.name()).points
            )

        if percentiles is None:
            percentiles = choose_set_of_percentiles(
                no_of_percentiles, sampling=sampling
            )
        elif not isinstance(percentiles, (tuple, list)):
            percentiles = [percentiles]
        percentiles = np.array(percentiles, dtype=np.float32)

        cube_units = forecast_probabilities.coord(threshold_coord.name()).units
        bounds_pairing = get_bounds_of_distribution(phenom_name, cube_units)

        # If a cube still has multiple realizations, slice over these to reduce
        # the memory requirements into manageable chunks.
        try:
            slices_over_realization = forecast_probabilities.slices_over("realization")
        except CoordinateNotFoundError:
            slices_over_realization = [forecast_probabilities]

        cubelist = iris.cube.CubeList([])
        for cube_realization in slices_over_realization:
            cubelist.append(
                self._probabilities_to_percentiles(
                    cube_realization, percentiles, bounds_pairing
                )
            )
        forecast_at_percentiles = cubelist.merge_cube()

        # Update cell methods on final cube
        if forecast_at_percentiles.cell_methods:
            format_cell_methods_for_diagnostic(forecast_at_percentiles)

        return forecast_at_percentiles


class ConvertLocationAndScaleParameters:
    """
    Base Class to support the plugins that compute percentiles and
    probabilities from the location and scale parameters.
    """

    def __init__(
        self, distribution: str = "norm", shape_parameters: Optional[ndarray] = None,
    ) -> None:
        """
        Initialise the class.

        In order to construct percentiles or probabilities from the location
        or scale parameter, the distribution for the resulting output needs
        to be selected. For use with the outputs from EMOS, where it has been
        assumed that the outputs from minimising the CRPS follow a particular
        distribution, then the same distribution should be selected, as used
        for the CRPS minimisation. The conversion to percentiles and
        probabilities from the location and scale parameter relies upon
        functionality within scipy.stats.

        Args:
            distribution:
                Name of a distribution supported by scipy.stats.
            shape_parameters:
                For use with distributions in scipy.stats (e.g. truncnorm) that
                require the specification of shape parameters to be able to
                define the shape of the distribution. For the truncated normal
                distribution, the shape parameters should be appropriate for
                the distribution constructed from the location and scale
                parameters provided.
                Please note that for use with
                :meth:`~improver.calibration.\
ensemble_calibration.ContinuousRankedProbabilityScoreMinimisers.\
calculate_truncated_normal_crps`,
                the shape parameters for a truncated normal distribution with
                a lower bound of zero should be [0, np.inf].

        """
        if distribution == "truncnorm":
            # Use scipy v1.3.3 truncnorm
            self.distribution = scipy_cont_distns.truncnorm
        else:
            try:
                self.distribution = getattr(stats, distribution)
            except AttributeError as err:
                msg = (
                    "The distribution requested {} is not a valid distribution "
                    "in scipy.stats. {}".format(distribution, err)
                )
                raise AttributeError(msg)

        if shape_parameters is None:
            if self.distribution.name == "truncnorm":
                raise ValueError(
                    "For the truncated normal distribution, "
                    "shape parameters must be specified."
                )
            shape_parameters = []
        self.shape_parameters = shape_parameters

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string."""
        result = (
            "<ConvertLocationAndScaleParameters: distribution: {}; "
            "shape_parameters: {}>"
        )
        return result.format(self.distribution.name, self.shape_parameters)

    def _rescale_shape_parameters(
        self, location_parameter: ndarray, scale_parameter: ndarray
    ) -> None:
        """
        Rescale the shape parameters for the desired location and scale
        parameters for the truncated normal distribution. The shape parameters
        for any other distribution will remain unchanged.

        For the truncated normal distribution, if the shape parameters are not
        rescaled, then :data:`scipy.stats.truncnorm` will assume that the shape
        parameters are appropriate for a standard normal distribution. As the
        aim is to construct a distribution using specific values for the
        location and scale parameters, the assumption of a standard normal
        distribution is not appropriate. Therefore the shape parameters are
        rescaled using the equations:

        .. math::
          a\\_rescaled = (a - location\\_parameter)/scale\\_parameter

          b\\_rescaled = (b - location\\_parameter)/scale\\_parameter

        Please see :data:`scipy.stats.truncnorm` for some further information.

        Args:
            location_parameter:
                Location parameter to be used to scale the shape parameters.
            scale_parameter:
                Scale parameter to be used to scale the shape parameters.
        """
        if self.distribution.name == "truncnorm":
            rescaled_values = []
            for value in self.shape_parameters:
                rescaled_values.append((value - location_parameter) / scale_parameter)
            self.shape_parameters = rescaled_values


class ConvertLocationAndScaleParametersToPercentiles(
    BasePlugin, ConvertLocationAndScaleParameters
):
    """
    Plugin focusing on generating percentiles from location and scale
    parameters. In combination with the EnsembleReordering plugin, this is
    Ensemble Copula Coupling.
    """

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string."""
        result = (
            "<ConvertLocationAndScaleParametersToPercentiles: "
            "distribution: {}; shape_parameters: {}>"
        )
        return result.format(self.distribution.name, self.shape_parameters)

    def _location_and_scale_parameters_to_percentiles(
        self,
        location_parameter: Cube,
        scale_parameter: Cube,
        template_cube: Cube,
        percentiles: List[float],
    ) -> Cube:
        """
        Function returning percentiles based on the supplied location and
        scale parameters.

        Args:
            location_parameter:
                Location parameter of calibrated distribution.
            scale_parameter:
                Scale parameter of the calibrated distribution.
            template_cube:
                Template cube containing either a percentile or realization
                coordinate. All coordinates apart from the percentile or
                realization coordinate will be copied from the template cube.
                Metadata will also be copied from this cube.
            percentiles:
                Percentiles at which to calculate the value of the phenomenon
                at.

        Returns:
            Cube containing the values for the phenomenon at each of the
            percentiles requested.

        Raises:
            ValueError: If any of the resulting percentile values are
                nans and these nans are not caused by a scale parameter of
                zero.
        """
        # Remove any mask that may be applied to location and scale parameters
        # and replace with ones
        location_data = np.ma.filled(location_parameter.data, 1).flatten()
        scale_data = np.ma.filled(scale_parameter.data, 1).flatten()

        # Convert percentiles into fractions.
        percentiles_as_fractions = np.array(
            [x / 100.0 for x in percentiles], dtype=np.float32
        )

        result = np.zeros(
            (len(percentiles_as_fractions), location_data.shape[0]), dtype=np.float32
        )

        self._rescale_shape_parameters(location_data, scale_data)

        percentile_method = self.distribution(
            *self.shape_parameters, loc=location_data, scale=scale_data
        )

        # Loop over percentiles, and use the distribution as the
        # "percentile_method" with the location and scale parameter to
        # calculate the values at each percentile.
        for index, percentile in enumerate(percentiles_as_fractions):
            percentile_list = np.repeat(percentile, len(location_data))
            result[index, :] = percentile_method.ppf(percentile_list)
            # If percent point function (PPF) returns NaNs, fill in
            # mean instead of NaN values. NaN will only be generated if the
            # scale parameter (standard deviation) is zero. Therefore, if the
            # scale parameter (standard deviation) is zero, the mean value is
            # used for all gridpoints with a NaN.
            if np.any(scale_data == 0):
                nan_index = np.argwhere(np.isnan(result[index, :]))
                result[index, nan_index] = location_data[nan_index]
            if np.any(np.isnan(result)):
                msg = (
                    "NaNs are present within the result for the {} "
                    "percentile. Unable to calculate the percent point "
                    "function."
                )
                raise ValueError(msg)

        # Reshape forecast_at_percentiles, so the percentiles dimension is
        # first, and any other dimension coordinates follow.
        result = result.reshape((len(percentiles),) + location_parameter.data.shape)

        for prob_coord_name in ["realization", "percentile"]:
            if template_cube.coords(prob_coord_name, dim_coords=True):
                prob_coord = template_cube.coord(prob_coord_name)
                template_slice = next(template_cube.slices_over(prob_coord))
                template_slice.remove_coord(prob_coord)
            elif template_cube.coords(prob_coord_name, dim_coords=False):
                template_slice = template_cube

        percentile_cube = create_cube_with_percentiles(
            percentiles, template_slice, result
        )
        # Define a mask to be reapplied later
        mask = np.logical_or(
            np.ma.getmaskarray(location_parameter.data),
            np.ma.getmaskarray(scale_parameter.data),
        )
        # Make the mask defined above fit the data size and then apply to the
        # percentile cube.
        mask_array = np.stack([mask] * len(percentiles))
        percentile_cube.data = np.ma.masked_where(mask_array, percentile_cube.data)
        # Remove cell methods associated with finding the ensemble mean
        percentile_cube.cell_methods = {}
        return percentile_cube

    def process(
        self,
        location_parameter: Cube,
        scale_parameter: Cube,
        template_cube: Cube,
        no_of_percentiles: Optional[int] = None,
        percentiles: Optional[List[float]] = None,
    ) -> Cube:
        """
        Generate ensemble percentiles from the location and scale parameters.

        Args:
            location_parameter:
                Cube containing the location parameters.
            scale_parameter:
                Cube containing the scale parameters.
            template_cube:
                Template cube containing either a percentile or realization
                coordinate. All coordinates apart from the percentile or
                realization coordinate will be copied from the template cube.
                Metadata will also be copied from this cube.
            no_of_percentiles:
                Integer defining the number of percentiles that will be
                calculated from the location and scale parameters.
            percentiles:
                List of percentiles that will be generated from the location
                and scale parameters provided.

        Returns:
            Cube for calibrated percentiles.
            The percentile coordinate is always the zeroth dimension.

        Raises:
            ValueError: Ensure that it is not possible to supply
                "no_of_percentiles" and "percentiles" simultaneously
                as keyword arguments.
        """
        if no_of_percentiles and percentiles:
            msg = (
                "Please specify either the number of percentiles or "
                "provide a list of percentiles. The number of percentiles "
                "provided was {} and the list of percentiles "
                "provided was {}".format(no_of_percentiles, percentiles)
            )
            raise ValueError(msg)

        if no_of_percentiles:
            percentiles = choose_set_of_percentiles(no_of_percentiles)
        calibrated_forecast_percentiles = self._location_and_scale_parameters_to_percentiles(
            location_parameter, scale_parameter, template_cube, percentiles
        )

        return calibrated_forecast_percentiles


class ConvertLocationAndScaleParametersToProbabilities(
    BasePlugin, ConvertLocationAndScaleParameters
):
    """
    Plugin to generate probabilities relative to given thresholds from the
    location and scale parameters of a distribution.
    """

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string."""
        result = (
            "<ConvertLocationAndScaleParametersToProbabilities: "
            "distribution: {}; shape_parameters: {}>"
        )
        return result.format(self.distribution.name, self.shape_parameters)

    def _check_template_cube(self, cube: Cube) -> None:
        """
        The template cube is expected to contain a leading threshold dimension
        followed by spatial (y/x) dimensions for a gridded cube. For a spot
        template cube, the spatial dimensions are not expected to be dimension
        coordinates. If the cube contains the expected dimensions,
        a threshold leading order is enforced.

        Args:
            cube:
                A cube whose dimensions are checked to ensure they match what
                is expected.

        Raises:
            ValueError: If cube is not of the expected dimensions.
        """
        require_dim_coords = False if cube.coords("wmo_id") else True
        check_for_x_and_y_axes(cube, require_dim_coords=require_dim_coords)
        dim_coords = get_dim_coord_names(cube)
        msg = (
            "{} expects a cube with only a leading threshold dimension, "
            "followed by spatial (y/x) dimensions. "
            "Got dimensions: {}".format(self.__class__.__name__, dim_coords)
        )

        try:
            threshold_coord = find_threshold_coordinate(cube)
        except CoordinateNotFoundError:
            raise ValueError(msg)

        if len(dim_coords) < 4:
            enforce_coordinate_ordering(cube, threshold_coord.name())
            return

        raise ValueError(msg)

    @staticmethod
    def _check_unit_compatibility(
        location_parameter: Cube, scale_parameter: Cube, probability_cube_template: Cube
    ) -> None:
        """
        The location parameter, scale parameters, and threshold values come
        from three different cubes. This is a sanity check to ensure the units
        are as expected, converting units of the location parameter and
        scale parameter if possible.

        Args:
            location_parameter:
                Cube of location parameter values.
            scale_parameter:
                Cube of scale parameter values.
            probability_cube_template:
                Cube containing threshold values.

        Raises:
            ValueError: If units of input cubes are not compatible.
        """
        threshold_units = find_threshold_coordinate(probability_cube_template).units

        try:
            location_parameter.convert_units(threshold_units)
            scale_parameter.convert_units(threshold_units)
        except ValueError as err:
            msg = (
                "Error: {} This is likely because the location parameter, "
                "scale parameter and template cube threshold units are "
                "not equivalent/compatible.".format(err)
            )
            raise ValueError(msg)

    def _location_and_scale_parameters_to_probabilities(
        self,
        location_parameter: Cube,
        scale_parameter: Cube,
        probability_cube_template: Cube,
    ) -> Cube:
        """
        Function returning probabilities relative to provided thresholds based
        on the supplied location and scale parameters.

        Args:
            location_parameter:
                Predictor for the calibrated forecast location parameter.
            scale_parameter:
                Scale parameter for the calibrated forecast.
            probability_cube_template:
                A probability cube that has a threshold coordinate, where the
                probabilities are defined as above or below the threshold by
                the spp__relative_to_threshold attribute. This cube matches
                the desired output cube format.

        Returns:
            Cube containing the data expressed as probabilities relative to
            the provided thresholds in the way described by
            spp__relative_to_threshold.
        """
        # Define a mask to be reapplied later
        loc_mask = np.ma.getmaskarray(location_parameter.data)
        scale_mask = np.ma.getmaskarray(scale_parameter.data)
        mask = np.logical_or(loc_mask, scale_mask)
        # Remove any mask that may be applied to location and scale parameters
        # and replace with ones
        location_parameter.data = np.ma.filled(location_parameter.data, 1)
        scale_parameter.data = np.ma.filled(scale_parameter.data, 1)
        thresholds = find_threshold_coordinate(probability_cube_template).points
        relative_to_threshold = probability_is_above_or_below(probability_cube_template)

        self._rescale_shape_parameters(
            location_parameter.data.flatten(), scale_parameter.data.flatten()
        )

        # Loop over thresholds, and use the specified distribution with the
        # location and scale parameter to calculate the probabilities relative
        # to each threshold.
        probabilities = np.empty_like(probability_cube_template.data)

        distribution = self.distribution(
            *self.shape_parameters,
            loc=location_parameter.data.flatten(),
            scale=scale_parameter.data.flatten(),
        )

        probability_method = distribution.cdf
        if relative_to_threshold == "above":
            probability_method = distribution.sf

        for index, threshold in enumerate(thresholds):
            probabilities[index, ...] = np.reshape(
                probability_method(threshold), probabilities.shape[1:]
            )

        probability_cube = probability_cube_template.copy(data=probabilities)
        # Make the mask defined above fit the data size and then apply to the
        # probability cube.
        mask_array = np.array([mask] * len(probabilities))
        probability_cube.data = np.ma.masked_where(mask_array, probability_cube.data)
        return probability_cube

    def process(
        self,
        location_parameter: Cube,
        scale_parameter: Cube,
        probability_cube_template: Cube,
    ) -> Cube:
        """
        Generate probabilities from the location and scale parameters of the
        distribution.

        Args:
            location_parameter:
                Cube containing the location parameters.
            scale_parameter:
                Cube containing the scale parameters.
            probability_cube_template:
                A probability cube that has a threshold coordinate, where the
                probabilities are defined as above or below the threshold by
                the spp__relative_to_threshold attribute. This cube matches
                the desired output cube format.

        Returns:
            A cube of diagnostic data expressed as probabilities relative
            to the thresholds found in the probability_cube_template.
        """
        self._check_template_cube(probability_cube_template)
        self._check_unit_compatibility(
            location_parameter, scale_parameter, probability_cube_template
        )

        probability_cube = self._location_and_scale_parameters_to_probabilities(
            location_parameter, scale_parameter, probability_cube_template
        )

        return probability_cube


class EnsembleReordering(BasePlugin):
    """
    Plugin for applying the reordering step of Ensemble Copula Coupling,
    in order to generate ensemble realizations with multivariate structure
    from percentiles. The percentiles are assumed to be in ascending order.

    Reference:
    Schefzik, R., Thorarinsdottir, T.L. & Gneiting, T., 2013.
    Uncertainty Quantification in Complex Simulation Models Using Ensemble
    Copula Coupling.
    Statistical Science, 28(4), pp.616-640.

    """

    @staticmethod
    def _recycle_raw_ensemble_realizations(
        post_processed_forecast_percentiles: Cube,
        raw_forecast_realizations: Cube,
        percentile_coord_name: str,
    ) -> Cube:
        """
        Function to determine whether there is a mismatch between the number
        of percentiles and the number of raw forecast realizations. If more
        percentiles are requested than ensemble realizations, then the ensemble
        realizations are recycled. This assumes that the identity of the
        ensemble realizations within the raw ensemble forecast is random, such
        that the raw ensemble realizations are exchangeable. If fewer
        percentiles are requested than ensemble realizations, then only the
        first n ensemble realizations are used.

        Args:
            post_processed_forecast_percentiles :
                Cube for post-processed percentiles.
                The percentiles are assumed
                to be in ascending order.
            raw_forecast_realizations:
                Cube containing the raw (not post-processed) forecasts.
            percentile_coord_name:
                Name of required percentile coordinate.

        Returns:
            Cube for the raw ensemble forecast, where the raw ensemble
            realizations have either been recycled or constrained,
            depending upon the number of percentiles present
            in the post-processed forecast cube.
        """
        plen = len(
            post_processed_forecast_percentiles.coord(percentile_coord_name).points
        )
        mlen = len(raw_forecast_realizations.coord("realization").points)
        if plen == mlen:
            pass
        else:
            raw_forecast_realizations_extended = iris.cube.CubeList()
            realization_list = []
            mpoints = raw_forecast_realizations.coord("realization").points
            # Loop over the number of percentiles and finding the
            # corresponding ensemble realization number. The ensemble
            # realization numbers are recycled e.g. 1, 2, 3, 1, 2, 3, etc.
            for index in range(plen):
                realization_list.append(mpoints[index % len(mpoints)])

            # Assume that the ensemble realizations are ascending linearly.
            new_realization_numbers = realization_list[0] + list(range(plen))

            # Extract the realizations required in the realization_list from
            # the raw_forecast_realizations. Edit the realization number as
            # appropriate and append to a cubelist containing rebadged
            # raw ensemble realizations.
            for realization, index in zip(realization_list, new_realization_numbers):
                constr = iris.Constraint(realization=realization)
                raw_forecast_realization = raw_forecast_realizations.extract(constr)
                raw_forecast_realization.coord("realization").points = index
                raw_forecast_realizations_extended.append(raw_forecast_realization)
            raw_forecast_realizations = MergeCubes()(
                raw_forecast_realizations_extended, slice_over_realization=True
            )
        return raw_forecast_realizations

    @staticmethod
    def rank_ecc(
        post_processed_forecast_percentiles: Cube,
        raw_forecast_realizations: Cube,
        random_ordering: bool = False,
        random_seed: Optional[int] = None,
    ) -> Cube:
        """
        Function to apply Ensemble Copula Coupling. This ranks the
        post-processed forecast realizations based on a ranking determined from
        the raw forecast realizations.

        Args:
            post_processed_forecast_percentiles:
                Cube for post-processed percentiles. The percentiles are
                assumed to be in ascending order.
            raw_forecast_realizations:
                Cube containing the raw (not post-processed) forecasts.
                The probabilistic dimension is assumed to be the zeroth
                dimension.
            random_ordering:
                If random_ordering is True, the post-processed forecasts are
                reordered randomly, rather than using the ordering of the
                raw ensemble.
            random_seed:
                If random_seed is an integer, the integer value is used for
                the random seed.
                If random_seed is None, no random seed is set, so the random
                values generated are not reproducible.

        Returns:
            Cube for post-processed realizations where at a particular grid
            point, the ranking of the values within the ensemble matches
            the ranking from the raw ensemble.
        """
        results = iris.cube.CubeList([])
        for rawfc, calfc in zip(
            raw_forecast_realizations.slices_over("time"),
            post_processed_forecast_percentiles.slices_over("time"),
        ):
            if random_seed is not None:
                random_seed = int(random_seed)
            random_seed = np.random.RandomState(random_seed)
            random_data = random_seed.rand(*rawfc.data.shape)
            if random_ordering:
                # Returns the indices that would sort the array.
                # As these indices are from a random dataset, only an argsort
                # is used.
                ranking = np.argsort(random_data, axis=0)
            else:
                # Lexsort returns the indices sorted firstly by the
                # primary key, the raw forecast data (unless random_ordering
                # is enabled), and secondly by the secondary key, an array of
                # random data, in order to split tied values randomly.
                sorting_index = np.lexsort((random_data, rawfc.data), axis=0)
                # Returns the indices that would sort the array.
                ranking = np.argsort(sorting_index, axis=0)
            # Index the post-processed forecast data using the ranking array.
            # The following uses a custom choose function that reproduces the
            # required elements of the np.choose method without the limitation
            # of having < 32 arrays or a leading dimension < 32 in the
            # input data array. This function allows indexing of a 3d array
            # using a 3d array.
            mask = np.ma.getmask(calfc.data)
            calfc.data = choose(ranking, calfc.data)
            if mask is not np.ma.nomask:
                calfc.data = np.ma.MaskedArray(calfc.data, mask, dtype=np.float32)
            results.append(calfc)
        # Ensure we haven't lost any dimensional coordinates with only one
        # value in.
        results = results.merge_cube()
        results = check_cube_coordinates(post_processed_forecast_percentiles, results)
        return results

    @staticmethod
    def _check_input_cube_masks(post_processed_forecast, raw_forecast):
        """
        Checks that if the raw_forecast is masked the post_processed_forecast
        is also masked. The code supports the post_processed_forecast being
        masked even if the raw_forecast isn't masked, but not vice versa.

        If both post_processed_forecast and raw_forecast are masked checks
        that both input cubes have the same mask applied to each
        x-y slice.

        Args:
            post_processed_forecast:
                The cube containing the post-processed
                forecast realizations.
            raw_forecast:
                The cube containing the raw (not post-processed)
                forecast.

        Raises:
            ValueError:
                If only the raw_forecast is masked
            ValueError:
                If the post_processed_forecast does not have same mask on all
                x-y slices
            ValueError:
                If the raw_forecast x-y slices do not all have the same mask
                as the post_processed_forecast.
        """
        if np.ma.is_masked(post_processed_forecast.data) and np.ma.is_masked(
            raw_forecast.data
        ):
            for aslice in post_processed_forecast.data.mask[1:, ...]:
                if np.any(aslice != post_processed_forecast.data.mask[0]):

                    message = (
                        "The post_processed_forecast does not have same"
                        " mask on all x-y slices"
                    )
                    raise (ValueError(message))
            for aslice in raw_forecast.data.mask[0:, ...]:
                if np.any(aslice != post_processed_forecast.data.mask[0]):
                    message = (
                        "The raw_forecast x-y slices do not all have the"
                        " same mask as the post_processed_forecast."
                    )
                    raise (ValueError(message))
        if np.ma.is_masked(raw_forecast.data) and not np.ma.is_masked(
            post_processed_forecast.data
        ):
            message = (
                "The raw_forecast provided has a mask, but the "
                "post_processed_forecast isn't masked. The "
                "post_processed_forecast and the raw_forecast should "
                "have the same mask applied to them."
            )
            raise (ValueError(message))

    def process(
        self,
        post_processed_forecast: Cube,
        raw_forecast: Cube,
        random_ordering: bool = False,
        random_seed: Optional[int] = None,
    ) -> Cube:
        """
        Reorder post-processed forecast using the ordering of the
        raw ensemble.

        Args:
            post_processed_forecast:
                The cube containing the post-processed
                forecast realizations.
            raw_forecast:
                The cube containing the raw (not post-processed)
                forecast.
            random_ordering:
                If random_ordering is True, the post-processed forecasts are
                reordered randomly, rather than using the ordering of the
                raw ensemble.
            random_seed:
                If random_seed is an integer, the integer value is used for
                the random seed.
                If random_seed is None, no random seed is set, so the random
                values generated are not reproducible.

        Returns:
            Cube containing the new ensemble realizations where all points
            within the dataset have been reordered in comparison to the
            input percentiles. This cube contains the same ensemble
            realization numbers as the raw forecast.
        """
        percentile_coord_name = find_percentile_coordinate(
            post_processed_forecast
        ).name()

        enforce_coordinate_ordering(post_processed_forecast, percentile_coord_name)
        enforce_coordinate_ordering(raw_forecast, "realization")

        self._check_input_cube_masks(post_processed_forecast, raw_forecast)

        raw_forecast = self._recycle_raw_ensemble_realizations(
            post_processed_forecast, raw_forecast, percentile_coord_name
        )
        post_processed_forecast_realizations = self.rank_ecc(
            post_processed_forecast,
            raw_forecast,
            random_ordering=random_ordering,
            random_seed=random_seed,
        )
        plugin = RebadgePercentilesAsRealizations()
        post_processed_forecast_realizations = plugin(
            post_processed_forecast_realizations,
            ensemble_realization_numbers=raw_forecast.coord("realization").points,
        )

        enforce_coordinate_ordering(post_processed_forecast_realizations, "realization")
        return post_processed_forecast_realizations
