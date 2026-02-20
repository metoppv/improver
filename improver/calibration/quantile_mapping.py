# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing quantile mapping bias correction.

Quantile mapping is a statistical calibration technique that adjusts forecast
values to match the distribution of reference data (i.e. observations or a differently
processed reference forecast). It works by:
1. Finding each forecast value's position (quantile) in the forecast distribution
2. Mapping that quantile to the corresponding value in the reference distribution

This corrects systematic biases while preserving spatial patterns.
"""

from typing import Literal, Optional, Tuple

import numpy as np
from iris.cube import Cube

from improver import PostProcessingPlugin


class QuantileMapping(PostProcessingPlugin):
    """Apply quantile mapping bias correction to forecast data."""

    def __init__(
        self,
        preservation_threshold: Optional[float] = None,
        method: Literal["step", "continuous"] = "step",
    ) -> None:
        """Initialize the quantile mapping plugin.

            Args:
                preservation_threshold:
                    Optional threshold value below which (exclusive) the forecast
                    values are not adjusted to be like the reference. Useful for
                    variables such as precipitation, where a user may be wary of mapping
                    0mm/hr precipitation values to non-zero values.
                method:
                    Choose from two methods of converting forecast values into quantiles
                    before mapping them onto the reference distribution: 'step' and
                    'continuous'. These methods differ in three ways:
                    1. How quantiles are assigned to ranked data ('plotting positions').
                    - 'step' uses rank/number of points (i/n), which corresponds to the
                    formal ECDF definition and treats the largest value as the 1.0
                    quantile (100th percentile).
                    - 'continuous' uses midpoint plotting positions ((i-0.5)/n), which
                    place values in the centre of their rank intervals and avoids
                    probabilities of exactly 0 or 1.
                    2. How probabilities are mapped back to values.
                    - 'step' uses flooring, so each probability maps to the nearest
                    lower observed value in the reference distribution, creating the
                    step-function mapping.
                    - 'continuous' uses interpolation, creating a smoother mapping where
                    small changes in probability lead to small changes in value.
                    3. How repeated values are treated.
                    - 'step' assigns the same quantile to repeated values, so they all
                    map to the same value in the reference distribution (creating flat
                    steps in the mapping).
                    - 'continuous' assigns different quantiles to repeated values,
                    spreading them evenly across their range, so they can map to
                    different values in the reference distribution.

                    Example
                    --------
                    With the following reference and forecast data (totalling 11 points
                    in each array), the two methods would produce their output as
                    illustrated below:
                        forecast = np.array([0, 0, 0, 0, 0, 0, 0, 0, 10, 20, 30])
                        reference = np.array([0, 0, 0, 0, 0, 0, 0, 10, 20, 40, 50])
                        num_points = 11

                    ---- Step method ----

                    1. The forecast data are sorted.

                        [0, 0, 0, 0, 0, 0, 0, 0, 10, 20, 30]

                    2. ECDF quantiles are assigned using i/n, where i is the number of
                    values less than or equal to each value.

                        ECDF counts:
                        [8, 8, 8, 8, 8, 8, 8, 8, 9, 10, 11]

                        Quantiles:
                        [8/11, 8/11, 8/11, 8/11, 8/11, 8/11, 8/11, 8/11, 9/11, 10/11,
                        11/11]
                        ≈ [0.727, ..., 0.727, 0.818, 0.909, 1.0]

                    3. These quantiles are mapped to the reference distribution using a
                    stepwise empirical quantile mapping. Each probability is mapped to
                    the reference value at the right edge of the corresponding ECDF step
                    (i.e. the smallest reference value whose empirical cumulative
                    probability is greater than or equal to the given probability),
                    yielding:

                        [10, 10, 10, 10, 10, 10, 10, 10, 20, 40, 50]

                    ---- Continuous method ----

                    1. The forecast data are ranked using a stable sorting algorithm
                    (np.argsort with kind='mergesort'), which assigns ranks in order of
                    appearance for repeated values:

                        Ranks:
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

                    2. Midpoint quantiles are assigned using (rank + 0.5) / n:

                        Quantiles:
                        [0.045, 0.136, 0.227, 0.318, 0.409, 0.500,
                        0.591, 0.682, 0.773, 0.864, 0.955]

                    3. These quantiles are mapped to the reference distribution using
                    linear interpolation between reference quantiles, yielding
                    smoothly varying mapped values. Repeated forecast values may
                    therefore map to different reference values rather than
                    collapsing to a single step.

                        [0, 0, 0, 0, 0, 0, 0, 10, 20, 40, 50]

                    Note: Due to statistical convention, the 'step' method uses standard
                    plotting positions (i/n), rather than (i/(n+1)). The consequences of
                    this choice are that the quantiles assigned to the forecast data
                    will be asymmetrically distributed: i/n produces quantiles ranging
                    from 1/n to 1, so probabilities are shifted upwards, especially near
                    the top end of the distribution. While the discrepancies are large
                    for small datasets (e.g. for n=5, quantiles are [0.2, 0.4, 0.6, 0.8,
                    1.0] vs [0.16666667, 0.33333333, 0.50000000, 0.66666667,
                    0.83333333]),the differences become negligible for larger datasets
                    (e.g. for n=1000,quantiles are [0.001, 0.002, ..., 0.999, 1.0] vs
                    [0.0005, 0.0015, ..., 0.9985, 0.9995]).
        Raises:
                ValueError:
                    If an unsupported method is specified.

        """
        self.preservation_threshold = preservation_threshold
        method = method.lower()
        if method not in ["step", "continuous"]:
            raise ValueError(
                f"Unsupported method '{method}'. Choose 'step' or 'continuous'."
            )
        self.method = method

    def _plotting_positions(self, num_points: int) -> np.ndarray:
        """
        Return plotting positions for a sorted sample of size n.

        The plotting positions determine the quantiles assigned to each data point
        when building the empirical CDF.
        """
        i = np.arange(1, num_points + 1)
        if self.method == "step":  # Standard plotting positions
            return i / num_points
        else:  # self.method == "continuous"
            return (i - 0.5) / num_points  # Midpoint plotting positions

    def _build_empirical_cdf(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build ECDF components (sorted values and quantiles).

        Args:
            data:
                1D array of input data values.

        Returns:
            Tuple of (sorted_values, quantiles) representing the empirical CDF.
        """
        sorted_values = np.sort(data)
        quantiles = self._plotting_positions(sorted_values.size)
        return sorted_values, quantiles

    def _forecast_to_quantiles(self, forecast_data: np.ndarray) -> np.ndarray:
        """Assign a quantile to each value in the forecast data based on its rank in the
        forecast distribution.

        Args:
            forecast_data:
                1D array of forecast values.

        Returns:
            1D array of quantiles corresponding to each forecast value.
        """
        num_points = forecast_data.size

        if self.method == "step":
            # Value -> quantile mapping. Repeated values collapse to the same
            # (right-most) quantile, creating a step-function mapping.
            sorted_values, quantiles = self._build_empirical_cdf(forecast_data)
            return np.interp(forecast_data, sorted_values, quantiles)
        else:  # self.method == 'continuous'
            # Rank-based quantiles: repeated values get their own quantiles spread
            # evenly across their range.
            order = np.argsort(forecast_data, kind="mergesort")
            ranks = np.empty_like(order)
            ranks[order] = np.arange(num_points)

            # Convert ranks to midpoint quantiles
            return (ranks + 0.5) / num_points

    def _inverted_cdf(
        self,
        reference_data: np.ndarray,
        quantiles: np.ndarray,
    ) -> np.ndarray:
        """Get distribution values at specified quantiles (discrete step method).

        Uses floored index lookup, rounding each quantile down to the nearest
        available data point. This creates a step-function mapping that's faster
        but less smooth than interpolation.

        Taken from:
        https://github.com/ecmwf-projects/ibicus/blob/main/ibicus/utils/_math_utils.py

        Args:
            reference_data:
                1D array of data values defining the reference distribution.
            quantiles:
                Quantiles to evaluate (values between 0 and 1).

        Returns:
            Values from the reference data corresponding to the requested quantiles.
        """
        sorted_reference = np.sort(reference_data)
        num_points = sorted_reference.size

        if self.method == "step":
            idx = np.floor((num_points - 1) * quantiles).astype(np.int32)
            idx = np.clip(idx, 0, num_points - 1)  # Ensure indices are within bounds
            return sorted_reference[idx]
        else:  # self.method == 'continuous'
            quantiles_reference = (np.arange(num_points) + 0.5) / num_points
            return np.interp(quantiles, quantiles_reference, sorted_reference)

    def _map_quantiles(
        self,
        reference_data: np.ndarray,
        forecast_data: np.ndarray,
    ) -> np.ndarray:
        """Transform forecast values to match the reference distribution.

        Behaviour depends on the self.method (see __init__).

        For each forecast value:

        1. Find its quantile position in the forecast distribution
        2. Map that quantile to the corresponding value in the reference distribution
           using the specified method (step or continuous).

        Examples:
        - Discrete
            If reference_data is [10, 20, 30, 40, 50] and forecast_data
            is [20, 25, 30, 35, 40], the forecast values are mapped to the corresponding
            values in the reference data distribution. This stretches the range of the
            forecast data, shifting the extreme values by 10 units in opposing
            directions. The median value is left unchanged as the two distributions are
            aligned at this point. The inter-quartile values are each shifted by 5 units
            in opposing directions, again reflecting the broader distribution found in
            the reference data.
        - Continuous
            Using the same reference and forecast data as above, the continuous method
            would produce a smoother mapping. The extreme values would still be shifted
            by 10 units, but the intermediate values would be adjusted more gradually.
            The median value would still be unchanged, but the inter-quartile values
            would be shifted by less than 5 units, reflecting the more continuous nature
            of the mapping.

        Args:
            reference_data:
                Target distribution (observed/historical data).
            forecast_data:
                Source distribution (biased forecasts to correct).

        Returns:
            Bias-corrected forecast values matching the reference distribution.
        """
        # Convert forecast values to quantiles in the forecast distribution
        forecast_quantiles = self._forecast_to_quantiles(forecast_data)

        # Map forecast quantiles to values in the reference distribution
        corrected_values = self._inverted_cdf(reference_data, forecast_quantiles)
        return corrected_values

    @staticmethod
    def _convert_reference_cube_to_forecast_units(
        reference_cube: Cube,
        forecast_cube: Cube,
    ) -> tuple[Cube, Cube]:
        """Ensure reference cube uses the same units as forecast cube.

        Args:
            reference_cube:
                The reference data cube.
            forecast_cube:
                The forecast data cube.

        Returns:
            Tuple of (reference_cube, forecast_cube) with matching units.

        Raises:
            ValueError: If units are incompatible and cannot be converted.
        """
        target_units = forecast_cube.units
        reference_cube = reference_cube.copy()

        # Convert reference_cube to target_units if needed
        if reference_cube.units != target_units:
            try:
                reference_cube.convert_units(target_units)
            except ValueError:
                raise ValueError(
                    f"Cannot convert cube with units {reference_cube.units} "
                    f"to target units {target_units}"
                )

        return (reference_cube, forecast_cube)

    def _process_masked_data(
        self,
        reference_cube: Cube,
        forecast_cube: Cube,
    ) -> np.ndarray:
        """Apply quantile mapping while properly handling masked data.

        Masked values are excluded from the calibration CDFs to avoid
        contaminating the statistics. They are preserved in their original
        (masked) state in the output.

        Args:
            reference_cube:
                The reference cube (with units already converted).
            forecast_cube:
                The forecast cube to calibrate.

        Returns:
                corrected_data_flat: 1D array with corrected values.

        """
        # Determine if either cube has masked data
        forecast_is_masked = np.ma.is_masked(forecast_cube.data)
        reference_is_masked = np.ma.is_masked(reference_cube.data)

        if forecast_is_masked or reference_is_masked:
            # Create combined mask using getmaskarray (returns False array if unmasked)
            combined_mask = np.ma.getmaskarray(forecast_cube.data) | np.ma.getmaskarray(
                reference_cube.data
            )

            # Flatten and get valid (non-masked) indices
            combined_mask_flat = combined_mask.flatten()
            valid_mask = ~combined_mask_flat

            # Extract underlying data arrays (ignoring masks temporarily)
            # We need the full arrays to reconstruct later, but will only
            # use valid_mask indices for quantile mapping calculations
            reference_data_flat = np.ma.getdata(reference_cube.data).flatten()
            forecast_data_flat = np.ma.getdata(forecast_cube.data).flatten()

            # Extract ONLY valid (non-masked) values for CDF calculations
            # Masked values are not included in these arrays
            reference_valid = reference_data_flat[valid_mask]
            forecast_valid = forecast_data_flat[valid_mask]

            # Apply quantile mapping using only valid values
            corrected_valid = self._map_quantiles(reference_valid, forecast_valid)

            # Reconstruct full array with corrected values at valid positions
            corrected_values_flat = forecast_data_flat.copy()
            corrected_values_flat[valid_mask] = corrected_valid
            corrected_values_flat = np.ma.masked_array(
                corrected_values_flat, mask=combined_mask_flat
            )

        else:
            # No masking needed
            corrected_values_flat = self._map_quantiles(
                reference_cube.data.flatten(),
                forecast_cube.data.flatten(),
            )

        return corrected_values_flat

    def _apply_preservation_threshold(
        self, output_cube: Cube, forecast_cube: Cube
    ) -> None:
        """Preserve original values below preservation threshold.

        Modifies output_cube.data in-place.

        Args:
            output_cube:
                The cube with calibrated data to modify.
            forecast_cube:
                The original forecast cube with values to preserve.
        """
        if self.preservation_threshold is None:
            return

        mask_below_threshold = np.ma.less(
            forecast_cube.data, self.preservation_threshold
        )
        # np.ma.where works for both masked and non-masked arrays
        output_cube.data = np.ma.where(
            mask_below_threshold, forecast_cube.data, output_cube.data
        )

    def _finalise_output_cube(
        self, corrected_values_flat: np.ndarray, forecast_cube: Cube, output_cube: Cube
    ) -> None:
        """Make final adjustments to output cube metadata and data type.
        Args:
            corrected_values_flat:
                1D array of corrected values to reshape and insert into output cube.
            forecast_cube:
                The original forecast cube, used to determine the shape and for
                preservation threshold.
            output_cube:
                The cube to finalize.
            output_mask:
                The mask to apply to the output cube, or None if no masking is needed.
        """
        # Reshape corrected data to match original shape and set data type to float32
        if corrected_values_flat.dtype != np.float32:
            corrected_values_flat = corrected_values_flat.astype(np.float32)

        output_cube.data = np.reshape(corrected_values_flat, forecast_cube.shape)

        # Preserve low values if threshold is set, modifying in-place
        self._apply_preservation_threshold(output_cube, forecast_cube)

    def process(
        self,
        reference_cube: Cube,
        forecast_cube: Cube,
    ) -> Cube:
        """Adjust forecast values to match the statistical distribution of reference
        data.

        This calibration method corrects biases in forecast data by transforming its
        values to follow the same distribution as a reference dataset.
        Unlike grid-point methods that match values at each location, this approach uses
        all data across the spatial domain to build the statistical distributions.

        This is particularly useful when forecasts have been smoothed and you want to
        restore realistic variation in the values while preserving the spatial patterns.

        Uses the discrete (floor) method for quantile lookup, which maps each quantile
        to the nearest available reference value, creating a step-function mapping.

        Args:
            reference_cube:
                The reference data that define what the "correct" distribution
                should look like.
            forecast_cube:
                The forecast data you want to correct (e.g. smoothed model output).

        Returns:
            Calibrated forecast cube with quantiles mapped to the reference
            distribution.

        Note:
            The output mask is the union of the reference and forecast masks. Output
            will be masked at any location where EITHER input is masked, as quantile
            mapping requires valid data from both sources. This may result in the
            output having more masked values than the forecast input.
        """

        # Ensure both cubes use the same units
        reference_cube_same_units, forecast_cube = (
            self._convert_reference_cube_to_forecast_units(
                reference_cube, forecast_cube
            )
        )

        # Create output cube to preserve metadata
        output_cube = forecast_cube.copy()

        # Apply quantile mapping (handles masked data automatically)
        corrected_values_flat = self._process_masked_data(
            reference_cube_same_units, forecast_cube
        )

        self._finalise_output_cube(corrected_values_flat, forecast_cube, output_cube)

        return output_cube
