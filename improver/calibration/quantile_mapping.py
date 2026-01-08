# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing quantile mapping bias correction.

Quantile mapping is a statistical calibration technique that adjusts forecast
values to match the distribution of reference (observed) data. It works by:
1. Finding each forecast value's position (quantile) in the forecast distribution
2. Mapping that quantile to the corresponding value in the reference distribution

This corrects systematic biases while preserving spatial patterns.
"""

from typing import Optional

import numpy as np
from iris.cube import Cube

from improver import PostProcessingPlugin


class QuantileMapping(PostProcessingPlugin):
    """Apply quantile mapping bias correction to forecast data."""

    def __init__(self, preservation_threshold: Optional[float] = None) -> None:
        """Initialize the quantile mapping plugin.

        Args:
            preservation_threshold:
                Optional threshold value below which (exclusive) the forecast
                values are not adjusted to be like the reference. Useful for variables
                such as precipitation, where a user may be wary of mapping 0mm/hr
                precipitation values to non-zero values.
        """
        self.preservation_threshold = preservation_threshold

    @staticmethod
    def _build_empirical_cdf(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build empirical cumulative distribution function (CDF).

        Args:
            data: 1D array of input data values.

        Returns:
            Tuple of (sorted_values, quantiles) representing the empirical CDF.

        """
        sorted_values = np.sort(data)
        num_points = sorted_values.shape[0]
        quantiles = np.arange(1, num_points + 1) / num_points
        return sorted_values, quantiles

    @staticmethod
    def _inverted_cdf(data: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
        """Get distribution values at specified quantiles (discrete step method).

        Uses floored index lookup, rounding each quantile down to the nearest
        available data point. This creates a step-function mapping that's faster
        but less smooth than interpolation.

        Taken from:
        https://github.com/ecmwf-projects/ibicus/blob/main/ibicus/utils/_math_utils.py

        Args:
            data:
                1D array of data values defining the distribution.
            quantiles:
                Quantiles to evaluate (values between 0 and 1).

        Returns:
            Values from the data corresponding to the requested quantiles.
        """
        sorted_values = np.sort(data)
        num_points = sorted_values.shape[0]
        floored_indices = np.array(
            np.floor((num_points - 1) * quantiles), dtype=np.int32
        )
        return sorted_values[floored_indices]

    def _map_quantiles(
        self,
        reference_data: np.ndarray,
        forecast_data: np.ndarray,
    ) -> np.ndarray:
        """Transform forecast values to match the reference distribution.

        For each forecast value:

        1. Find its quantile position in the forecast distribution
        2. Map that quantile to the corresponding value in the reference distribution
           using discrete (floor) method

        For example, if reference_data is [10, 20, 30, 40, 50] and forecast_data
        is [5, 15, 25, 35, 45], the forecast systematically underestimates by 5 units.
        The corrected values will be [10, 20, 30, 40, 50], mapped to match the
        reference distribution.

        Args:
            reference_data:
                Target distribution (observed/historical data).
            forecast_data:
                Source distribution (biased forecasts to correct).

        Returns:
            Bias-corrected forecast values matching the reference distribution.
        """
        # Build empirical CDF for the forecast distribution
        sorted_forecast_values, forecast_empirical_quantiles = (
            self._build_empirical_cdf(forecast_data)
        )

        # Find where each forecast value sits in the forecast distribution
        # (i.e., determine its quantile, clipped to [0, 1])
        forecast_quantiles = np.interp(
            forecast_data, sorted_forecast_values, forecast_empirical_quantiles
        )

        # Map the quantiles to values in the reference distribution
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

        # Convert reference_cube to target_units if needed
        if reference_cube.units != target_units:
            try:
                reference_cube = reference_cube.copy()
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
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
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
            Tuple of:
                - corrected_data_flat: 1D array with corrected values.
                - output_mask: The mask to apply, or None if data is not masked.
        """
        # Determine if either cube has masked data
        forecast_is_masked = np.ma.is_masked(forecast_cube.data)
        reference_is_masked = np.ma.is_masked(reference_cube.data)

        if forecast_is_masked or reference_is_masked:
            # Create combined mask using getmaskarray (returns False array if not masked)
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

            output_mask = combined_mask
        else:
            # No masking needed
            output_mask = None
            corrected_values_flat = self._map_quantiles(
                reference_cube.data.flatten(),
                forecast_cube.data.flatten(),
            )

        return corrected_values_flat, output_mask

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
        self,
        corrected_values_flat: np.ndarray,
        forecast_cube: Cube,
        output_cube: Cube,
        output_mask,
    ) -> None:
        """Make final adjustments to output cube metadata and data type.
        Args:
            output_cube:
                The cube to finalize.
        """
        # Reshape corrected data to match original shape and set data type to float32
        if corrected_values_flat.dtype != np.float32:
            corrected_values_flat = corrected_values_flat.astype(np.float32)

        corrected_data_reshaped = np.reshape(corrected_values_flat, forecast_cube.shape)

        # Reinstate original mask if applicable
        if output_mask is not None:
            output_cube.data = np.ma.masked_array(
                corrected_data_reshaped, mask=output_mask
            )
        else:
            output_cube.data = corrected_data_reshaped

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
        reference_cube, forecast_cube = self._convert_reference_cube_to_forecast_units(
            reference_cube, forecast_cube
        )

        # Create output cube to preserve metadata
        output_cube = forecast_cube.copy()

        # Apply quantile mapping (handles masked data automatically)
        corrected_values_flat, output_mask = self._process_masked_data(
            reference_cube, forecast_cube
        )

        self._finalise_output_cube(
            corrected_values_flat, forecast_cube, output_cube, output_mask
        )

        return output_cube
