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

import warnings
from typing import Literal, Optional, Tuple

import numpy as np
from iris.cube import Cube

from improver import PostProcessingPlugin


class QuantileMapping(PostProcessingPlugin):
    """Apply quantile mapping bias correction to forecast data."""

    def __init__(
        self,
        preservation_threshold: Optional[float] = None,
        occurrence_threshold: Optional[float] = None,
        non_occurrence_value: float = 0.0,
        method: Literal["step", "continuous"] = "step",
    ) -> None:
        """Initialize the quantile mapping plugin.

        Args:
            preservation_threshold:
                Optional threshold value below which (exclusive) reference
                values are preserved in the output. The threshold mask is
                computed from the reference cube (not the forecast cube),
                which is useful for variables such as precipitation where a
                user may be wary of converting originally 0 mm/hr or small
                reference values into non-zero values.
            occurrence_threshold:
                Optional threshold that enables occurrence correction before
                quantile mapping. When supplied, forecast pixels are first
                thinned so that the proportion of "occurring" forecast
                pixels (values strictly above this threshold) matches the
                occurrence fraction of the reference field. The lowest-
                intensity surplus occurring forecast pixels are set to
                non_occurrence_value before quantile mapping is applied only
                to the remaining occurring pixels against the occurring
                reference values. This is suited to one-sided threshold
                variables where the field has a clear non-occurrence state
                (for example precipitation occurrence). For precipitation,
                this can reduce weak-intensity "drizzle halos" where the
                forecast has an unrealistically broad wet area.
            non_occurrence_value:
                Value used to represent non-occurrence when
                occurrence_threshold is set. This should be at or below
                occurrence_threshold. Default is 0.0.
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
                probability is less than or equal to the given probability),
                yielding:

                    [10, 10, 10, 10, 10, 10, 10, 10, 20, 40, 50]

                ---- Continuous method ----

                1. The forecast data are ranked using a stable sorting algorithm
                (np.argsort with kind='mergesort'), which assigns ranks in order of
                appearance for repeated values:

                    Ranks:
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

                2. Midpoint quantiles are assigned using (rank - 0.5) / n:

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
            ValueError: If occurrence_threshold is not finite when set.
            ValueError: If non_occurrence_value is not finite when occurrence_threshold
                is set.
            ValueError: If non_occurrence_value is greater than occurrence_threshold.
            ValueError: If an unsupported method is specified.

        """
        self.preservation_threshold = preservation_threshold
        self.occurrence_threshold = occurrence_threshold
        self.non_occurrence_value = non_occurrence_value

        if self.occurrence_threshold is not None:
            if not np.isfinite(self.occurrence_threshold):
                raise ValueError("occurrence_threshold must be finite.")
            if not np.isfinite(self.non_occurrence_value):
                raise ValueError("non_occurrence_value must be finite.")
            if self.non_occurrence_value > self.occurrence_threshold:
                raise ValueError(
                    "non_occurrence_value must be less than or equal to "
                    "occurrence_threshold."
                )

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
            ranks[order] = np.arange(1, num_points + 1)

            # Convert ranks to midpoint quantiles
            return (ranks - 0.5) / num_points

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
            quantiles_reference = self._plotting_positions(num_points)
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
                Target distribution (observations or a differently processed forecast).
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

    def _calibrate(
        self,
        reference_data: np.ndarray,
        forecast_data: np.ndarray,
    ) -> np.ndarray:
        """Dispatch to the appropriate calibration strategy.

        When occurrence_threshold is set, applies occurrence correction before
        quantile mapping (see :meth:`_apply_occurrence_correction`).
        Otherwise, applies standard quantile mapping directly
        (see :meth:`_map_quantiles`).

        Args:
            reference_data: Array of valid (non-masked) reference values.
            forecast_data: Array of valid (non-masked) forecast values.

        Returns:
            Array of calibrated forecast values.
        """
        if self.occurrence_threshold is not None:
            return self._apply_occurrence_correction(reference_data, forecast_data)
        return self._map_quantiles(reference_data, forecast_data)

    def _apply_occurrence_correction(
        self,
        reference_data: np.ndarray,
        forecast_data: np.ndarray,
    ) -> np.ndarray:
        """Match forecast occurrence fraction to reference, then quantile-map
        occurring pixels.

        This pre-processing step is applied before quantile mapping when
        occurrence_threshold is set. It is intended for one-sided threshold
        variables where values above the threshold indicate occurrence and
        non-occurrence is represented by non_occurrence_value.

        A common example is precipitation, where this can suppress excessive
        areas of weak rain ("drizzle halos") by matching the forecast wet-area
        fraction to the reference before correcting intensities.

        Algorithm:

          1. Compute the fraction of occurring pixels in the reference field (values
            strictly above occurrence_threshold).
          2. Find the forecast value corresponding to the same exceedance fraction. This
            is the cutoff below which surplus occurring forecast pixels will be set
            to non_occurrence_value.
          3. Set all forecast pixels at or below this cutoff to non_occurrence_value.
            This step only represents a change for pixels not already equal to
            non_occurrence_value.
          4. Apply quantile mapping using only the occurring pixels from each field
            (those still strictly above occurrence_threshold after step 3) using the
            _map_quantiles method.
          5. Return an array with non_occurrence_value for non-occurring pixels and
              corrected values for occurring pixels.

        Args:
            reference_data:
                Array of valid (non-masked) reference values.
            forecast_data:
                Array of valid (non-masked) forecast values.

        Returns:
            Array of the same size as forecast_data with corrected values for
            occurring pixels and non_occurrence_value for non-occurring pixels.

        Warns:
            UserWarning: If the reference occurrence fraction is exactly 0 or 1,
                indicating a degenerate case (all or no occurrence in the reference
                data) where occurrence correction is likely uninformative.
        """
        # Step 1: occurrence fraction of reference field
        reference_occurrence_fraction = np.mean(
            reference_data > self.occurrence_threshold
        )

        if reference_occurrence_fraction in (0.0, 1.0):
            warnings.warn(
                "Reference occurrence fraction is exactly 0 or 1. "
                "Occurrence correction is degenerate for this situation (all or no "
                "occurrence in the reference data) and may indicate an uninformative "
                "occurrence_threshold choice.",
                UserWarning,
            )

        # Step 2: forecast value at the corresponding exceedance fraction
        forecast_cutoff = np.quantile(
            forecast_data, 1.0 - reference_occurrence_fraction
        )

        # Step 3: suppress surplus occurring forecast pixels
        forecast_data[forecast_data <= forecast_cutoff] = self.non_occurrence_value

        # Step 4: quantile map occurring pixels only
        result = np.full(forecast_data.size, self.non_occurrence_value, dtype=float)
        forecast_occurrence_mask = forecast_data > self.occurrence_threshold
        reference_occurrence_mask = reference_data > self.occurrence_threshold

        forecast_occurrence = forecast_data[forecast_occurrence_mask]
        reference_occurrence = reference_data[reference_occurrence_mask]

        if forecast_occurrence.size > 0 and reference_occurrence.size > 0:
            result[forecast_occurrence_mask] = self._map_quantiles(
                reference_occurrence, forecast_occurrence
            )

        return result

    def _process_masked_data(
        self,
        reference_cube: Cube,
        forecast_cube: Cube,
    ) -> np.ndarray:
        """Apply quantile mapping while properly handling masked data.

        Masked values in reference_cube are excluded from CDF calculations.
        Forecast mask is preserved in output to avoid filling intentionally masked
        locations.

        Args:
            reference_cube:
                The reference cube (with units already converted).
            forecast_cube:
                The forecast cube to calibrate.

        Returns:
            corrected_data_flat:
                1D array with corrected values (forecast mask preserved).
        """
        # Flatten data
        reference_data_flat = reference_cube.data.flatten()
        forecast_data_flat = forecast_cube.data.flatten()

        # Extract valid (non-masked) data for quantile mapping
        if np.ma.is_masked(reference_data_flat):
            reference_valid = reference_data_flat.compressed()
        else:
            reference_valid = reference_data_flat

        if np.ma.is_masked(forecast_data_flat):
            forecast_mask = np.ma.getmaskarray(forecast_data_flat)
            forecast_valid = forecast_data_flat.compressed()

            # Apply quantile mapping to valid data only
            corrected_valid = self._calibrate(reference_valid, forecast_valid)

            # Reconstruct with forecast mask preserved
            corrected_values_flat = np.empty_like(forecast_data_flat.data)
            corrected_values_flat[~forecast_mask] = corrected_valid
            corrected_values_flat = np.ma.masked_array(
                corrected_values_flat, mask=forecast_mask
            )
        else:
            # No forecast masking - apply directly
            corrected_values_flat = self._calibrate(reference_valid, forecast_data_flat)

        return corrected_values_flat

    def _apply_preservation_threshold(
        self, output_cube: Cube, reference_cube: Cube
    ) -> None:
        """Preserve reference values below preservation threshold.

        The threshold comparison is applied to reference_cube.data. Where
        reference values are below the threshold, output values are replaced
        with reference values. This helps prevent expansion of low/zero-valued
        regions (for example non-zero precipitation area) caused by quantile
        mapping.

        Modifies output_cube.data in-place.

        Args:
            output_cube:
                The cube with calibrated data to modify.
            reference_cube:
                The original reference cube with values to preserve.
        """
        if self.preservation_threshold is None:
            return

        mask_below_threshold = np.ma.less(
            reference_cube.data, self.preservation_threshold
        )
        # np.ma.where works for both masked and non-masked arrays
        output_cube.data = np.ma.where(
            mask_below_threshold, reference_cube.data, output_cube.data
        )

    def _finalise_output_cube(
        self, corrected_values_flat: np.ndarray, reference_cube: Cube, output_cube: Cube
    ) -> None:
        """Make final adjustments to output cube metadata and data type.
        Args:
            corrected_values_flat:
                1D array of corrected values to reshape and insert into output cube.
            reference_cube:
                The original reference cube, used to determine the shape and for
                preservation threshold.
            output_cube:
                The cube to finalize.
            output_mask:
                The mask to apply to the output cube, or None if no masking is needed.
        """
        # Reshape corrected data to match original shape and set data type to float32
        if corrected_values_flat.dtype != np.float32:
            corrected_values_flat = corrected_values_flat.astype(np.float32)

        output_cube.data = np.reshape(corrected_values_flat, reference_cube.shape)

        # Preserve low values if threshold is set, modifying in-place
        self._apply_preservation_threshold(output_cube, reference_cube)

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

        self._finalise_output_cube(
            corrected_values_flat, reference_cube_same_units, output_cube
        )
        return output_cube
