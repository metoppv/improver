# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing quantile mapping classes."""

from typing import Literal, Optional

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
            data: Input data values.

        Returns:
            Tuple of (sorted_values, quantiles) representing the empirical CDF.

        """
        sorted_values = np.sort(data)
        num_points = sorted_values.shape[0]
        quantiles = np.arange(1, num_points + 1) / num_points
        return sorted_values, quantiles

    @staticmethod
    def _inverted_cdf(data: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
        """Calculate values using discrete quantile lookup (rounding down to nearest data
        point).

        This method rounds each quantile down to the nearest available data point in the
        dataset, creating a step-function mapping. Faster but less smooth than
        interpolation. Always returns actual values from the data. Taken
        from https://github.com/ecmwf-projects/ibicus/blob/main/ibicus/utils/_math_utils.py.

        Args:
            data:
                Data values defining the distribution.
            quantiles:
                Quantiles to evaluate (between 0 and 1).

        Returns:
            Values corresponding to the requested quantiles.
        """
        sorted_values = np.sort(data)
        num_points = sorted_values.shape[0]
        floored_indices = np.array(
            np.floor((num_points - 1) * quantiles), dtype=np.int32
        )
        return sorted_values[floored_indices]

    def _interpolated_inverted_cdf(
        self, data: np.ndarray, quantiles: np.ndarray
    ) -> np.ndarray:
        """Calculate values at provided quantiles using linear interpolation.

        This method is slower but produces a continuous mapping.

        Args:
            data:
                Data values defining the distribution.
            quantiles:
                Quantiles to evaluate (between 0 and 1).

        Returns:
            Values corresponding to the requested quantiles.
        """
        sorted_values, empirical_quantiles = self._build_empirical_cdf(data)
        return np.interp(quantiles, empirical_quantiles, sorted_values)

    def apply_quantile_mapping(
        self,
        reference_data: np.ndarray,
        forecast_data: np.ndarray,
        values_to_map: Optional[np.ndarray] = None,
        mapping_method: Literal["floor", "interp"] = "floor",
    ) -> np.ndarray:
        """Apply quantile mapping to transform forecast values to match a reference
        distribution.

        Guidance on method choice
        -------------------------
        Consider the following example.
        - reference_data: [10, 20, 30, 40, 50]
        - forecast_data:  [5, 15, 25, 35, 45]
        - values_to_map:  [7.5, 17.5, 27.5, 37.5, 47.5, 60]

        The forecast data systematically underestimates the reference data by 5 units.
        The following mapped values will be produced with each approach:
        - floor:   [20, 20, 30, 40, 50, 50]
        - interp:  [12.5, 22.5, 32.5, 42.5, 50.0, 50.0]

        Args:
            reference_data:
                Target distribution (observed historical data).
            forecast_data:
                Source distribution (biased model forecasts).
            values_to_map:
                New forecast values to transform. If None, applies
                quantile-mapped transformation to forecast_data.
            mapping_method:
                mapping_method for inverse CDF calculation:
                - "floor": Use floored index lookup (discrete steps). Faster.
                - "interp": Use linear interpolation (continuous). Slower.

        Returns:
            Bias-corrected values in the reference distribution.

        Raises:
            ValueError:
                If an unknown method is provided.
        """
        if values_to_map is None:
            values_to_map = forecast_data

        if mapping_method not in ["floor", "interp"]:
            raise ValueError(
                f"Unknown mapping method: {mapping_method}. Choose 'floor' or 'interp'."
            )

        # Build empirical CDF for forecast distribution
        sorted_forecast_values, forecast_empirical_quantiles = (
            self._build_empirical_cdf(forecast_data)
        )

        # Map values to quantiles in forecast distribution (clips to [0, 1])
        target_quantiles = np.interp(
            values_to_map, sorted_forecast_values, forecast_empirical_quantiles
        )

        # Invert CDF using chosen method
        if mapping_method == "floor":
            corrected_values = self._inverted_cdf(reference_data, target_quantiles)
        elif mapping_method == "interp":
            corrected_values = self._interpolated_inverted_cdf(
                reference_data, target_quantiles
            )

        return corrected_values

    @staticmethod
    def _convert_cubes_to_forecast_units(
        reference_cube: Cube,
        forecast_cube: Cube,
        forecast_to_calibrate: Optional[Cube] = None,
    ) -> tuple[Cube, Cube, Optional[Cube]]:
        """Convert all cubes to common units without modifying originals.

        Args:
            reference_cube:
                The reference forecast cube.
            forecast_cube:
                The forecast cube to calibrate.
            forecast_to_calibrate:
                Optional different forecast cube to calibrate.

        Returns:
            Tuple of (reference_cube, forecast_cube, forecast_to_calibrate)
            all converted to common units.

        Raises:
            ValueError: If cubes have incompatible units.
        """
        target_units = (
            forecast_to_calibrate.units
            if forecast_to_calibrate is not None
            else forecast_cube.units
        )

        # Convert each cube to target_units if needed
        converted_cubes = []
        for cube in [reference_cube, forecast_cube, forecast_to_calibrate]:
            if cube is not None and cube.units != target_units:
                try:
                    cube = cube.copy()
                    cube.convert_units(target_units)
                except ValueError:
                    raise ValueError(
                        f"Cannot convert cube with units {cube.units} "
                        f"to target units {target_units}"
                    )
            converted_cubes.append(cube)

        return tuple(converted_cubes)

    def process(
        self,
        reference_cube: Cube,
        forecast_cube: Cube,
        forecast_to_calibrate: Optional[Cube] = None,
        mapping_method: Literal["floor", "interp"] = "floor",
    ) -> Cube:
        """Adjust forecast values to match the statistical distribution of reference
        data.

        This calibration method corrects biases in forecast data by transforming its
        values to follow the same distribution as a reference dataset.
        Unlike grid-point methods that match values at each location, this approach uses
        all data across the spatial domain to build the statistical distributions.

        This is particularly useful when forecasts have been smoothed and you want to
        restore realistic variation in the values while preserving the spatial patterns.

        Args:
            reference_cube:
                The reference data that define what the "correct" distribution
                should look like.
            forecast_cube:
                The forecast data you want to correct (e.g. smoothed model output).
            forecast_to_calibrate:
                Optional different forecast values to correct using the same mapping.
                If not provided, the forecast_cube data itself will be corrected.
            mapping_method:
                Method for inverse CDF calculation. Either "floor" (discrete steps,
                faster) or "interp" (linear interpolation; slower, continuous).

        Returns:
            Calibrated forecast cube with quantiles mapped to the reference distribution
            or forecast_to_calibrate data adjusted with the same learned mapping.
        """

        # Convert all cubes to common units
        reference_cube, forecast_cube, forecast_to_calibrate = (
            self._convert_cubes_to_forecast_units(
                reference_cube, forecast_cube, forecast_to_calibrate
            )
        )

        # Create a copy of the forecast_cube or forecast_to_calibrate cube to hold
        # output data and preserve metadata.
        output_cube = (
            forecast_cube.copy()
            if forecast_to_calibrate is None
            else forecast_to_calibrate.copy()
        )

        # Extract data, handling masked arrays
        if np.ma.is_masked(reference_cube.data):
            reference_data_flat = reference_cube.data.filled().flatten()
        else:
            reference_data_flat = reference_cube.data.flatten()

        if np.ma.is_masked(forecast_cube.data):
            forecast_data_flat = forecast_cube.data.filled().flatten()
        else:
            forecast_data_flat = forecast_cube.data.flatten()

        # Determine values to map and output shape
        if forecast_to_calibrate is None:
            # Use forecast_cube data
            if np.ma.is_masked(output_cube.data):
                values_to_map_flat = output_cube.data.filled().flatten()
            else:
                values_to_map_flat = output_cube.data.flatten()
            output_shape = forecast_cube.shape
            output_mask = (
                forecast_cube.data.mask if np.ma.is_masked(forecast_cube.data) else None
            )
        else:
            # Use provided cube's data
            output_cube = forecast_to_calibrate.copy()
            if np.ma.is_masked(forecast_to_calibrate.data):
                values_to_map_flat = forecast_to_calibrate.data.filled().flatten()
            else:
                values_to_map_flat = forecast_to_calibrate.data.flatten()
            output_shape = forecast_to_calibrate.shape
            output_mask = (
                forecast_to_calibrate.data.mask
                if np.ma.is_masked(forecast_to_calibrate.data)
                else None
            )

        corrected_values_flat = self.apply_quantile_mapping(
            reference_data_flat, forecast_data_flat, values_to_map_flat, mapping_method
        )

        # Reshape mapped data to original shape and ensure float32
        corrected_data_reshaped = np.reshape(corrected_values_flat, output_shape)
        if corrected_data_reshaped.dtype != np.float32:
            corrected_data_reshaped = corrected_data_reshaped.astype(np.float32)

        # Preserve mask if original data was masked
        if output_mask is not None:
            output_cube.data = np.ma.masked_array(
                corrected_data_reshaped, mask=output_mask
            )
        else:
            output_cube.data = corrected_data_reshaped

        # Preserve values below preservation_threshold if provided
        if self.preservation_threshold is not None:
            # Get the source data to preserve (forecast_cube or forecast_to_calibrate)
            original_source_data = (
                forecast_cube.data
                if forecast_to_calibrate is None
                else forecast_to_calibrate.data
            )
            mask_below_threshold = original_source_data < self.preservation_threshold
            # Update masked arrays only if input was masked
            if np.ma.is_masked(original_source_data):
                output_cube.data = np.ma.where(
                    mask_below_threshold, original_source_data, output_cube.data
                )
            else:
                output_cube.data = np.where(
                    mask_below_threshold, original_source_data, output_cube.data
                )

        return output_cube
