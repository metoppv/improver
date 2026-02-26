# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugin for adding stochastic noise to a cube using Short-Space Fourier Transform
(SSFT).
"""

import os
from typing import Optional

import numpy as np
from dask import compute, delayed
from iris.cube import Cube

from improver import BasePlugin


class StochasticNoise(BasePlugin):
    """Class to apply spatially-structured stochastic noise to non-positive regions of a
    field, building on the Short-Space Fourier Transform (SSFT) approach from
    Nerini et al. (2017).

    This plugin is intended for use with positive zero-bounded diagnostics only, and is
    a particularly useful tool for Ensemble Copula Coupling-Quantile (ECC-Q) realization
    generation. While EEC-Q is used to improve the accuracy of forecasts by calibrating
    ensemble members to better represent the true distribution of the forecast variable,
    the rank-based reordering (sorting) of ensemble members at each grid point can lead
    to unrealistic individual members (e.g. single-pixel precipitation artifacts) when
    multiple raw ensemble members have identical values ('ties') of zero (very common
    in precipitation forecasts) and the post-processed calibrated probabilities
    indicate a non-zero value should occur. By adding spatially-structured noise to
    break ties in these non-positive regions, more realistic spatial structures can be
    generated in the final ECC-Q realizations, while still respecting the calibrated
    probabilities.
    """

    def __init__(
        self,
        ssft_init_params: dict = {},
        ssft_generate_params: dict = {},
        db_threshold: float = 0.03,
        db_threshold_units: str = "mm/hr",
        num_workers: Optional[int] = len(os.sched_getaffinity(0)),
        scale_non_positive_noise: bool = False,
    ):
        """
        Initialise the plugin.

        If ssft_init_params or ssft_generate_params are not provided, default values
        from the Pysteps documentation will be used.

        Args:
            ssft_init_params:
                Keyword arguments for initializing SSFT filter using
                pysteps.noise.fftgenerators.initialize_nonparam_2d_ssft_filter.
            ssft_generate_params:
                Keyword arguments for generating stochastic noise using
                pysteps.noise.fftgenerators.generate_noise_2d_ssft_filter.
            db_threshold:
                Threshold value below which data will be set to a constant in dB scale
                to avoid issues with log(0). Value provided in units of
                `db_threshold_units`.
                Default is 0.03 mm/hr.
            db_threshold_units:
                Units of the db_threshold value. Default is "mm/hr".
            num_workers:
                Number of worker threads for parallel FFT computation.
                If not specified, uses the smaller of the plugin's default (number of
                available CPUs) or the number of realizations in the input cube.
            scale_non_positive_noise:
                If True, noise in non-positive regions (where template.data <= 0) will
                be scaled such that the maximum noise value in those regions is zero and
                all other noise values are negative. This prevents the addition of
                positive noise to non-positive regions, which could artificially
                increase values where the input cube indicates no signal should occur.
                Default is False.

        Example dictionaries for initializing and generating SSFT filter::

            ssft_init_params = {"win_size": (100, 100), "overlap": 0.3, "war_thr": 0.1}
            ssft_generate_params = {"overlap": 0.3, "seed": 0}

        See Pysteps documentation for further keyword arguments.
        """
        if db_threshold <= 0:
            raise ValueError("db_threshold must be a positive value.")

        self.ssft_init_params = ssft_init_params
        self.ssft_generate_params = ssft_generate_params
        self.db_threshold = db_threshold
        self.db_threshold_units = db_threshold_units
        self.num_workers = num_workers
        self.scale_non_positive_noise = scale_non_positive_noise

    def _to_dB(self, cube: Cube) -> Cube:
        """Convert cube data to dB scale and apply thresholding using db_threshold
        specified in the plugin initialization.

        Function based on dB_transform function (with arg inverse=False) from
        https://github.com/pySTEPS/pysteps/blob/master/pysteps/utils/transformation.py.

        Args:
            cube:
                Cube containing data to be converted to dB scale.
        Returns:
            Cube with data converted from linear scale to dB scale.
        """
        threshold_dB = 10.0 * np.log10(self.db_threshold)
        mask = cube.data < self.db_threshold
        cube.data[~mask] = 10.0 * np.log10(cube.data[~mask])
        # The below offsets sub-threshold values. The choice to subtract 5 is arbitrary,
        # and ensures masked values have a distinct value, which is later handled in
        # _from_dB by setting values below the threshold to zero.
        cube.data[mask] = threshold_dB - 5  # Offset sub-threshold values
        return cube

    def _from_dB(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """Convert cube data from dB scale with thresholding.

        Function based on dB_transform function (with arg inverse=True) from
        https://github.com/pySTEPS/pysteps/blob/master/pysteps/utils/transformation.py.

        Args:
            data:
                data in dB scale.
        Returns:
            np.ndarray with data converted from dB scale to original scale.
            Note: After conversion to original scale, values below the threshold
            are set to zero.
        """
        linear = 10 ** (data / 10.0)
        linear[linear < self.db_threshold] = 0.0
        return linear

    def do_fft(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """
        Generate stochastic noise using SSFT for a 2-D array slice (one realization).

        Args:
            data:
                2D array for which stochastic noise is to be added.
        Returns:
            np.ndarray:
                2D array of generated stochastic noise.
        """
        from pysteps.noise.fftgenerators import (
            generate_noise_2d_ssft_filter,
            initialize_nonparam_2d_ssft_filter,
        )

        nonparametric_filter = initialize_nonparam_2d_ssft_filter(
            data,
            **self.ssft_init_params,
        )
        stochastic_noise = generate_noise_2d_ssft_filter(
            nonparametric_filter, **self.ssft_generate_params
        )

        return stochastic_noise

    def process(self, input_cube: Cube) -> Cube:
        """
        Add locally-conditioned stochastic noise to a cube object using Short-Space
        Fourier Transform (SSFT).

        Args:
            input_cube:
                Cube to which stochastic noise will be added.
        Returns:
            Cube with added stochastic noise.
        """
        # Store original cube units and mask
        original_units = input_cube.units
        original_mask = None
        if np.ma.isMaskedArray(input_cube.data):
            original_mask = input_cube.data.mask.copy()

        # Convert to db_threshold_units for processing
        template = input_cube.copy()
        template.convert_units(self.db_threshold_units)

        # Fill masked values with 0 for processing (dask does not support native numpy
        # masked arrays)
        if np.ma.isMaskedArray(template.data):
            template.data = np.ma.filled(template.data, 0.0).astype(np.float32)

        # Identify non-positive regions where noise should be added
        non_positive_mask = template.data <= 0

        # If no non-positive values, return input unchanged (output would be
        # unchanged with SSFT noise addition only to non-positive regions)
        if not np.any(non_positive_mask):
            return input_cube

        # Create a copy of the template in dB scale to use for SSFT processing
        template_dB = self._to_dB(template.copy())

        # Build delayed processing tasks for each realization
        tasks = []
        for slice in template_dB.slices_over("realization"):
            data_slice = slice.data.astype(np.float32)
            tasks.append(delayed(self.do_fft)(data_slice))

        # Set number of workers for parallel processing
        num_workers = min(
            self.num_workers,
            len(template.coord("realization").points),
        )

        # Compute all SSFT noise arrays (in dB scale) in parallel
        results = compute(*tasks, scheduler="threads", num_workers=num_workers)

        # Convert dB to linear scale
        noise_linear = template.copy()
        noise_linear.data = np.zeros_like(template.data, dtype=np.float32)
        for k, result_db in enumerate(results):
            lin_noise = self._from_dB(data=result_db)
            noise_linear.data[k] = lin_noise

        # If requested, scale noise in non-positive regions to prevent increasing values
        # where there should be no signal
        if self.scale_non_positive_noise:
            max_noise_non_positiveregions = np.max(noise_linear.data[non_positive_mask])
            noise_linear.data[non_positive_mask] = (
                noise_linear.data[non_positive_mask] - max_noise_non_positiveregions
            )

        # Add noise only to non-positive regions, leave positive regions
        # unchanged
        output_cube = template.copy()
        output_cube.data[non_positive_mask] = (
            template.data[non_positive_mask] + noise_linear.data[non_positive_mask]
        )

        # Restore original mask
        if original_mask is not None:
            output_cube.data = np.ma.masked_array(output_cube.data, mask=original_mask)

        # Convert back to original units
        output_cube.convert_units(original_units)

        return output_cube
