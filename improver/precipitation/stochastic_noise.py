# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugin for adding stochastic noise to a cube using Short-Space Fourier Transform
(SSFT).
"""

import os
from typing import Optional

import cf_units
import numpy as np
from dask import compute, delayed
from iris.cube import Cube

from improver import BasePlugin


class StochasticNoise(BasePlugin):
    """Class to add stochastic noise to a cube object using Short-Space Fourier
    Transform (SSFT).

    The SSFT approach from Nerini et al. (2017) generates stochastic noise with
    realistic spatial structure, where noise values at neighbouring grid points are
    correlated rather than independent. This is particularly useful for Ensemble Copula
    Coupling-Quantile (ECC-Q) realization generation, where post-processing may indicate
    non-zero precipitation should occur at locations where all raw ensemble members had
    zero. In ECC reordering, these locations create ties (all raw members have identical
    zero values) that cannot be meaningfully reordered. The spatially-structured noise
    breaks these ties by adding small contiguous precipitation patches in dry regions,
    avoiding unrealistic single-pixel artifacts while respecting the calibrated forecast
    probabilities.
    """

    def __init__(
        self,
        ssft_init_params: dict = {},
        ssft_generate_params: dict = {},
        db_threshold: float = 0.03,
        db_threshold_units: str = "mm/hr",
        num_workers: Optional[int] = len(os.sched_getaffinity(0)),
        scale_dry_noise: bool = False,
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
                to avoid issues with log(0).
                Default is 0.03 mm/hr.
            db_threshold_units:
                Units of the db_threshold value. Default is "mm/hr".
            num_workers:
                Number of worker threads for parallel FFT computation.
                If not specified, uses the smaller of the plugin's default (number of
                available CPUs) or the number of realizations in the input cube.
            scale_dry_noise:
                If True, noise in dry regions (where template.data <= 0) will be scaled
                such that the maximum noise value in those regions is zero and all other
                noise values are negative.
                This prevents the addition of positive noise to dry regions, which could
                artificially increase precipitation values where the input cube
                indicates no precipitation should occur.
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
        self.scale_dry_noise = scale_dry_noise

    def _convert_threshold_units(self, cube: Cube) -> float:
        """
        Convert db_threshold to the units of the provided cube.

        Args:
            cube:
                Cube whose units will be used for conversion.
        Returns:
            float:
                db_threshold value converted to the units of the cube.
        """
        if str(cube.units) != self.db_threshold_units:
            threshold = cf_units.Unit(self.db_threshold_units).convert(
                self.db_threshold, cube.units
            )
            return threshold
        else:
            return self.db_threshold

    def _to_dB(self, cube: Cube) -> Cube:
        """Convert cube data to dB scale with thresholding.

        Function based on dB_transform function (with arg inverse=False) from
        https://github.com/pySTEPS/pysteps/blob/master/pysteps/utils/transformation.py.

        Args:
            cube:
                Cube containing data to be converted to dB scale.
        Returns:
            Cube with data converted from linear scale to dB scale.
        """
        threshold = self._convert_threshold_units(cube)
        threshold_dB = 10.0 * np.log10(threshold)
        mask = cube.data < threshold
        cube.data[~mask] = 10.0 * np.log10(cube.data[~mask])
        # The below offsets sub-threshold values. The choice to subtract 5 is arbitrary,
        # and ensures masked values have a distinct value, which is later handled in
        # _from_dB by setting values below the threshold to zero.
        cube.data[mask] = threshold_dB - 5  # Offset sub-threshold values
        return cube

    def _from_dB(self, data: np.ndarray, units_cube: Cube) -> np.ndarray:
        """Convert cube data from dB scale with thresholding.

        Function based on dB_transform function (with arg inverse=True) from
        https://github.com/pySTEPS/pysteps/blob/master/pysteps/utils/transformation.py.

        Args:
            data:
                data in dB scale.
            units_cube:
                Cube whose units will be used for threshold conversion.
        Returns:
            np.ndarray with data converted from dB scale to original scale.
            Note: After conversion to original scale, values below the threshold
            are set to zero.
        """
        linear = 10 ** (data / 10.0)
        db_threshold = self._convert_threshold_units(units_cube)
        linear[linear < db_threshold] = 0.0
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

    def stochastic_noise_to_input_cube(
        self,
        input_cube: Cube,
    ) -> Cube:
        """
        Add locally-conditioned stochastic noise to a cube using SSFT.

        Args:
            input_cube:
                Cube to which stochastic noise will be added.
        Returns:
            Cube with added stochastic noise.
        """
        # Store original precipitation units and mask
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

        # Identify dry regions (no precipitation) where noise should be added
        dry_mask = template.data <= 0

        # If no dry values, return input unchanged (output would be unchanged with SSFT
        # noise addition only to dry regions)
        if not np.any(dry_mask):
            output_cube = input_cube.copy()
            return output_cube

        # Create a copy of the template in dB scale to use for SSFT processing
        template_dB = self._to_dB(template.copy())

        # Build delayed processing tasks for each realization
        n_realiz = template.coord("realization").points.size
        tasks = []
        for k in range(n_realiz):
            realiz_data = template_dB.data[k].astype(np.float32)
            tasks.append(delayed(self.do_fft)(realiz_data))

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
            lin_noise = self._from_dB(data=result_db, units_cube=template)
            noise_linear.data[k] = lin_noise

        # If requested, scale noise in dry regions to prevent increasing values where
        # there should be no precipitation
        if self.scale_dry_noise:
            max_noise_dry_regions = np.max(noise_linear.data[dry_mask])
            noise_linear.data[dry_mask] = (
                noise_linear.data[dry_mask] - max_noise_dry_regions
            )

        # Add noise only to dry regions (zero precipitation), leave wet regions
        # unchanged
        output_cube = template.copy()
        output_cube.data[dry_mask] = (
            template.data[dry_mask] + noise_linear.data[dry_mask]
        )

        # Restore original mask
        if original_mask is not None:
            output_cube.data = np.ma.masked_array(output_cube.data, mask=original_mask)

        # Convert back to original units
        output_cube.convert_units(original_units)

        return output_cube

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
        output = self.stochastic_noise_to_input_cube(input_cube)
        return output
