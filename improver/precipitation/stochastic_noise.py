# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Plugin for adding stochastic noise to a dependence template using Short-Space Fourier
Transform (SSFT).
"""

import os

import cf_units
import numpy as np
from dask import compute, delayed
from iris.cube import Cube
from pysteps.noise.fftgenerators import (
    generate_noise_2d_ssft_filter,
    initialize_nonparam_2d_ssft_filter,
)

from improver import BasePlugin


class StochasticNoise(BasePlugin):
    """Class to add stochastic noise to a dependence template cube object using SSFT."""

    def __init__(
        self,
        ssft_init_params: dict = None,
        ssft_generate_params: dict = None,
        threshold: float = 0.03,
        threshold_units: str = "mm/hr",
    ):
        """
        Initialise the plugin.

        If ssft_init_params or ssft_generate_params are not provided, default values
        will be used.

        Args:
            ssft_init_params:
                Keyword arguments for initializing SSFT filter using
                pysteps.noise.fftgenerators.initialize_nonparam_2d_ssft_filter.
                - Recommended: win_size, overlap, war_thr.
            ssft_generate_params:
                Keyword arguments for generating stochastic noise using
                pysteps.noise.fftgenerators.generate_noise_2d_ssft_filter.
                - Recommended: overlap, seed.
            threshold:
                Threshold value below which data will be set to a constant in dB scale.
                Default is 0.03 mm/hr.
            threshold_units:
                Units of the threshold value. Default is "mm/hr".

        Example dictionaries for initializing and generating SSFT filter:
            ssft_init_params={
                "win_size": (100, 100),
                "overlap": 0.3,
                "war_thr": 0.1
            }
            ssft_generate_params={
                "overlap": 0.3,
                "seed": 0
            }

        See Pysteps documentation for further keyword arguments.
        """
        if ssft_init_params is None:
            ssft_init_params = {
                "win_size": (100, 100),
                "overlap": 0.3,
                "war_thr": 0.1,
            }
        if ssft_generate_params is None:
            ssft_generate_params = {
                "overlap": 0.3,
                "seed": 0,
            }
        self.ssft_init_params = ssft_init_params
        self.ssft_generate_params = ssft_generate_params
        self.threshold = threshold
        self.threshold_units = threshold_units

    def _convert_threshold_units(self, cube: Cube) -> float:
        """
        Convert threshold to the units of the provided cube.

        Args:
            cube:
                Cube whose units will be used for conversion.
        Returns:
            float:
                Threshold value converted to the units of the cube.
        """
        if cube.units != self.threshold_units:
            try:
                threshold = cf_units.Unit(self.threshold_units).convert(
                    self.threshold, cube.units
                )
                return threshold
            except Exception:
                raise ValueError(
                    f"Cannot convert threshold from {self.threshold_units} to {cube.units}"
                )
        else:
            return self.threshold

    def _to_dB(self, cube: Cube) -> Cube:
        """Convert cube data to dB scale with thresholding.

        Function based on dB_transform function (with arg inverse=False) from
        https://github.com/pySTEPS/pysteps/blob/master/pysteps/utils/transformation.py.

        Args:
            cube:
                Cube containing data to be converted to dB scale.
        Returns:
            Cube with data converted from original scale to dB scale.
        """
        threshold = self._convert_threshold_units(cube)
        threshold_dB = 10.0 * np.log10(threshold)
        mask = cube.data < threshold
        # Ensure no zero or negative values are passed to log10
        safe_data = np.copy(cube.data[~mask])
        safe_data[safe_data <= 0] = threshold  # Replace non-positive with threshold
        cube.data[~mask] = 10.0 * np.log10(safe_data)
        cube.data[mask] = threshold_dB - 5  # Use -5 dB offset for sub-threshold values
        return cube

    def _from_dB(self, cube: Cube) -> Cube:
        """Convert cube data from dB scale with thresholding.

        Function based on dB_transform function (with arg inverse=True) from
        https://github.com/pySTEPS/pysteps/blob/master/pysteps/utils/transformation.py.

        Args:
            cube:
                Cube containing data to be converted from dB scale.
        Returns:
            Cube with data converted from dB scale to original scale.
            Note: After conversion to original scale, values below the threshold
            are set to zero.
        """
        cube.data = 10 ** (cube.data / 10.0)
        threshold = self._convert_threshold_units(cube)
        cube.data[cube.data < threshold] = 0
        return cube

    def do_fft(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """
        Function to generate stochastic noise using SSFT.

        Args:
            data (np.ndarray):
                2D/3D array for which stochastic noise is to be added.
            ssft_init_params (dict):
                Dictionary of parameters for initializing SSFT filter.
            ssft_generate_params (dict):
                Dictionary of parameters for generating stochastic noise using SSFT.
                - Recommended: overlap, seed.
        Returns:
            np.ndarray:
                2D/3D array of generated stochastic noise.
        """
        nonparametric_filter = initialize_nonparam_2d_ssft_filter(
            data,
            **self.ssft_init_params,
        )
        stochastic_noise = generate_noise_2d_ssft_filter(
            nonparametric_filter, **self.ssft_generate_params
        )

        # Handle possible NaN output from generate_noise_2d_ssft_filter
        if np.any(np.isnan(stochastic_noise)):
            stochastic_noise = np.nan_to_num(stochastic_noise, nan=0.0)
        return stochastic_noise

    def stochastic_noise_to_dependence_template(
        self,
        dependence_template: Cube,
    ) -> Cube:
        """
        Add stochastic noise to a dependence template cube object using SSFT.

        Args:
            dependence_template:
                Cube to which stochastic noise will be added.
        Returns:
            Original dependence template cube with added stochastic noise.
        """
        template = dependence_template.copy()
        template_dB = self._to_dB(template.copy())
        noise_dB = template.copy()

        realization_coord = template.coord("realization")
        delay_results = []
        for k in range(len(realization_coord.points)):
            realization_data = template_dB.data[k]
            # If all values are masked, skip FFT
            if np.ma.isMaskedArray(realization_data):
                if np.all(realization_data.mask):
                    delay_results.append(
                        delayed(np.zeros_like)(realization_data, dtype=np.float32)
                    )
                    continue
                realization_data = np.ma.filled(realization_data, 0).astype(np.float32)
                # Check for constant arrays
                # (e.g. log-transformed zeros)
                delay_results.append(
                    delayed(self.do_fft)(realization_data.astype(np.float32))
                )

        num_workers = min(len(template.coord("realization").points), os.cpu_count() - 1)

        results = compute(*delay_results, scheduler="threads", num_workers=num_workers)

        for k, result in enumerate(results):
            # Ensure result is float32 for consistency
            noise_dB.data[k] = np.array(result, dtype=np.float32)

        noise = self._from_dB(noise_dB.copy())

        # Set noise to zero where dependence template data is below threshold
        # to prevent spurious wet pixels
        dry_mask = template.data < self._convert_threshold_units(template)
        noise.data[dry_mask] = 0

        # Add noise to dependence template
        output_cube = template.copy()
        output_cube.data = (template.data + noise.data).astype(np.float32)

        return output_cube

    def process(self, dependence_template: Cube) -> Cube:
        """
        Add stochastic noise to a dependence template cube object using SSFT.

        Args:
            dependence_template:
                Cube to which stochastic noise will be added.
        Returns:
            Original dependence template cube with added stochastic noise.
        """
        output = self.stochastic_noise_to_dependence_template(dependence_template)
        return output
