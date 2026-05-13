# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugin for adding stochastic noise to a cube using Short-Space Fourier Transform
(SSFT).
"""

import warnings
from typing import Optional

import numpy as np
from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.utilities.cube_checker import validate_cube_dimensions


class StochasticNoise(BasePlugin):
    """Class to apply spatially-structured stochastic noise to non-positive regions of a
    field, building on the Short-Space Fourier Transform (SSFT) approach from
    Nerini et al. (2017).

    This plugin is intended for use with positive zero-bounded diagnostics only, and is
    a particularly useful tool for Ensemble Copula Coupling-Quantile (ECC-Q) realization
    generation. While ECC-Q is used to improve the accuracy of forecasts by calibrating
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
        ssft_init_params: Optional[dict] = None,
        ssft_generate_params: Optional[dict] = None,
        db_threshold: float = 0.03,
        db_threshold_units: str = "mm/hr",
        scale_non_positive_noise: bool = False,
        allow_seeded_parallel_processing: bool = False,
        arbitrary_offset: float = 5.0,
    ):
        """
        Initialise the plugin.

        If ssft_init_params or ssft_generate_params are not provided, default values
        from the Pysteps documentation will be used.

        Args:
            ssft_init_params:
                Keyword arguments for initializing SSFT filter using
                pysteps.noise.fftgenerators.initialize_nonparam_2d_ssft_filter.
                Default is an empty dict, which will use the pysteps defaults.
            ssft_generate_params:
                Keyword arguments for generating stochastic noise using
                pysteps.noise.fftgenerators.generate_noise_2d_ssft_filter.
                Default is an empty dict, which will use the pysteps defaults.
            db_threshold:
                Threshold value below which data will be set to a constant in dB scale
                to avoid issues with log(0). Value provided in units of
                `db_threshold_units`.
                Default is 0.03 mm/hr.
            db_threshold_units:
                Units of the db_threshold value.
                Default is "mm/hr".
            scale_non_positive_noise:
                If True, noise in non-positive regions (where template.data <= 0) will
                be scaled such that the maximum noise value in those regions is zero and
                all other noise values are negative. This prevents the addition of
                positive noise to non-positive regions, which could artificially
                increase values where the input cube indicates no signal should occur.
                Default is False.
            allow_seeded_parallel_processing:
                If True, allows multiple workers to be used even when a seed is
                provided in ssft_generate_params. This may improve computation speed,
                but can introduce run-to-run variation because pySTEPS uses global RNG
                seeding. If False, seeded runs are forced to a single worker for
                reproducibility. Default is False.
            arbitrary_offset:
                An arbitrary offset value to add to the dB values of sub-threshold
                pixels. This is used to ensure that all sub-threshold pixels have a
                distinct value in dB space, which allows them to be handled
                appropriately in the _from_dB method. The default value of 5 was chosen
                to provide a clear separation from the threshold value in dB space, but
                can be adjusted if needed.

        Raises:
            ValueError:
                If db_threshold is not a positive value.

        Warnings:
            If a seed is provided in ssft_generate_params and
            allow_seeded_parallel_processing is True, a warning is raised to indicate
            that using multiple workers with a fixed seed may introduce run-to-run
            variation because pySTEPS uses global RNG seeding.

        Example dictionaries for initializing and generating SSFT filter::

            ssft_init_params = {"win_size": (100, 100), "overlap": 0.3, "war_thr": 0.1}
            ssft_generate_params = {"overlap": 0.3, "seed": 0}

        See Pysteps documentation for further keyword arguments.
        """
        if db_threshold <= 0:
            raise ValueError("db_threshold must be a positive value.")

        self.ssft_init_params = ssft_init_params or {}
        self.ssft_generate_params = ssft_generate_params or {}
        self.db_threshold = db_threshold
        self.db_threshold_units = db_threshold_units
        self.scale_non_positive_noise = scale_non_positive_noise
        self.allow_seeded_parallel_processing = allow_seeded_parallel_processing
        self.arbitrary_offset = arbitrary_offset

        if (
            "seed" in self.ssft_generate_params
        ) and self.allow_seeded_parallel_processing:
            warnings.warn(
                "Using multiple workers with a fixed seed may introduce run-to-run "
                "variation because pySTEPS uses global RNG seeding. Set "
                "allow_seeded_parallel_processing to False for reproducibility.",
                UserWarning,
            )

    def _process_single_realization(self, input_cube: Cube) -> Cube:
        """Process a cube containing a single realization (or no realization coord).

        Args:
            input_cube:
                Cube to which stochastic noise will be added.

        Returns:
            Cube with added stochastic noise.
        """
        validate_cube_dimensions(
            cube=input_cube,
            required_dimensions=["x", "y"],
            exact_match=False,
        )

        # Store original cube units and mask
        original_units = input_cube.units
        original_mask = None
        if np.ma.isMaskedArray(input_cube.data):
            original_mask = input_cube.data.mask.copy()

        # Convert to db_threshold_units for processing
        template = input_cube.copy()
        template.convert_units(self.db_threshold_units)

        # Fill masked values with 0 for processing
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

        # Compute SSFT noise
        result = self.do_fft(template_dB.data)

        # Convert generated noise from dB to linear scale
        noise_linear = self._from_dB(data=result).astype(np.float32, copy=False)

        # If requested, scale noise in non-positive regions to prevent increasing values
        # where there should be no signal
        if self.scale_non_positive_noise:
            max_noise_non_positiveregions = np.max(noise_linear[non_positive_mask])
            noise_linear[non_positive_mask] = (
                noise_linear[non_positive_mask] - max_noise_non_positiveregions
            )

        # Add noise only to non-positive regions, leave positive regions unchanged
        output_cube = template.copy()
        output_cube.data[non_positive_mask] = (
            template.data[non_positive_mask] + noise_linear[non_positive_mask]
        )

        # Restore original mask
        if original_mask is not None:
            output_cube.data = np.ma.masked_array(output_cube.data, mask=original_mask)

        # Convert back to original units
        output_cube.convert_units(original_units)

        return output_cube

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
        cube.data[mask] = (
            threshold_dB - self.arbitrary_offset
        )  # Offset sub-threshold values
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

        While this plugin accepts any cube with "x" and "y" dimensions, it is
        recommended to first slice the cube over the realization dimension and
        parallelize the processing of individual realizations using the plugin on each
        slice, to improve performance. This extraction and later merging of realization
        slices can be easily achieved using the improver CLI `extract` and
        `merge` functionality, respectively.

        Args:
            input_cube:
                Cube to which stochastic noise will be added. Must contain "x" and "y"
                dimensions, and may optionally contain a "realization" dimension.
        Returns:
            Cube with added stochastic noise.
        Warnings:
                If the input cube contains a "realization" dimension, a warning is
                raised to indicate that processing will be slower than necessary, and
                that it is recommended to process each realization separately.
        """
        # Check if input_cube has a realization dimension. If so, process each
        # realization slice separately and merge results.
        # If not, process the cube directly.
        realization_dim_coords = input_cube.coords("realization", dim_coords=True)
        if not realization_dim_coords:
            return self._process_single_realization(input_cube)

        warnings.warn(
            "Input cube has a multi-realization dimension. For best performance, "
            "prefer passing single-realization cubes and processing "
            "each realization separately. Processing will continue by iterating over "
            "realization slices.",
            UserWarning,
        )

        output_slices = CubeList(
            self._process_single_realization(slc)
            for slc in input_cube.slices_over("realization")
        )
        return output_slices.merge_cube()
