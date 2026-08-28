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
from improver.clustering.cluster_sources_utils import (
    get_source_for_forecast_period,
    parse_cluster_sources_attribute,
)
from improver.utilities.cube_checker import validate_cube_dimensions


class StochasticNoise(BasePlugin):
    """Class to apply spatially-structured stochastic noise (randomly generated noise
    with specific statistical properties) to a field, building on the Short-Space
    Fourier Transform (SSFT) approach from Nerini et al. (2017), as implemented in the
    pySTEPS library.

    This plugin is intended for use with positive zero-bounded diagnostics only, and is
    a particularly useful tool for Ensemble Copula Coupling-Quantile (ECC-Q) realization
    generation. While ECC-Q is used to improve the accuracy of forecasts by calibrating
    ensemble members to better represent the true distribution of the forecast variable,
    the rank-based reordering (sorting) of ensemble members at each grid point can lead
    to unrealistic individual members (e.g. single-pixel precipitation artifacts) when
    multiple raw ensemble members have identical values ('ties') of zero (very common
    in precipitation forecasts) and the post-processed calibrated probabilities
    indicate a non-zero value should occur. By adding spatially-structured stochastic
    noise to break ties in these non-positive regions, more realistic spatial structures
    can be generated in the final ECC-Q realizations, while still respecting the
    calibrated probabilities.

    Optionally, the plugin can also apply stochastic noise to positive (wet) regions
    to diversify ensemble members, for example when generating recycled realizations.
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
        wet_noise_floor: Optional[float] = None,
        dry_fallback_range: Optional[tuple] = None,
        apply_noise_to_positive_regions: bool = False,
        wet_noise_amplitude: float = 1.0,
        apply_noise_to_positive_regions_by_source: Optional[str] = None,
    ):
        """
        Initialise the plugin. For a typical input field e.g. a precipitation field
        with some positive values for precipitation spread across the domain and some
        zero values, the plugin will add stochastic noise to the zero values using
        the SSFT approach, while leaving the positive values unchanged (or adding noise
        if apply_noise_to_positive_regions is True). For fields that contain
        insufficient spatial variability to derive meaningful SSFT perturbations (for
        example completely dry, nearly dry, or otherwise near-constant fields),
        referred to here as degenerate fields, the plugin will generate fallback
        stochastic noise ("dry fallback noise") in linear space. This noise uses the
        wet_noise_floor and dry_fallback_range arguments to ensure that the fallback
        noise is strictly non-positive and does not exceed the noise added to wet
        regions.

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
                If this is true, wet_noise_floor must be set, so that totally dry fields
                do not receive noise that exceeds noise given to fields that are wet.
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
            wet_noise_floor:
                Optional lower bound for noise in non-positive regions after scaling,
                in linear units of db_threshold_units. Must be negative if set.
                This can be used to limit the magnitude of negative SSFT-derived
                wet-member noise. This value must be less than the SSFT-derived
                noise in wet regions. Any generated noise below the floor value
                will be set to the floor value, potentially resulting in more ties
                when used in conjunction with Ensemble Copula Coupling.
                Default is None (no floor).
            dry_fallback_range:
                Optional range (min_value, max_value) for dry fallback noise in
                linear units of db_threshold_units. Provide as a Python tuple string, e.g.
                "(-10.0, -5.0)". Both values must be <= 0 and (min_value < max_value).
                If wet_noise_floor is set and this is not provided, this defaults to
                (2 * wet_noise_floor, wet_noise_floor) to keep dry fallback below the
                wet floor. If wet_noise_floor is set and dry_fallback_range is provided,
                the max_value of dry_fallback_range must be <= wet_noise_floor to ensure
                separation between dry-fallback and wet noise ranges.
            apply_noise_to_positive_regions:
                If True, stochastic noise will also be applied to positive (wet) regions
                in addition to non-positive (dry) regions. This can be used to diversify
                ensemble members, for example when generating recycled realizations.
                The magnitude of noise applied to positive regions is controlled by
                wet_noise_amplitude. Default is False (noise only to non-positive regions).
            wet_noise_amplitude:
                Multiplicative scaling factor for stochastic noise applied to positive
                regions when apply_noise_to_positive_regions is True. A value of 1.0
                applies the full SSFT-generated noise; smaller values (e.g. 0.1) apply
                modest noise for subtle diversification. Has no effect if
                apply_noise_to_positive_regions is False. Default is 1.0.
            apply_noise_to_positive_regions_by_source:
                Optional comma-separated list of forecast source names (e.g.
                "gl_ens,ecgl_ens") for which wet-region noise should be applied.
                When set, overrides apply_noise_to_positive_regions flag with
                source-aware logic by querying the cube's cluster_sources attribute.
                Noise is applied to positive regions only if the current forecast
                period's source is in this list. Default is None (use
                apply_noise_to_positive_regions flag instead).

        Raises:
            ValueError:
                If db_threshold is not a positive value.
            ValueError:
                If wet_noise_floor is provided and is non-negative.
            ValueError:
                If wet_noise_floor is provided while
                scale_non_positive_noise is False.
            ValueError:
                If dry_fallback_range does not contain exactly two values.
            ValueError:
                If dry_fallback_range does not satisfy
                min_value < max_value <= 0.
            ValueError:
                If both wet_noise_floor and dry_fallback_range are provided
                and dry_fallback_range max exceeds wet_noise_floor.
            ValueError:
                If wet_noise_amplitude is not positive.

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

        if wet_noise_amplitude <= 0:
            raise ValueError("wet_noise_amplitude must be positive.")

        self.ssft_init_params = ssft_init_params or {}
        self.ssft_generate_params = ssft_generate_params or {}
        self.db_threshold = db_threshold
        self.db_threshold_units = db_threshold_units
        self.scale_non_positive_noise = scale_non_positive_noise
        self.allow_seeded_parallel_processing = allow_seeded_parallel_processing
        self.arbitrary_offset = arbitrary_offset
        self.wet_noise_floor = wet_noise_floor
        self.apply_noise_to_positive_regions = apply_noise_to_positive_regions
        self.wet_noise_amplitude = wet_noise_amplitude

        if (
            apply_noise_to_positive_regions
            and apply_noise_to_positive_regions_by_source is not None
        ):
            raise ValueError(
                "Cannot specify both apply_noise_to_positive_regions=True and "
                "apply_noise_to_positive_regions_by_source. Use one or the other."
            )

        self.apply_noise_to_positive_regions_by_source = (
            apply_noise_to_positive_regions_by_source
        )
        if self.apply_noise_to_positive_regions_by_source:
            self.target_sources = {
                s.strip().lower()
                for s in self.apply_noise_to_positive_regions_by_source.split(",")
            }
        else:
            self.target_sources = set()

        if self.wet_noise_floor is not None and self.wet_noise_floor >= 0:
            raise ValueError("wet_noise_floor must be negative if provided.")

        if self.wet_noise_floor is not None and not self.scale_non_positive_noise:
            raise ValueError(
                "scale_non_positive_noise must be True when wet_noise_floor is set, "
                "to guarantee separation between dry-fallback and wet noise ranges."
            )

        if dry_fallback_range is not None and len(dry_fallback_range) != 2:
            raise ValueError("dry_fallback_range must contain exactly two values.")

        if dry_fallback_range is None and self.wet_noise_floor is not None:
            dry_fallback_range = (2.0 * self.wet_noise_floor, self.wet_noise_floor)

        self.dry_fallback_range = dry_fallback_range
        if self.dry_fallback_range is not None:
            dry_min, dry_max = self.dry_fallback_range
            if not (dry_min < dry_max <= 0):
                raise ValueError(
                    "dry_fallback_range must satisfy min_value < max_value <= 0."
                )
            if self.wet_noise_floor is not None and dry_max > self.wet_noise_floor:
                raise ValueError(
                    "dry_fallback_range max must be <= wet_noise_floor when both are set."
                )

        if (
            "seed" in self.ssft_generate_params
        ) and self.allow_seeded_parallel_processing:
            warnings.warn(
                "Using multiple workers with a fixed seed may introduce run-to-run "
                "variation because pySTEPS uses global RNG seeding. Set "
                "allow_seeded_parallel_processing to False for reproducibility.",
                UserWarning,
            )

    def _should_apply_wet_noise_by_source(self, input_cube: Cube) -> bool:
        """Determine if wet-region noise should be applied based on forecast source.

        Queries cluster_sources attribute to find which model is active for this
        realization and forecast period. If the source matches one of the target
        sources specified in apply_noise_to_positive_regions_by_source, returns True.

        Args:
            input_cube:
                Input cube with realization and forecast_period coordinates.
                May have cluster_sources attribute.

        Returns:
            True if source matches target_sources, False otherwise.
            Returns False if cluster_sources attribute missing or malformed.
        """
        try:
            cluster_sources = parse_cluster_sources_attribute(input_cube)
            if not cluster_sources:
                return False

            # Get realization and forecast period indices
            realization_idx = int(input_cube.coord("realization").points[0])
            fp_seconds = int(input_cube.coord("forecast_period").points[0])

            source = get_source_for_forecast_period(
                cluster_sources, realization_idx, fp_seconds
            )
            return source is not None and source.lower() in self.target_sources

        except (AttributeError, KeyError, ValueError, TypeError):
            # Graceful fallback if cluster_sources missing/malformed or coords unavailable
            return False

    def _process_single_realization(self, input_cube: Cube) -> Cube:
        """Add stochastic noise to a cube containing a single realization
        (or no realization coord). For non-degenerate fields e.g. precipitation fields
        with some positive values, the plugin will add stochastic noise to the
        non-positive regions using the SSFT approach, while leaving the positive values
        unchanged (or adding noise if apply_noise_to_positive_regions is True).
        For degenerate fields (for example completely dry, nearly dry, or otherwise
        near-constant fields), fallback noise is generated in linear space.

        Args:
            input_cube:
                Cube to which stochastic noise will be added.

        Returns:
            Cube with added stochastic noise.

        Raises:
            ValueError: If a degenerate field is detected for SSFT initialisation and
                ``wet_noise_floor`` has not been configured (which means no default
                ``dry_fallback_range`` is available).

        Warns:
            UserWarning: If a degenerate field is detected for SSFT initialisation,
                or if SSFT initialisation fails for any reason, a warning is raised
                to indicate that linear fallback stochastic noise generation will be
                used instead.
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
        positive_mask = template.data > 0

        # Determine whether to apply wet-region noise based on source metadata
        apply_wet_noise = self.apply_noise_to_positive_regions
        if self.apply_noise_to_positive_regions_by_source:
            apply_wet_noise = self._should_apply_wet_noise_by_source(input_cube)

        # If no non-positive values and not applying noise to positive regions,
        # return input unchanged
        if not np.any(non_positive_mask) and not apply_wet_noise:
            return input_cube

        # Create a copy of the template in dB scale to use for SSFT processing
        template_dB = self._to_dB(template.copy())

        # Constant fields in dB space are degenerate for SSFT. In this case generate
        # fallback noise directly in linear space so it can still break ties.
        used_linear_fallback = False
        if self._is_degenerate_field(template_dB.data):
            warnings.warn(
                "Degenerate input field detected for SSFT initialization. "
                "Using linear fallback stochastic noise generation.",
                UserWarning,
            )
            noise_linear = self._fallback_noise_linear(template_dB.data.shape)
            used_linear_fallback = True
        else:
            # Compute SSFT noise; may fail if individual windows are degenerate,
            # in which case fall back to linear noise generation.
            try:
                result = self.do_fft(template_dB.data)
                # Convert generated noise from dB to linear scale
                noise_linear = self._from_dB(data=result).astype(np.float32, copy=False)
            except ValueError:
                # SSFT can fail when individual windows (not the whole field)
                # are constant-valued or in other edge cases. Fall back to linear noise
                # as a graceful degradation.
                warnings.warn(
                    "SSFT initialisation failed. "
                    "Falling back to linear stochastic noise generation.",
                    UserWarning,
                )
                noise_linear = self._fallback_noise_linear(template_dB.data.shape)
                used_linear_fallback = True

        # Guard against non-finite values from SSFT output fields.
        # Treat these as zero-noise contributions.
        if not np.all(np.isfinite(noise_linear)):
            noise_linear = np.where(np.isfinite(noise_linear), noise_linear, 0.0)

        # If requested, scale noise in non-positive regions to prevent increasing values
        # where there should be no signal
        if self.scale_non_positive_noise:
            max_noise_non_positiveregions = np.max(noise_linear[non_positive_mask])
            noise_linear[non_positive_mask] = (
                noise_linear[non_positive_mask] - max_noise_non_positiveregions
            )

        # Apply constraints to separate dry-fallback and wet-member noise ranges.
        if used_linear_fallback:
            # Only enforce dry fallback range constraints if there are non-positive
            # regions to apply them to
            if np.any(non_positive_mask):
                if self.dry_fallback_range is None:
                    raise ValueError(
                        "Degenerate input field detected but wet_noise_floor is not set. "
                        "Set wet_noise_floor to guarantee separation between dry-fallback "
                        "and wet noise ranges."
                    )
                dry_min, dry_max = self.dry_fallback_range
                dry_values = noise_linear[non_positive_mask]
                dry_vmin = np.min(dry_values)
                dry_vmax = np.max(dry_values)
                if dry_vmax > dry_vmin:
                    normalized = (dry_values - dry_vmin) / (dry_vmax - dry_vmin)
                    noise_linear[non_positive_mask] = dry_min + normalized * (
                        dry_max - dry_min
                    )
                else:
                    # Guard against zero dynamic range (all dry_values equal), where
                    # normalization would divide by zero; clamp to dry_max to keep values
                    # inside the configured dry fallback interval.
                    noise_linear[non_positive_mask] = dry_max
        elif self.scale_non_positive_noise and self.wet_noise_floor is not None:
            # Ensure scaled wet-member noise does not go below the configured
            # wet_noise_floor.
            noise_linear[non_positive_mask] = np.maximum(
                noise_linear[non_positive_mask], self.wet_noise_floor
            )

        # Add noise to selected regions
        output_cube = template.copy()

        # Always add noise to non-positive regions
        if np.any(non_positive_mask):
            output_cube.data[non_positive_mask] = (
                template.data[non_positive_mask] + noise_linear[non_positive_mask]
            )

        # Optionally add noise to positive regions
        if apply_wet_noise and np.any(positive_mask):
            scaled_wet_noise = noise_linear[positive_mask] * self.wet_noise_amplitude
            output_cube.data[positive_mask] = (
                template.data[positive_mask] + scaled_wet_noise
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
        # Treat any non-finite transformed values as below-threshold values
        linear[~np.isfinite(linear)] = 0.0
        linear[linear < self.db_threshold] = 0.0
        return linear

    def do_fft(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """
        Generate stochastic noise using SSFT for a 2-D array slice (one realization).

        This may raise ValueError if individual windows within the field are
        degenerate (constant-valued), even if the overall field has variation.
        In such cases, the caller should fall back to linear noise generation.

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

    @staticmethod
    def _is_degenerate_field(data: np.ndarray) -> bool:
        """Return True if field has no dynamic range for SSFT initialisation."""
        return not np.any(data > np.min(data))

    def _fallback_noise_linear(self, shape: tuple) -> np.ndarray:
        """Generate strictly non-positive fallback noise in linear space.

        If a seed is configured in ``ssft_generate_params``, this returns
        reproducible noise. The resulting field has a maximum value slightly
        below zero so dry fields remain dry while still receiving tie-break noise.

        Args:
            shape:
                Target 2-D output shape.

        Returns:
            Fallback 2-D noise field in linear units.
        """
        seed = self.ssft_generate_params.get("seed")
        if seed is not None:
            seed = int(seed)
        random_state = np.random.RandomState(seed)

        sigma = max(self.db_threshold * 0.1, np.finfo(np.float32).eps)
        epsilon = max(np.finfo(np.float32).eps, self.db_threshold)
        noise = random_state.normal(loc=0.0, scale=sigma, size=shape)
        noise = noise - np.max(noise) - epsilon
        return noise.astype(np.float32)

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
