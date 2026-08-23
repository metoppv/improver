#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to add stochastic noise to a cube using Short-Space Fourier Transform (SSFT)."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    input_cube: cli.inputcube,
    *,
    ssft_init_params: str = None,
    ssft_generate_params: str = None,
    db_threshold: float = 0.03,
    db_threshold_units: str = "mm/hr",
    scale_non_positive_noise=False,
    allow_seeded_parallel_processing: bool = False,
    wet_noise_floor: float = None,
    dry_fallback_range: str = None,
    apply_noise_to_positive_regions: bool = False,
    wet_noise_amplitude: float = 1.0,
    apply_noise_to_positive_regions_by_source: str = None,
):
    """
    Class to apply spatially-structured stochastic noise to non-positive regions of a
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
    indicate a non-zero value should occur. By adding spatially-structured stochastic
    noise to break ties in these non-positive regions, more realistic spatial structures
    can be generated in the final ECC-Q realizations, while still respecting the
    calibrated probabilities.

    For a typical input field e.g. a precipitation field with some positive values
    for precipitation spread across the domain and some zero values, the plugin will
    add stochastic noise to the zero values using the SSFT approach, while leaving
    the positive values unchanged (or adding noise if apply_noise_to_positive_regions
    is True). For fields that contain insufficient spatial variability to derive
    meaningful SSFT perturbations (for example completely dry, nearly dry, or otherwise
    near-constant fields), referred to here as degenerate fields, the plugin will
    generate fallback stochastic noise ("dry fallback noise") in linear space. This
    noise uses the wet_noise_floor and dry_fallback_range arguments to ensure that the
    fallback noise is strictly non-positive and does not exceed the noise added to
    wet regions.

    While this plugin accepts any cube with "x" and "y" dimensions, it is
    recommended to first slice the cube over the realization dimension and
    parallelize the processing of individual realizations using the plugin on each
    slice, to improve performance. This extraction and later merging of realization
    slices can be easily achieved using the improver CLI `extract` and
    `merge` functionality, respectively.


    See Pysteps documentation for further keyword arguments.

    Args:
        input_cube:
            Cube to which stochastic noise will be added. Typically a dependence
            template cube for ECC-Q realization generation, where noise is added to
            non-positive regions (e.g., locations with zero precipitation) to break ties
            in the raw ensemble and allow meaningful reordering.
        ssft_init_params:
            Keyword arguments for initializing SSFT filter using
            `pysteps.noise.fftgenerators.initialize_nonparam_2d_ssft_filter
            <https://pysteps.readthedocs.io/en/stable/generated/pysteps.noise.fftgenerators.initialize_nonparam_2d_ssft_filter.html>`_.
            Provide as Python dict string,
            e.g., "{'win_size': (100, 100), 'overlap': 0.3}".
            This string will be converted to a Python dict using `ast.literal_eval`,
            therefore please ensure that the string is a valid Python dict
            representation.
            Recommended keys: win_size, overlap, war_thr.
            Default is an empty dict, which will use the pysteps defaults.
        ssft_generate_params:
            Keyword arguments for generating stochastic noise using
            `pysteps.noise.fftgenerators.generate_noise_2d_ssft_filter
            <https://pysteps.readthedocs.io/en/stable/generated/pysteps.noise.fftgenerators.generate_noise_2d_ssft_filter.html>`_.
            Provide as Python dict string, e.g., "{'overlap': 0.3, 'seed': 0}".
            This string will be converted to a Python dict using `ast.literal_eval`,
            therefore please ensure that the string is a valid Python dict
            representation.
            Recommended keys: overlap, seed.
            Default is an empty dict, which will use the pysteps defaults.
        db_threshold:
            Threshold value below which data will be set to a constant in dB scale
            to avoid issues with log(0). Value provided in units of
            `db_threshold_units`.
            Default is 0.03 mm/hr.
        db_threshold_units:
            Units of the db_threshold value. Default is "mm/hr".
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
        wet_noise_floor:
            Optional lower bound for noise in non-positive regions after scaling,
            in linear units of db_threshold_units. Must be negative if set.
            Required when the input field is degenerate (e.g. all-zero), to guarantee
            separation between dry-fallback and wet-member noise. This can be used to
            limit the magnitude of negative SSFT-derived wet-member noise. This value
            must be less than the SSFT-derived noise in wet regions. Any generated
            noise below the floor value will be set to the floor value, potentially
            resulting in more ties when used in conjunction with Ensemble Copula
            Coupling. Default is None (no floor).
        dry_fallback_range:
            Optional range (min_value, max_value) for dry fallback noise in
            linear units of db_threshold_units. Provide as a Python tuple string, e.g.
            "(-10.0, -5.0)". This string will be converted to a Python tuple using
            `ast.literal_eval`, therefore please ensure that the string is a valid
            Python tuple representation. Both values must be <= 0 and
            (min_value < max_value). If wet_noise_floor is set and this is
            not provided, this defaults to (2 * wet_noise_floor, wet_noise_floor)
            to keep dry fallback below the wet floor. If wet_noise_floor is set and
            dry_fallback_range is provided, the max_value of dry_fallback_range
            must be <= wet_noise_floor to ensure separation between dry-fallback
            and wet noise ranges.
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
            period's source is in this list. Cannot be used together with
            apply_noise_to_positive_regions=True. Default is None (use
            apply_noise_to_positive_regions flag instead).

    Returns:
        Cube with added stochastic noise.
    """
    import ast

    from improver.calibration.stochastic_noise import StochasticNoise

    # Parse string representations to dicts
    if ssft_init_params and isinstance(ssft_init_params, str):
        ssft_init_params = ast.literal_eval(ssft_init_params)
    else:
        ssft_init_params = {}
    if ssft_generate_params and isinstance(ssft_generate_params, str):
        ssft_generate_params = ast.literal_eval(ssft_generate_params)
    else:
        ssft_generate_params = {}

    parsed_dry_fallback_range = None
    if dry_fallback_range and isinstance(dry_fallback_range, str):
        parsed_dry_fallback_range = ast.literal_eval(dry_fallback_range)

    plugin_kwargs = {
        "ssft_init_params": ssft_init_params,
        "ssft_generate_params": ssft_generate_params,
        "db_threshold": db_threshold,
        "db_threshold_units": db_threshold_units,
        "scale_non_positive_noise": scale_non_positive_noise,
        "allow_seeded_parallel_processing": allow_seeded_parallel_processing,
        "wet_noise_floor": wet_noise_floor,
        "dry_fallback_range": parsed_dry_fallback_range,
        "apply_noise_to_positive_regions": apply_noise_to_positive_regions,
        "wet_noise_amplitude": wet_noise_amplitude,
        "apply_noise_to_positive_regions_by_source": apply_noise_to_positive_regions_by_source,
    }

    plugin = StochasticNoise(**plugin_kwargs)

    result = plugin.process(input_cube)
    return result
