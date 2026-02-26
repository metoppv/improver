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
    num_workers: int = None,
    scale_non_positive_noise=False,
):
    """
    Class to apply spatially-structured stochastic noise to non-positive regions of a
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
            Recommended keys: win_size, overlap, war_thr.
        ssft_generate_params:
            Keyword arguments for generating stochastic noise using
            `pysteps.noise.fftgenerators.generate_noise_2d_ssft_filter
            <https://pysteps.readthedocs.io/en/stable/generated/pysteps.noise.fftgenerators.generate_noise_2d_ssft_filter.html>`_.
            Provide as Python dict string, e.g., "{'overlap': 0.3, 'seed': 0}".
            Recommended keys: overlap, seed.
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
            If True, noise in non-positive regions (where template.data <= 0) will be
            scaled such that the maximum noise value in those regions is zero and all
            other noise values are negative. This prevents the addition of positive
            noise to non-positive regions, which could artificially increase values
            where the input cube indicates no signal should occur.
            Default is False.

    Returns:
        Cube with added stochastic noise.

    See Pysteps documentation for further keyword arguments.
    """
    import ast

    from improver.stochastic_noise import StochasticNoise

    # Parse string representations to dicts
    if ssft_init_params and isinstance(ssft_init_params, str):
        ssft_init_params = ast.literal_eval(ssft_init_params)
    else:
        ssft_init_params = {}
    if ssft_generate_params and isinstance(ssft_generate_params, str):
        ssft_generate_params = ast.literal_eval(ssft_generate_params)
    else:
        ssft_generate_params = {}

    plugin_kwargs = {
        "ssft_init_params": ssft_init_params,
        "ssft_generate_params": ssft_generate_params,
        "db_threshold": db_threshold,
        "db_threshold_units": db_threshold_units,
        "scale_non_positive_noise": scale_non_positive_noise,
    }
    if num_workers is not None:
        plugin_kwargs["num_workers"] = num_workers

    plugin = StochasticNoise(**plugin_kwargs)

    result = plugin.process(input_cube)
    return result
