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
    scale_dry_noise=False,
):
    """
    Add stochastic noise to a dependence template cube object using Short-Space
    Fourier Transform (SSFT).

    Args:
        ssft_init_params:
            Keyword arguments for initializing SSFT filter using
            pysteps.noise.fftgenerators.initialize_nonparam_2d_ssft_filter.
            Provide as Python dict string,
            e.g., "{'win_size': (100, 100), 'overlap': 0.3}".
            Recommended keys: win_size, overlap, war_thr.
        ssft_generate_params:
            Keyword arguments for generating stochastic noise using
            pysteps.noise.fftgenerators.generate_noise_2d_ssft_filter.
            Provide as Python dict string, e.g., "{'overlap': 0.3, 'seed': 0}".
            Recommended keys: overlap, seed.
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


    Returns:
        Cube with added stochastic noise.

    See Pysteps documentation for further keyword arguments.
    """
    import ast

    from improver.precipitation.stochastic_noise import StochasticNoise

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
        "scale_dry_noise": scale_dry_noise,
    }
    if num_workers is not None:
        plugin_kwargs["num_workers"] = num_workers

    plugin = StochasticNoise(**plugin_kwargs)

    result = plugin.process(input_cube)
    return result
