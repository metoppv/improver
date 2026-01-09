#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to add stochastic noise to a dependence template cube using SSFT."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    dependence_template: cli.inputcube,
    *,
    ssft_init_params=None,
    ssft_generate_params=None,
    threshold: float = 0.03,
    threshold_units: str = "mm/hr",
):
    """
    Add stochastic noise to a dependence template cube object using SSFT.

    Args:
        ssft_init_params:
            Keyword arguments for initializing SSFT filter using
            pysteps.noise.fftgenerators.initialize_nonparam_2d_ssft_filter.
            Provide as Python dict string, e.g. "{'win_size': (100, 100), 'overlap': 0.3}".
            Recommended keys: win_size, overlap, war_thr.
        ssft_generate_params:
            Keyword arguments for generating stochastic noise using
            pysteps.noise.fftgenerators.generate_noise_2d_ssft_filter.
            Provide as Python dict string, e.g. "{'overlap': 0.3, 'seed': 0}".
            Recommended keys: overlap, seed.
        threshold:
            Threshold value below which data will be set to a constant in dB scale.
            Default is 0.03 mm/hr.
        threshold_units:
            Units of the threshold value. Default is "mm/hr".

    Returns:
        Original dependence template cube with added stochastic noise.

    See Pysteps documentation for further keyword arguments.
    """
    import ast

    from improver.precipitation.stochastic_noise import StochasticNoise

    # Parse string representations to dicts
    if ssft_init_params and isinstance(ssft_init_params, str):
        ssft_init_params = ast.literal_eval(ssft_init_params)
    if ssft_generate_params and isinstance(ssft_generate_params, str):
        ssft_generate_params = ast.literal_eval(ssft_generate_params)

    if not ssft_init_params:
        ssft_init_params = {
            "win_size": (100, 100),
            "overlap": 0.3,
            "war_thr": 0.1,
        }
    if not ssft_generate_params:
        ssft_generate_params = {
            "overlap": 0.3,
            "seed": 0,
        }
    plugin = StochasticNoise(
        ssft_init_params=ssft_init_params,
        ssft_generate_params=ssft_generate_params,
        threshold=threshold,
        threshold_units=threshold_units,
    )

    result = plugin.process(dependence_template)
    return result
