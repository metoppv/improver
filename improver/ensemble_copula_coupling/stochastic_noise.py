# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Plugin for adding stochastic noise to a dependence template using Short-Space Fourier
Transform (SSFT).
"""

import copy
import os
import time

import numpy as np
from dask import compute, delayed
from pysteps.noise.fftgenerators import (
    generate_noise_2d_ssft_filter,
    initialize_nonparam_2d_ssft_filter,
)

from improver import BasePlugin

# I HAVE TAKEN THE EXACT WORDING OF PARAMETERS FROM THE PYSTEPS DOCUMENTATION
# (https://pysteps.readthedocs.io/en/latest/generated/pysteps.noise.fftgenerators.initialize_nonparam_2d_ssft_filter.html)


# It seems https://pysteps.readthedocs.io/en/latest/generated/pysteps.utils.interface.get_method.html#pysteps.utils.interface.get_method:~:text=scipy.fftpack-,pyfftw,-pyfftw.interfaces.numpy_fft]
# could be faster than use of numpy (default) or scipy. But this would require pyfftw to be installed.
# It can be accessed by setting the fft_method parameter in the initialize_nonparam_2d_ssft_filter
# and generate_noise_2d_ssft_filter functions to fft_method='pyfftw'.

"""
How the pluin is structured:
- Init function: to set up the parameters for SSFT
- Generate function: to generate the stochastic noise using SSFT
- Function to add the stochastic noise to the dependence template
"""
UNITS_DICT = {}


class StochasticNoise(BasePlugin):
    """Class to add stochastic noise to dependence template using SSFT."""

    ## NOTE: I had to use the below 2 lots of kwargs approach as each function in
    ## do_fft can allow different arguments for the same parameter. This is tidier than,
    ## e.g.

    def __init__(
        self,
        kwargs_initialize: dict,
        kwargs_generate: dict,
    ):
        """
        Initialise the plugin.

        Args:
            kwargs_initialize:
                Keyword arguments for initializing SSFT filter using
                pysteps.noise.fftgenerators.initialize_nonparam_2d_ssft_filter.
                - Recommended: win_size, overlap, war_thr.
            kwargs_generate:
                Keyword arguments for generating stochastic noise using
                pysteps.noise.fftgenerators.generate_noise_2d_ssft_filter.
                - Recommended: overlap, seed.

        Example dictionaries for initializing and generating SSFT filter:
            stochastic_noise_initialize_dict={
                "win_size": (100, 100),
                "overlap": 0.3,
                "war_thr": 0.1
            }
            stochastic_noise_generate_dict={
                "overlap": 0.3,
                "seed": 0
            }
        """
        self.kwargs_initialize = kwargs_initialize
        self.kwargs_generate = kwargs_generate

    def do_fft(
        self, data: np.ndarray, kwargs_initialize: dict, kwargs_generate: dict
    ) -> np.ndarray:
        """
        Function to generate stochastic noise using SSFT.

        Args:
            data (np.ndarray):
                2D / 3D array for which stochastic noise is to be added.
            stochastic_noise_dict1 (dict):
                Dictionary of parameters for initializing SSFT filter.
            stochastic_noise_dict2 (dict):
                Dictionary of parameters for generating stochastic noise using SSFT.
                - Recommended: overlap, seed.
        """
        print("Initialising ssft")
        Fnp = initialize_nonparam_2d_ssft_filter(
            data,
            **kwargs_initialize,
        )
        print("Generating stochastic noise")
        stochastic_noise = generate_noise_2d_ssft_filter(Fnp, **kwargs_generate)
        return stochastic_noise

    def stochastic_noise_to_dependence_template2(
        self, dependence_template, spatial_processing
    ):
        def to_dB(cube):
            threshold = 0.03
            condition1 = cube.data >= threshold
            condition2 = cube.data < threshold

            cube.data[condition1] = 10.0 * np.log10(cube.data[condition1])
            threshold_dB = 10.0 * np.log10(threshold)
            cube.data[condition2] = threshold_dB - 5
            return cube

        def from_dB(cube):
            cube.data = 10 ** (cube.data / 10.0)
            threshold = 0.03
            cube.data[cube.data < threshold] = 0
            return cube

        original_units = copy.deepcopy(dependence_template.units)
        dependence_template.convert_units(UNITS_DICT[dependence_template.name()])
        stochastic_noise_dB = dependence_template.copy()
        stochastic_noise = dependence_template.copy()
        dependence_template_dB = dependence_template.copy()

        dependence_template_dB.data = to_dB(dependence_template.copy()).data

        t0 = time.time()

        delay_results = []
        for k in range(len(dependence_template.coord("realization").points)):
            delay_results.append(
                delayed(self.do_fft)(
                    dependence_template_dB.data[k],
                    self.kwargs_initialize,
                    self.kwargs_generate,
                )
            )

        num_workers = min(
            len(dependence_template.coord("realization").points), os.cpu_count() - 1
        )

        results = compute(
            *delay_results,
            scheduler="threads",
            num_workers=num_workers,
        )

        for k, result in enumerate(results):
            stochastic_noise_dB.data[k] = result

        t1 = time.time()
        print("Time taken to compute stochastic noise = ", t1 - t0)

        stochastic_noise.data = from_dB(stochastic_noise_dB.copy()).data

        if spatial_processing == "stochastic_noise3":
            max_sn = np.max(stochastic_noise.data[dependence_template.data <= 0])
            stochastic_noise.data[dependence_template.data <= 0] = (
                stochastic_noise.data[dependence_template.data <= 0] - max_sn
            )

        dependence_template.data = dependence_template.data + stochastic_noise.data

        dependence_template.convert_units(original_units)
        return dependence_template

    def process():
        pass
