# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""RainForests calibration Plugins."""

import warnings

import numpy as np
from iris.cube import Cube, CubeList

from improver import PostProcessingPlugin

# Passed to choose_set_of_percentiles to set of evenly spaced percentiles
DEFAULT_ERROR_PERCENTILES_COUNT = 19
DEFAULT_OUTPUT_REALIZATIONS_COUNT = 100


class ApplyRainForestsCalibration(PostProcessingPlugin):
    """Class to calibrate input forecast given a series of RainForests tree models."""

    def __init__(self, model_config_dict: dict, threads: int):
        """Initialise the tree model variables used in the application of RainForests
        Calibration.

        Args:
            model_config_dict:
                Dictionary containing Rainforests model configuration variables.
            threads:
                Number of threads to use during prediction with tree-model objects.

        Dictionary is of format::

            {
                "-50.0" : {
                    "lightgbm_model" : "<path_to_lightgbm_model_object>",
                    "treelite_model" : "<path_to_treelite_model_object>"
                },
                "-25.0" : {
                    "lightgbm_model" : "<path_to_lightgbm_model_object>",
                    "treelite_model" : "<path_to_treelite_model_object>"
                },
                ...,
                "50.0" : {
                    "lightgbm_model" : "<path_to_lightgbm_model_object>",
                    "treelite_model" : "<path_to_treelite_model_object>"
                }
            }

        The keys specify the error threshold value, while the associated values
        are the path to the corresponding tree-model objects for that threshold.

        Treelite predictors are used if treelite_runitme is an installed dependency
        and an associated path has been provided for all thresholds, otherwise lightgbm
        Boosters are used as the default tree model type.
        """
        from lightgbm import Booster

        try:
            from treelite_runtime import Predictor

            self.treelite_enabled = True
        except ModuleNotFoundError:
            warnings.warn(
                "Module treelite_runtime unavailable. Defaulting to using lightgbm Boosters."
            )
            self.treelite_enabled = False

        error_thresholds = list(model_config_dict.keys())

        lightgbm_model_filenames = [
            model_config_dict[threshold].get("lightgbm_model")
            for threshold in error_thresholds
        ]
        treelite_model_filenames = [
            model_config_dict[threshold].get("treelite_model")
            for threshold in error_thresholds
        ]
        if (None not in treelite_model_filenames) and self.treelite_enabled:
            self.tree_models = [
                Predictor(libpath=file, verbose=False, nthread=threads)
                for file in treelite_model_filenames
            ]
        else:
            self.tree_models = [
                Booster(model_file=file).reset_parameter({"num_threads": threads})
                for file in lightgbm_model_filenames
            ]

        self.error_thresholds = np.array(error_thresholds, dtype=np.float32)

    def process(
        self,
        forecast_cube: Cube,
        feature_cubes: CubeList,
        error_percentiles_count: int = DEFAULT_ERROR_PERCENTILES_COUNT,
        output_realizations_count: int = DEFAULT_OUTPUT_REALIZATIONS_COUNT,
    ) -> Cube:
        """Apply rainforests calibration to forecast cube.

        Ensemble forecasts must be in realization representation. Deterministic forecasts
        can be processed to produce a pseudo-ensemble; a realization dimension will be added
        to deterministic forecast cubes if one is not already present.

        The calibration is done in a situation dependent fashion using a series of
        decision-tree models to construct representative error distributions which are
        then used to map each input ensemble member onto a series of realisable values.

        These error distributions are formed in a two-step process:

        1. Evalute error CDF defined over the specified error_thresholds. Each exceedence
        probability is evaluated using the corresponding decision-tree model.

        2. Interpolate the CDF to extract a series of percentiles for the error distribution.
        The error percentiles are then applied to each associated ensemble realization to
        produce a series of realisable values; collectively these series form a calibrated
        super-ensemble, which is then sub-sampled to provide the calibrated forecast.

        Args:
            forecast_cube:
                Cube containing the forecast to be calibrated; must be as realizations.
            feature_cubes:
                Cubelist containing the feature variables (physical parameters) used as inputs
                to the tree-models for the generation of the associated error distributions.
                Feature cubes are expected to have the same dimensions as forecast_cube, with
                the exception of the realization dimension. Where the feature_cube contains a
                realization dimension this is expected to be consistent, otherwise the cube will
                be broadcast along the realization dimension.
            error_percentiles_count:
                The number of error percentiles to extract from the associated error CDFs
                evaluated via the tree-models. These error percentiles are applied to each
                ensemble realization to produce a series of values, which collectively form
                the super-ensemble. The resulting super-ensemble will be of
                size = forecast.realization.size * error_percentiles_count.
            output_realizations_count:
                The number of ensemble realizations that will be extracted from the
                super-ensemble. If realizations_count is None, all realizations will
                be returned.

        Returns:
            The calibrated forecast cube.
        """
        # Until calibration processing steps are added, the behaviour here will be to return
        # the forecast cube without calibration. This will enable other integration work to
        # proceed concurrently with development of this Plugin.
        return forecast_cube
