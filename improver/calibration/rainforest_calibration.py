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

from typing import Any, List

from iris.cube import Cube, CubeList
from numpy import ndarray

from improver import BasePlugin, PostProcessingPlugin

# Passed to choose_set_of_percentiles to set of evenly spaced percentiles
DEFAULT_ERROR_PERCENTILES_COUNT = 19
DEFAULT_OUTPUT_REALIZATIONS_COUNT = 199

class TrainRainForestsTreeModels(BasePlugin):
    """Class to train tree models for use in RainForests calibration."""


class ApplyRainForestsCalibration(PostProcessingPlugin):
    """Class to calibrate input forecast given a series of RainForests tree models."""

    def process(
        self,
        forecast_cube: Cube,
        feature_cubes: CubeList,
        error_thresholds: ndarray,
        tree_models: List[Any],
        error_percentiles_count: int = DEFAULT_ERROR_PERCENTILES_COUNT,
        output_realizations_count: int = DEFAULT_OUTPUT_REALIZATIONS_COUNT,
    ) -> Cube:
        """Apply rainforests calibration to forecast cube.

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
                Cube containing the forecast to be calibrated.
            feature_cubes:
                Cubelist containing the feature variables for prediction.
            error_thresholds:
                Error thresholds corresponding to the treelite_models.
            tree_models:
                List of decision-tree models predicting the probability that the error is above
                the specified thresholds. These models can be either light-gbm Boosters, or
                treelite Predictors (compiled versions of the light-gbm models).
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
        pass
