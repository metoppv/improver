# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
"""CLI to apply rainforests calibration."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    forecast: cli.inputcube,
    *features: cli.inputcube,
    model_config: cli.inputjson,
    error_percentiles_count: int = 19,
    output_realizations_count: int = 100,
    threads: int = 1,
):
    """
    Calibrate a forecast cube using the Rainforests method.

    Ensemble forecasts must be in realization representation. Deterministic forecasts
    can be processed to produce a pseudo-ensemble; a realization dimension will be added
    to deterministic forecast cubes if one is not already present.

    This calibration is done in a situation dependent fashion using a series of
    decision-tree models to construct representative error distributions which are
    then used to map each input ensemble member onto a series of realisable values.
    These series collectively form a super-ensemble, from which realizations are
    sampled to produce the calibrated forecast.

    Args:
        forecast_cube (iris.cube.Cube):
            Cube containing the forecast to be calibrated; must be as realizations.
        feature_cubes (iris.cube.Cubelist):
            Cubelist containing the feature variables (physical parameters) used as inputs
            to the tree-models for the generation of the associated error distributions.
            Feature cubes are expected to have the same dimensions as forecast_cube, with
            the exception of the realization dimension. Where the feature_cube contains a
            realization dimension this is expected to be consistent, otherwise the cube will
            be broadcast along the realization dimension.
        model_config (dict):
            Dictionary containing RainForests model configuration data.
        error_percentiles_count (int):
            The number of error percentiles to apply to each ensemble realization.
            The resulting super-ensemble will be of size = forecast.realization.size *
            error_percentiles_count.
        output_realizations_count (int):
            The number of realizations to output for the calibrated ensemble.
            These realizations are sampled by taking equispaced percentiles
            from the super-ensemble. If None is supplied, then all realizations
            from the super-ensemble will be returned.
        threads (int):
            Number of threads to use during prediction with tree-model objects.

    Returns:
        iris.cube.Cube:
            The forecast cube following calibration.
    """
    from iris.cube import CubeList

    from improver.calibration.rainforest_calibration import ApplyRainForestsCalibration

    return ApplyRainForestsCalibration(model_config, threads).process(
        forecast,
        CubeList(features),
        error_percentiles_count=error_percentiles_count,
        output_realizations_count=output_realizations_count,
    )
