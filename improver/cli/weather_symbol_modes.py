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
"""CLI to generate modal categories over periods."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube,
    broad_categories: cli.inputjson = None,
    wet_categories: cli.inputjson = None,
    intensity_categories: cli.inputjson = None,
    day_weighting: int = 1,
    day_start: int = 6,
    day_end: int = 18,
    wet_bias: int = 1,
    ignore_intensity: bool = False,
    model_id_attr: str = None,
    record_run_attr: str = None,
):
    """Generates a modal weather code for the period covered by the input
    categorical cubes. Where there are different categories available
    for night and day, the modal code returned is always a day code, regardless
    of the times covered by the input files. The weather codes provided are expected
    to end at midnight and therefore represent either a full day or a partial day.

    Args:
        cubes (iris.cube.CubeList):
            A cubelist containing categorical cubes that cover the period
            over which a modal category is desired.
        broad_categories (dict):
            A JSON file containing a definition for a broad category grouping.
            The expected categories are wet and dry.
        wet_categories (dict):
            A JSON file containing a definition for a wet category grouping.
        intensity_categories (dict):
            A JSON file containing a definition for an intensity category grouping.
        day_weighting:
            Weighting to provide day time weather codes. A weighting of 1 indicates
            the default weighting. A weighting of 2 indicates that the weather codes
            during the day time period will be duplicated, so that they count twice
            as much when computing a representative weather code.
        day_start:
            Hour defining the start of the daytime period.
        day_end:
            Hour defining the end of the daytime period.
        wet_bias:
            Bias to provide wet weather codes. A bias of 1 indicates the
            default, where half of the codes need to be a wet code,
            in order to generate a wet code. A bias of 3 indicates that
            only a quarter of codes are required to be wet, in order to generate
            a wet symbol. To generate a wet symbol, the fraction of wet symbols
            therefore need to be greater than or equal to 1 / (1 + wet_bias).
        ignore_intensity:
            Boolean indicating whether weather codes of different intensities
            should be grouped together when establishing the most representative
            weather code. The most common weather code from the options available
            representing different intensities will be used as the representative
            weather code.
        model_id_attr (str):
            Name of attribute recording source models that should be
            inherited by the output cube. The source models are expected as
            a space-separated string.
        record_run_attr:
            Name of attribute used to record models and cycles used in
            constructing the categorical data.

    Returns:
        iris.cube.Cube:
            A cube of modal weather codes over a period.
    """
    from improver.wxcode.modal_code import ModalFromGroupings

    if not cubes:
        raise RuntimeError("Not enough input arguments. See help for more information.")

    return ModalFromGroupings(
        broad_categories,
        wet_categories,
        intensity_categories,
        day_weighting,
        day_start,
        day_end,
        wet_bias,
        ignore_intensity,
        model_id_attr=model_id_attr,
        record_run_attr=record_run_attr,
    )(cubes)
