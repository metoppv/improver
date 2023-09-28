#!/usr/bin/env python
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
"""CLI to generate modal weather symbols over periods."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube, model_id_attr: str = None, record_run_attr: str = None
):
    """Generates a modal weather symbol for the period covered by the input
    weather symbol cubes. Where there are different weather codes available
    for night and day, the modal code returned is always a day code, regardless
    of the times covered by the input files.

    Args:
        cubes (iris.cube.CubeList):
            A cubelist containing weather symbols cubes that cover the period
            over which a modal symbol is desired.
        model_id_attr (str):
            Name of attribute recording source models that should be
            inherited by the output cube. The source models are expected as
            a space-separated string.
        record_run_attr:
            Name of attribute used to record models and cycles used in
            constructing the weather symbols.

    Returns:
        iris.cube.Cube:
            A cube of modal weather symbols over a period.
    """
    from improver.wxcode.modal_code import ModalWeatherCode

    if not cubes:
        raise RuntimeError("Not enough input arguments. See help for more information.")

    return ModalWeatherCode(
        model_id_attr=model_id_attr,
        record_run_attr=record_run_attr,
    )(cubes)
