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
"""CLI to generate weather symbols."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube,
    wxtree: cli.inputjson = None,
    model_id_attr: str = None,
    record_run_attr: str = None,
    target_period: int = None,
    check_tree: bool = False,
    title: str = None,
):
    """ Processes cube for Weather symbols.

    Args:
        cubes (iris.cube.CubeList):
            A cubelist containing the diagnostics required for the
            weather symbols decision tree, these at co-incident times.
        wxtree (dict):
            A JSON file containing a weather symbols decision tree definition.
        model_id_attr (str):
            Name of attribute recording source models that should be
            inherited by the output cube. The source models are expected as
            a space-separated string.
        record_run_attr:
            Name of attribute used to record models and cycles used in
            constructing the weather symbols.
        target_period:
            The period in seconds that the weather symbol being produced should
            represent. This should correspond with any period diagnostics, e.g.
            precipitation accumulation, being used as input. This is used to scale
            any threshold values that are defined with an associated period in
            the decision tree. It will only be used if the decision tree
            provided has threshold values defined with an associated period.
        check_tree (bool):
            If set, the decision tree will be checked to see if it conforms to
            the expected format and that all nodes can be reached; the only other
            argument required is the path to the decision tree. If the tree is found
            to be valid the required inputs will be listed. Setting this flag will
            prevent the CLI performing any other actions.
        title (str):
            An optional title to assign to the title attribute of the resulting
            weather symbol output. This will override the title generated from
            the inputs, where this generated title is only set if all of the
            inputs share a common title.

    Returns:
        iris.cube.Cube:
            A cube of weather symbols.
    """
    if check_tree:
        from improver.wxcode.utilities import check_tree

        return check_tree(wxtree, target_period=target_period)

    from iris.cube import CubeList

    from improver.wxcode.weather_symbols import WeatherSymbols

    if not cubes:
        raise RuntimeError("Not enough input arguments. See help for more information.")

    return WeatherSymbols(
        wxtree,
        model_id_attr=model_id_attr,
        record_run_attr=record_run_attr,
        target_period=target_period,
        title=title,
    )(CubeList(cubes))
