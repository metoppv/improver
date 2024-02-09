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
"""CLI to generate categorical data."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcubelist,
    decision_tree: cli.inputjson = None,
    model_id_attr: str = None,
    record_run_attr: str = None,
    target_period: int = None,
    check_tree: bool = False,
    title: str = None,
):
    """Generates categorical data. Uses a decision tree to determine which category represents
    each location in the input cubes. Used to generate categorical data like weather symbols.

    Args:
        cubes (iris.cube.CubeList):
            A cubelist containing the diagnostics required for the
            decision tree, these at co-incident times.
        decision_tree (dict):
            A JSON file containing a decision tree definition. Full information on decision
            trees can be found in improver.categorical.decision_tree.
        model_id_attr (str):
            Name of attribute recording source models that should be
            inherited by the output cube. The source models are expected as
            a space-separated string.
        record_run_attr:
            Name of attribute used to record models and cycles used in
            constructing the categorical data.
        target_period:
            The period in seconds that the categorical data being produced should
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
            output. This will override the title generated from
            the inputs, where this generated title is only set if all of the
            inputs share a common title.

    Returns:
        iris.cube.Cube:
            A cube of categorical data.
    """
    if check_tree:
        from improver.categorical.utilities import check_tree

        return check_tree(decision_tree, target_period=target_period)

    from iris.cube import CubeList

    from improver.categorical.decision_tree import ApplyDecisionTree
    from improver.utilities.flatten import flatten

    if not cubes:
        raise RuntimeError("Not enough input arguments. See help for more information.")

    cubes = flatten(cubes)
    return ApplyDecisionTree(
        decision_tree,
        model_id_attr=model_id_attr,
        record_run_attr=record_run_attr,
        target_period=target_period,
        title=title,
    )(CubeList(cubes))
