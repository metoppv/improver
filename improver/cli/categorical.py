#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
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
