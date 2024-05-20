#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to generate modal categories over periods."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube,
    decision_tree: cli.inputjson = None,
    model_id_attr: str = None,
    record_run_attr: str = None,
):
    """Generates a modal category for the period covered by the input
    categorical cubes. Where there are different categories available
    for night and day, the modal code returned is always a day code, regardless
    of the times covered by the input files.
    Designed for use with weather symbol data.

    Args:
        cubes (iris.cube.CubeList):
            A cubelist containing categorical cubes that cover the period
            over which a modal category is desired.
        decision_tree (dict):
            A JSON file containing a decision tree definition.
        model_id_attr (str):
            Name of attribute recording source models that should be
            inherited by the output cube. The source models are expected as
            a space-separated string.
        record_run_attr:
            Name of attribute used to record models and cycles used in
            constructing the categorical data.

    Returns:
        iris.cube.Cube:
            A cube of modal categories over a period.
    """
    from improver.categorical.modal_code import ModalCategory

    if not cubes:
        raise RuntimeError("Not enough input arguments. See help for more information.")

    return ModalCategory(
        decision_tree, model_id_attr=model_id_attr, record_run_attr=record_run_attr,
    )(cubes)
