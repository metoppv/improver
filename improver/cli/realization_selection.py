#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

"""Script to select realizations based on clustering results."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube,
    forecast_period: int,
    model_id_attr: str = "mosg__model_configuration",
):
    """Select realizations from input forecast cubes according to cluster assignments.

    Args:
        cubes (list of Cube):
            List of input cubes, including forecast cubes and a cluster cube.
            The cluster cube is identified by the presence of the
            "primary_input_realization_to_cluster_medoid" attribute.
        forecast_period (int):
            The forecast period (in seconds) to use for interrogating the cluster
            mapping attributes in order to select the appropriate realizations.
        model_id_attr (str):
            The name of the cube attribute used to identify the model source.

    Returns:
        Cube:
            A merged Cube containing the selected realizations, with realization
            indices matching the cluster indices.
    """
    from improver.clustering.realization_clustering import RealizationSelection

    return RealizationSelection(
        forecast_period=forecast_period, model_id_attr=model_id_attr
    )(cubes)
