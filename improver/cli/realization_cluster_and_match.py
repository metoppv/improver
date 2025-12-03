#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to run clustering and matching of realizations."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube,
    hierarchy: cli.inputjson,
    n_clusters: int,
    model_id_attr: str = "mosg__model_configuration",
    target_grid_name: str = "target_grid",
    clustering_method: str = "KMedoids",
    regrid_mode: str = "esmf-area-weighted",
    clustering_kwargs: cli.inputjson = None,
    regrid_kwargs: cli.inputjson = None,
):
    """Cluster primary input and match secondary inputs to clusters.

    Clusters the primary input using the specified clustering method, then
    matches secondary input realizations to the resulting clusters based on
    mean squared error. The hierarchy configuration specifies which input is
    primary (to be clustered) and which are secondary (to be matched).

    Args:
        cubes (list of iris.cube.Cube):
            Input cubes containing primary and secondary forecast data.
            Different forecast sources must be identifiable using the
            model_id_attr attribute.
        hierarchy (dict):
            Dictionary defining the hierarchy of inputs. Specifies the primary
            input (which is clustered) and secondary inputs (which are matched
            to clusters). The order of secondary_inputs defines precedence,
            with earlier entries having higher priority. Format::

                {
                    "primary_input": "model_name",
                    "secondary_inputs": {"model2": [0, 6], "model3": [0, 24]},
                }

            The lists specify forecast period hours. A two-element list [start, end]
            will be expanded to the range start to end inclusive (e.g., [0, 6]
            includes 0, 1, 2, 3, 4, 5, 6). Lists with other lengths are treated as
            explicit forecast period hours. Only forecast periods that actually exist
            in the input cubes within these ranges will be processed. The hour values
            will be automatically converted to seconds to match the forecast_period
            coordinate units in the input cubes.
        n_clusters (int):
            Number of clusters to create. This determines how many representative
            realizations will be selected from the primary input.
        model_id_attr (str):
            Name of the attribute used to identify different models within
            the input cubes.
            Default: "mosg__model_configuration"
        target_grid_name (str):
            Name of the target grid cube for regridding. The input cubes
            must include a cube with this name.
            Default: "target_grid"
        clustering_method (str):
            Clustering method to use. Currently only "KMedoids" is supported.
            Default: "KMedoids"
        regrid_mode (str):
            Regridding mode to use for regridding to the target grid.
            Valid options include "bilinear", "nearest", "esmf-area-weighted",
            "nearest-with-mask", etc.
            Default: "esmf-area-weighted"
        clustering_kwargs (dict):
            Additional keyword arguments to pass to the clustering method.
            Can be provided as a JSON file path or a JSON string. Common
            options for KMedoids include:

            - random_state (int): Random seed for reproducibility
            - max_iter (int): Maximum number of iterations

            Example::

                {"random_state": 42, "max_iter": 300}

            Default: None (no additional kwargs)
        regrid_kwargs (dict):
            Additional keyword arguments to pass to RegridLandSea for
            regridding. Can be provided as a JSON file path or a JSON string.
            Common options include:

            - mdtol (float): Tolerance of missing data for esmf-area-weighted
              regridding (0 to 1, default 1)
            - extrapolation_mode (str): Mode to fill regions outside domain
            - landmask (Cube): Land-sea mask for mask-aware regridding

            Example::

                {"mdtol": 0.5}

            Default: None (no additional kwargs)

    Returns:
        iris.cube.Cube:
            Cube containing the clustered and matched realizations, with
            secondary inputs matched to clusters according to the hierarchy.
    """
    from iris.cube import CubeList

    from improver.clustering.realization_clustering import (
        RealizationClusterAndMatch,
    )

    # Use clustering_kwargs if provided, otherwise use empty dict
    clustering_kw = clustering_kwargs if clustering_kwargs is not None else {}

    # Add n_clusters to kwargs
    clustering_kw["n_clusters"] = n_clusters

    # Use regrid_kwargs if provided, otherwise use empty dict
    regrid_kw = regrid_kwargs if regrid_kwargs is not None else {}

    plugin = RealizationClusterAndMatch(
        hierarchy=hierarchy,
        model_id_attr=model_id_attr,
        clustering_method=clustering_method,
        target_grid_name=target_grid_name,
        regrid_mode=regrid_mode,
        regrid_kwargs=regrid_kw,
        **clustering_kw,
    )

    return plugin(CubeList(cubes))
