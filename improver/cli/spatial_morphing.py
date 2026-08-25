#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

"""Script to apply spatial morphing between forecast sources at a fixed validity time."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube,
    forecast_period: int,
    cluster_number: int,
    model_id_attr: str = "mosg__model_configuration",
    cycletime: str = None,
    selection_attr: str = "spatial_morphing",
    selection_attr_value: str = "cluster_medoid",
    transitions: cli.inputjson = None,
    model_path: str = None,
    scaling: str = "minmax",
    clipping_bounds: cli.comma_separated_list = None,
    clip_in_scaled_space: bool = False,
    clip_to_physical_bounds: bool = False,
    max_batch: int = 1,
    parallel_backend: str = None,
    n_workers: int = 1,
    transition_weights_scheme: str = "linear",
):
    """Apply spatial morphing between forecast sources at a fixed validity time.

    This plugin selects realizations from multiple forecast sources according to
    cluster assignments, then applies Google FILM spatial morphing to create
    seamless transitions between different source models.

    Args:
        cubes (list of Cube):
            List of input cubes, including forecast cubes from different sources
            and a cluster cube with mapping attributes from
            RealizationClusterAndMatch. The cluster cube is identified by the
            presence of the "primary_input_realization_to_cluster_medoid"
            attribute.
        forecast_period (int):
            The forecast period (in seconds) to use for interrogating the cluster
            mapping attributes in order to select the appropriate realizations
            from each forecast source.
        cluster_number (int):
            The cluster index (int) to select realizations for. Only this cluster
            will be processed; output will be a single realization with this index.
        model_id_attr (str):
            The name of the cube attribute used to identify the model source.
            Default: "mosg__model_configuration".
        cycletime (str):
            The forecast_reference_time on the input forecast cubes will be reset
            to this value. The forecast periods will be adjusted accordingly with
            the validity times kept fixed. cycletime should be provided in the
            format YYYYMMDDTHHMMZ (e.g., 20240101T0000Z). If not provided, the
            forecast_reference_time on the input cubes will be left unchanged.
        selection_attr (str):
            Optional name of a cube attribute to add to the output to identify
            that realizations were selected and morphed using this plugin.
            If not provided, no attribute is added.
        selection_attr_value (str):
            The value to assign to the selection_attr attribute.
            Default is "spatial_morphing".
        transitions (dict):
            JSON dictionary containing an explicit "transitions" list. Each
            transition must define "source_a", "source_b",
            "start_forecast_period_minutes", and
            "end_forecast_period_minutes".
        model_path (str):
            Path to TensorFlow Hub module for Google FILM model. Required if
            spatial morphing between different sources is performed.
        scaling (str):
            Scaling method for FILM interpolation: "log10" or "minmax".
            Default: "minmax".
        clipping_bounds (tuple or dict):
            Optional JSON dict/tuple specifying (min, max) bounds for clipping
            interpolated data. Example: {"min": -50, "max": 150} or [0, 100].
        clip_in_scaled_space (bool):
            If True, clipping applied before reverse scaling.
            Default: True.
        clip_to_physical_bounds (bool):
            If True, clipping applied after reverse scaling.
            Default: False.
        max_batch (int):
            Maximum batch size for FILM inference. Default: 1.
        parallel_backend (str):
            Parallelization backend ("loky") or None for serial.
            Default: None.
        n_workers (int):
            Number of workers for parallel processing. Default: 1.
        transition_weights_scheme (str):
            Scheme for computing transition weights: "linear" or "smoothstep".
            Default: "linear".

    Returns:
        Cube:
            Single Cube containing the selected and (if applicable) spatially
            morphed realization, with realization index set to cluster_number.
    """
    from improver.utilities.spatial_morphing import SpatialMorphing

    morphing = SpatialMorphing(
        forecast_period=forecast_period,
        cluster_number=cluster_number,
        model_id_attr=model_id_attr,
        cycletime=cycletime,
        selection_attr=selection_attr,
        selection_attr_value=selection_attr_value,
        transitions=transitions,
        model_path=model_path,
        scaling=scaling,
        clipping_bounds=clipping_bounds,
        clip_in_scaled_space=clip_in_scaled_space,
        clip_to_physical_bounds=clip_to_physical_bounds,
        max_batch=max_batch,
        parallel_backend=parallel_backend,
        n_workers=n_workers,
        transition_weights_scheme=transition_weights_scheme,
    )
    return morphing.process(*cubes)
