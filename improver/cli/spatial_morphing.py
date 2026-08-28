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
    apply_quantile_mapping: bool = True,
    occurrence_threshold: float = 0.0,
):
    """Apply spatial morphing between forecast sources at a fixed validity time.

    This CLI wraps the SpatialMorphing plugin to select a realization from each
    source model for a requested cluster and optionally apply a Google FILM
    transition between source fields.

    Args:
        cubes (list of Cube):
            Input cubes containing forecast data from one or more source models and
            a cluster cube with mapping attributes produced by
            RealizationClusterAndMatch.
        forecast_period (int):
            Forecast period in seconds used to identify the relevant cluster-source
            mapping.
        cluster_number (int):
            Cluster index to select and return as the processed output realization.
        model_id_attr (str):
            Cube attribute used to identify the source model. Defaults to
            "mosg__model_configuration".
        cycletime (str):
            Forecast reference time to apply to the input cubes. If supplied, the
            forecast periods are updated while validity times remain fixed.
        selection_attr (str):
            Optional cube attribute name to add to the output indicating that this
            realization was selected using the spatial morphing workflow.
        selection_attr_value (str):
            Value assigned to ``selection_attr`` when it is set.
        transitions (dict):
            Explicit transition specification. This may be provided as a JSON
            dictionary containing a "transitions" list, with each entry defining
            "source_a", "source_b", "start_forecast_period_minutes", and
            "end_forecast_period_minutes".
        model_path (str):
            Path to the TensorFlow Hub module used by Google FILM.
        scaling (str):
            Scaling method used by the FILM interpolation step. Supported values are
            "log10" and "minmax".
        clipping_bounds (tuple or dict):
            Optional lower and upper bounds used when clipping interpolated values.
        clip_in_scaled_space (bool):
            If True, clipping is applied before reverse scaling.
        clip_to_physical_bounds (bool):
            If True, clipping is applied after reverse scaling to the physical range.
        max_batch (int):
            Maximum batch size for FILM inference.
        parallel_backend (str):
            Parallel backend to use for FILM inference, or None for serial execution.
        n_workers (int):
            Number of workers used for parallel processing.
        transition_weights_scheme (str):
            Weighting scheme used during the transition, chosen from "linear" or
            "smoothstep".
        apply_quantile_mapping (bool):
            If True, apply quantile mapping to the morphed result using a weighted
            source field.
        occurrence_threshold (float):
            Threshold used by the quantile mapping routine to determine whether a
            value should be mapped.

    Returns:
        Cube:
            Single cube containing the selected realization and any requested
            spatial morphing transition, with realization index set to
            cluster_number.
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
        apply_quantile_mapping=apply_quantile_mapping,
        occurrence_threshold=occurrence_threshold,
    )
    return morphing.process(*cubes)
