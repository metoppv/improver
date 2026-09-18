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
    transition_weights_scheme: str = "linear",
    morphing_method: str = "google_film",
    apply_suppression: bool = False,
    suppression_config: cli.inputjson = None,
    suppression_stages: cli.comma_separated_list = None,
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
        transition_weights_scheme (str):
            Weighting scheme used during the transition, chosen from "linear" or
            "smoothstep".
        morphing_method (str):
            Spatial morphing backend to use for transitions. Supported values are
            "google_film" (default) and "linear".
        apply_suppression (bool):
            If True, apply the local suppression workflow to the morphed result.
            This workflow is intended for precipitation-like fields only, because
            it uses wet-occurrence, local-concentration, and upper-tail intensity
            diagnostics that are meaningful for precipitation and not for general
            scalar meteorological variables.
        suppression_config (dict or None):
            Optional JSON dictionary containing tuning values for the local
            suppression stages applied to the morphed field, including the wet
            occurrence_threshold. This configuration is intended for precipitation
            diagnostics only. You can provide a partial dictionary and omit keys
            you do not want to tune; unspecified settings use built-in defaults.
            For example:
            {"occurrence_threshold": 0.5, "maximum_suppression": 0.6,
            "sigmoid_clip_limit": 20.0}
        suppression_stages (list or None):
            Optional comma-separated list of suppression stages to apply. Supported
            values are weak_signal, convective, and upper_tail. These stages are
            designed for precipitation diagnostics and are not generally suitable
            for non-precipitation fields.

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
        transition_weights_scheme=transition_weights_scheme,
        morphing_method=morphing_method,
        apply_suppression=apply_suppression,
        suppression_config=suppression_config,
        suppression_stages=suppression_stages,
    )
    return morphing.process(*cubes)
