#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to fill gaps in forecast period sequences using temporal interpolation."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube,
    interval_in_mins: int = None,
    interpolation_method: str = "linear",
    cluster_sources_attribute: str = None,
    interpolation_window_in_hours: int = None,
    model_path: str = None,
    scaling: str = "minmax",
    clipping_bounds: cli.comma_separated_list = (0.0, 1.0),
):
    """Fill gaps in forecast period sequences using temporal interpolation.

    Identifies missing forecast periods in a sequence and fills them using
    temporal interpolation. Can optionally regenerate periods at forecast
    source transitions when cluster_sources configuration is provided.

    Args:
        cubes (iris.cube.CubeList or iris.cube.Cube):
            A cube or cubelist containing cubes with potentially missing
            forecast periods. Can be a single Cube with forecast_period as
            a dimension coordinate (will be sliced automatically), or multiple
            individual cubes representing different forecast periods. All cubes
            should have forecast_period coordinates and represent the same
            diagnostic at different forecast periods.
        interval_in_mins (int):
            The expected interval between forecast periods in minutes.
            Used to identify gaps in the sequence. If not provided, gaps
            will not be filled, but periods can still be regenerated if
            cluster_sources_attribute is set.
        interpolation_method (str):
            ["linear", "solar", "daynight", "google_film"]
            Method of interpolation to use. Default is "linear".
            - "linear": Standard linear interpolation
            - "solar": Interpolation scaled by solar elevation angle
            - "daynight": Linear interpolation with night-time values set to zero
            - "google_film": Deep learning model for frame interpolation
        cluster_sources_attribute (str):
            Name of cube attribute containing cluster sources dictionary.
            This dictionary maps realization indices to their forecast sources
            and periods. When provided with interpolation_window_in_hours,
            enables identification and regeneration of forecast periods at
            source transitions. Format: {realization_index: {source_name: [periods]}}
        interpolation_window_in_hours (int):
            Time window (in hours) to use as a +/- range around forecast
            source transition points. Used with cluster_sources_attribute
            to identify which periods should be regenerated. For example,
            if set to 3 hours and a transition occurs at T+24, periods
            T+21, T+24, and T+27 will be regenerated if they fall within
            the sequence.
        model_path (str):
            Path to the TensorFlow Hub module for the Google FILM model.
            Required when interpolation_method is "google_film". Can be a
            local path or a TensorFlow Hub URL.
        scaling (str):
            Scaling method to apply to the data before interpolation when
            using "google_film" method. Options are "log10" or "minmax".
            Default is "minmax".
        clipping_bounds (str):
            Comma-separated lower and upper bounds for clipping interpolated
            values when using "google_film" method. E.g. "0.0,1.0".
            Default is "0.0,1.0".

    Returns:
        iris.cube.Cube:
            A single merged cube with all forecast periods filled. The cube
            will have time as a dimension coordinate and will include:
            - All original time slices (except those regenerated at transitions)
            - Interpolated slices filling identified gaps
            - Regenerated slices at source transitions (if configured)
    """
    from improver.utilities.temporal_interpolation import ForecastPeriodGapFiller

    plugin = ForecastPeriodGapFiller(
        interval_in_minutes=interval_in_mins,
        interpolation_method=interpolation_method,
        cluster_sources_attribute=cluster_sources_attribute,
        interpolation_window_in_hours=interpolation_window_in_hours,
        model_path=model_path,
        scaling=scaling,
        clipping_bounds=clipping_bounds,
    )

    result = plugin.process(*cubes)

    return result
