#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to fill gaps in the forecast trajectory using temporal interpolation."""

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
    clipping_bounds: cli.comma_separated_list = None,
    clip_in_scaled_space: bool = False,
    clip_to_physical_bounds: bool = False,
):
    """Fill gaps in the forecast trajectory using temporal interpolation.

    This CLI identifies missing points in a forecast trajectory (i.e. gaps in
    validity time, or equivalently forecast period, for a fixed forecast_reference_time)
    and fills them using temporal interpolation. Optionally, it can regenerate points
    at forecast source transitions when cluster_sources configuration is provided.

    The intended input is an iris Cube or CubeList where the
    forecast_reference_time coordinate is fixed, and the time and
    forecast_period coordinates are increasing. The expectation is that the time
    and forecast_period coordinates are associated with each other, so share a
    dimension if the input is an iris Cube. This CLI will fill gaps in the
    forecast trajectory, i.e. fill gaps in the validity time, or equivalently,
    the forecast_period.

    Args:
        cubes (iris.cube.CubeList or iris.cube.Cube):
            An iris Cube or CubeList with a fixed forecast_reference_time
            coordinate, and increasing time and forecast_period coordinates. The
            time and forecast_period coordinates should be associated (share a
            dimension if a Cube). The input may have missing points in the
            forecast trajectory.
        interval_in_mins (int):
            The expected interval between points in the forecast trajectory (in
            minutes). Used to identify gaps in the sequence. If not provided,
            gaps will not be filled, but points can still be regenerated if
            cluster_sources_attribute is set.
        interpolation_method (str):
            ["linear", "solar", "daynight", "google_film"]
            Method of interpolation to use. Default is "linear".
            - "linear": Standard linear interpolation
            - "solar": Interpolation scaled by solar elevation angle
            - "daynight": Linear interpolation with night-time values set to zero
            - "google_film": Deep learning model for frame interpolation
        cluster_sources_attribute (str):
            Name of cube attribute containing cluster sources dictionary. This
            dictionary maps realization indices to their forecast sources and
            forecast periods. These forecast periods are for a specific forecast
            trajectory (i.e. with a set of validity times increasing into the future
            from a fixed forecast_reference_time). When provided with
            interpolation_window_in_hours, enables identification and regeneration of
            forecast periods at source transitions. Format: {realization_index: {source_name:
            [periods]}}
        interpolation_window_in_hours (int):
            Time window (in hours) to use as a +/- range around forecast source
            transition points. Used with cluster_sources_attribute to identify
            which forecast periods should be regenerated. For example, if set to
            3 hours and a transition occurs at a given period, periods 3 hours
            before, at, and after the transition will be regenerated if they
            fall within the sequence.
        model_path (str):
            Path to the TensorFlow Hub module for the Google FILM model.
            Required when interpolation_method is "google_film". Can be a local
            path or a TensorFlow Hub URL.
        scaling (str):
            Scaling method to apply to the data before interpolation when using
            "google_film" method. Options are "log10" or "minmax". Default is
            "minmax".
        clipping_bounds (str):
            Comma-separated lower and upper bounds for clipping interpolated
            values when using "google_film" method. E.g. "0.0,1.0". Default is
            None.
        clip_in_scaled_space (bool):
            If True, apply clipping to the interpolated data while still in
            scaled space (i.e. before reversing any scaling). If False, no clipping
            is applied at this stage. Default is False.
        clip_to_physical_bounds (bool):
            If True, apply clipping to the interpolated data after reversing any
            scaling using the physical bounds. If clipping_bounds are supplied these
            are used, otherwise the min and max values from the input cubes are used.
            If False, no clipping is applied at this stage. Default is False.

    Returns:
        iris.cube.Cube:
            A single merged cube with all points in the forecast trajectory
            filled. The cube will have time as a dimension coordinate and will
            include:
            - All original time slices (except those regenerated at transitions)
            - Interpolated slices filling identified gaps
            - Regenerated slices at source transitions (if configured)
    """
    from improver.utilities.temporal_interpolation import ForecastTrajectoryGapFiller

    plugin = ForecastTrajectoryGapFiller(
        interval_in_minutes=interval_in_mins,
        interpolation_method=interpolation_method,
        cluster_sources_attribute=cluster_sources_attribute,
        interpolation_window_in_hours=interpolation_window_in_hours,
        model_path=model_path,
        scaling=scaling,
        clipping_bounds=clipping_bounds,
        clip_in_scaled_space=clip_in_scaled_space,
        clip_to_physical_bounds=clip_to_physical_bounds,
    )

    result = plugin.process(*cubes)

    return result
