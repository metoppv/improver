# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Utilities for parsing and querying cluster_sources cube attributes."""

import json
from typing import Optional


def parse_cluster_sources_attribute(cube) -> dict:
    """Parse the cluster_sources dictionary from a cube attribute.

    The cluster_sources attribute is typically set by RealizationClusterAndMatch
    and consumed by RealizationSelection, StochasticNoise, and temporal
    interpolation plugins to determine which forecast source (model) was used
    for each cluster at each forecast period.

    Args:
        cube:
            A cube containing the cluster_sources attribute as a JSON string.

    Returns:
        Dictionary mapping cluster indices (as strings) to model mappings.
        Format: {cluster_idx: {model_name: [forecast_periods_in_seconds]}}

    Raises:
        ValueError: If cluster_sources JSON string cannot be parsed.
    """
    cluster_sources_attr = cube.attributes.get("cluster_sources")
    if cluster_sources_attr is None:
        return {}

    # Parse JSON string if needed
    if isinstance(cluster_sources_attr, str):
        try:
            return json.loads(cluster_sources_attr)
        except json.JSONDecodeError as err:
            raise ValueError(f"Failed to parse cluster_sources JSON: {err}")
    elif isinstance(cluster_sources_attr, dict):
        return cluster_sources_attr
    else:
        raise ValueError(
            f"cluster_sources attribute must be a dictionary or JSON string, "
            f"got {type(cluster_sources_attr)}"
        )


def find_nearest_forecast_period_gte(
    forecast_periods: Optional[set[int]], target_fp: int
) -> tuple[int, bool]:
    """Find the nearest forecast period >= target_fp from a set of forecast periods.

    This is used for transition handling when querying secondary mappings: if an
    exact forecast period is not available, we find the next earliest one.

    Args:
        forecast_periods:
            Set of available forecast periods (in seconds).
            If None or empty, returns (target_fp, False).
        target_fp:
            The target forecast period (in seconds) to find.

    Returns:
        Tuple of (nearest_fp, use_secondary):
            - nearest_fp: The smallest forecast period from forecast_periods that is
              >= target_fp, or target_fp if none exist or forecast_periods is empty.
            - use_secondary: Boolean indicating whether the nearest_fp is from
              forecast_periods (True) or equals target_fp (False).
    """
    if forecast_periods:
        valid_fps = [fp for fp in forecast_periods if fp >= target_fp]
        if valid_fps:
            return min(valid_fps), True
        return target_fp, False
    return target_fp, False


def get_source_for_forecast_period(
    cluster_sources: dict, realization_idx: int, fp_seconds: int
) -> Optional[str]:
    """Determine which forecast source (model) is active for a given realization and
    forecast period.

    Uses transition handling: if the exact forecast period is not found in
    cluster_sources for this realization, we look for the largest period
    that is still <= the target period (conservative: stay with last-known source).

    Args:
        cluster_sources:
            Dictionary mapping cluster/realization indices (as strings or ints)
            to model sources. Format: {realization_idx: {model_name: [fp1, fp2, ...]}}
        realization_idx:
            The realization/cluster index to query.
        fp_seconds:
            The forecast period in seconds.

    Returns:
        The name of the forecast source (model) active at this forecast period,
        or None if the realization is not found or no source covers this period.

    Example:
        >>> sources = {
        ...     "0": {
        ...         "uk_ens": [3600, 21600, 43200, 86400, 129600, 172800],
        ...         "gl_ens": [432000, 475200, 518400],
        ...     }
        ... }
        >>> get_source_for_forecast_period(sources, "0", 90000)
        'uk_ens'
        >>> get_source_for_forecast_period(sources, "0", 432000)
        'gl_ens'
        >>> get_source_for_forecast_period(
        ...     sources, "0", 450000
        ... )  # Between uk_ens end & gl_ens start
        'uk_ens'  # Returns last-known source
    """
    # Convert to string key in case int was provided
    real_key = str(realization_idx)

    if real_key not in cluster_sources:
        return None

    model_dict = cluster_sources[real_key]

    # Exact match: check if fp_seconds is in any model's period list
    for model_name, periods in model_dict.items():
        if fp_seconds in periods:
            return model_name

    # Transition handling: find largest period <= fp_seconds (last-known source)
    # This ensures conservative behavior at model boundaries
    best_model = None
    best_period = -1

    for model_name, periods in model_dict.items():
        for period in periods:
            if period <= fp_seconds and period > best_period:
                best_period = period
                best_model = model_name

    return best_model
