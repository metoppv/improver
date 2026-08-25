# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugin for spatial morphing between forecast sources using Google FILM."""

import json
from collections import defaultdict
from typing import Any, Dict, FrozenSet, Optional, Tuple

import numpy as np
from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.clustering.realization_clustering import RealizationSelection
from improver.utilities.temporal import (
    reset_forecast_reference_time_and_period,
    validate_cycletime_format,
)
from improver.utilities.temporal_interpolation import (
    GoogleFilmInterpolation,
    _as_tuple_if_list,
)


class SpatialMorphing(BasePlugin):
    """Spatially morph between forecast sources for a selected realization cluster.

    This plugin builds upon RealizationSelection to select realizations from multiple
    forecast sources according to cluster assignments, then applies spatial morphing
    using Google FILM to create seamless transitions between different source models.

    Unlike hard joins (RealizationSelection alone), this plugin produces spatially
    smooth blended forecasts where different sources contribute smoothly based on
    configured transition characteristics.

    Workflow:
    1. Split input cubes into forecast cubes and cluster cube (from RealizationClusterAndMatch).
    2. Validate that all forecast cubes have the same validity time.
    3. Optionally reset forecast_reference_time based on cycletime.
    4. Parse cluster mapping attributes to identify which realization from which
       forecast source corresponds to the requested cluster.
    5. Extract the selected realizations from each source.
    6. Apply Google FILM spatial morphing to create seamless blended output.

    This plugin is designed to work with output from RealizationClusterAndMatch,
    providing a more direct spatial morphing alternative to the traditional
    RealizationSelection → ForecastTrajectoryGapFiller pipeline.
    """

    def __init__(
        self,
        forecast_period: int,
        cluster_number: int,
        model_id_attr: str = "mosg__model_configuration",
        cycletime: Optional[str] = None,
        selection_attr: Optional[str] = None,
        selection_attr_value: str = "spatial_morphing",
        cluster_sources_attribute: str = "cluster_sources",
        interpolation_window_by_source_pair: Optional[Dict[str, int]] = None,
        interpolation_window_in_minutes: int = 180,
        model_path: Optional[str] = None,
        scaling: str = "minmax",
        clipping_bounds: Optional[Tuple[float, float]] = None,
        clip_in_scaled_space: bool = True,
        clip_to_physical_bounds: bool = False,
        max_batch: Optional[int] = 1,
        parallel_backend: Optional[str] = None,
        n_workers: Optional[int] = 1,
        model_loader: Any = None,
        transition_weights_scheme: str = "linear",
    ) -> None:
        """Initialise the SourceSpatialMorphing plugin.

        Args:
            forecast_period: The forecast period (in seconds) to use for interrogating
                the cluster mapping attributes in order to select the appropriate
                realizations from each forecast source.
            cluster_number: The cluster index (int) to select realizations for.
                Only this cluster will be processed; output will be a single realization
                with this index.
            model_id_attr: The name of the cube attribute used to identify the model
                source. Default: "mosg__model_configuration".
            cycletime: The forecast_reference_time on the input forecast cubes will be
                reset to this value. The forecast periods will be adjusted accordingly
                with the validity times kept fixed. cycletime should be provided in the
                format YYYYMMDDTHHMMZ (e.g., 20240101T0000Z). If not provided, the
                forecast_reference_time on the input cubes will be left unchanged.
            selection_attr: Optional name of a cube attribute to add to the output
                to identify that these realizations were selected using this plugin.
                If not provided (None), no attribute is added. Example:
                "realization_selection_method".
            selection_attr_value: The value (e.g. a description of the selection
                method) to assign to the selection_attr attribute. Default is
                "spatial_morphing". Only used if selection_attr is provided.
            cluster_sources_attribute: Name of the cube attribute containing
                cluster_sources metadata. Default: "cluster_sources".
            interpolation_window_by_source_pair: Optional dictionary mapping
                source-pair keys to transition windows in minutes. Keys must
                identify two source names separated by "|" or ",". Matching is
                order-insensitive.
            interpolation_window_in_minutes: Default transition window in minutes
                (±window around the transition point). Used when no specific
                source-pair window is configured. Default: 180 (3 hours).
            model_path: Path to TensorFlow Hub module for Google FILM model.
                Required if spatial morphing between different sources is performed.
            scaling: Scaling method for FILM interpolation: "log10" or "minmax".
                Default: "minmax".
            clipping_bounds: Optional (min, max) bounds for clipping interpolated data.
            clip_in_scaled_space: If True, clipping applied before reverse scaling.
                Default: True.
            clip_to_physical_bounds: If True, clipping applied after reverse scaling.
                Default: False.
            max_batch: Maximum batch size for FILM inference. Default: 1.
            parallel_backend: Parallelization backend ("loky") or None for serial.
                Default: None.
            n_workers: Number of workers for parallel processing. Default: 1.
            model_loader: Optional callable to load the TensorFlow model.
            transition_weights_scheme: Scheme for computing transition weights:
                "linear" or "smoothstep". Default: "linear".
        """
        self.forecast_period = forecast_period
        self.cluster_number = cluster_number
        self.model_id_attr = model_id_attr
        self.cycletime = cycletime
        if self.cycletime is not None:
            validate_cycletime_format(self.cycletime)
        self.selection_attr = selection_attr
        self.selection_attr_value = selection_attr_value
        self.cluster_sources_attribute = cluster_sources_attribute
        self.interpolation_window_in_minutes = interpolation_window_in_minutes
        self.interpolation_window_by_source_pair = (
            self._parse_interpolation_window_by_source_pair(
                interpolation_window_by_source_pair
            )
        )

        # Store Google FILM config
        self.model_path = model_path
        self.scaling = scaling
        self.clipping_bounds = _as_tuple_if_list(clipping_bounds)
        self.clip_in_scaled_space = clip_in_scaled_space
        self.clip_to_physical_bounds = clip_to_physical_bounds
        self.max_batch = max_batch
        self.parallel_backend = parallel_backend
        self.n_workers = n_workers
        self.model_loader = model_loader
        self.transition_weights_scheme = transition_weights_scheme

        # Create RealizationSelection helper for accessing cluster mapping methods
        self._selection_helper = RealizationSelection(
            forecast_period=forecast_period,
            model_id_attr=model_id_attr,
            cycletime=cycletime,
            selection_attr=selection_attr,
            selection_attr_value=selection_attr_value,
        )

    @staticmethod
    def _prepare_source_pair_key(key: str) -> FrozenSet[str]:
        """Convert a source-pair key into an order-insensitive frozenset.

        Args:
            key: Source-pair key with two source names separated by "|" or ",".

        Returns:
            A frozenset containing two source names.

        Raises:
            ValueError: If the key does not define exactly two non-empty source names.
        """
        delimiter = "|" if "|" in key else ","
        parts = [part.strip() for part in key.split(delimiter)]
        if len(parts) != 2 or any(not part for part in parts):
            raise ValueError(
                "Source-pair key must contain exactly two source names separated "
                f"by '|' or ','. Got: {key}"
            )
        source_pair = frozenset(parts)
        if len(source_pair) != 2:
            raise ValueError(
                f"Source-pair key must contain two distinct source names. Got: {key}"
            )
        return source_pair

    def _parse_interpolation_window_by_source_pair(
        self, interpolation_window_by_source_pair: Optional[Dict[str, int]]
    ) -> Dict[FrozenSet[str], int]:
        """Validate and normalise source-pair windows into seconds.

        Args:
            interpolation_window_by_source_pair:
                Optional dictionary mapping source-pair keys to window minutes.

        Returns:
            A dictionary mapping normalised source-pair keys to window seconds.

        Raises:
            ValueError: If the dictionary is invalid or contains non-positive values.
        """
        if interpolation_window_by_source_pair is None:
            return {}
        if not isinstance(interpolation_window_by_source_pair, dict):
            raise ValueError(
                "interpolation_window_by_source_pair must be a dictionary."
            )

        result = {}
        for key, value in interpolation_window_by_source_pair.items():
            if not isinstance(key, str):
                raise ValueError(
                    "interpolation_window_by_source_pair keys must be strings."
                )
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    "interpolation_window_by_source_pair values must be positive "
                    "integers in minutes."
                )
            result[self._prepare_source_pair_key(key)] = value * 60

        return result

    def _get_transition_window_in_seconds(
        self, sources_before: FrozenSet[str], sources_after: FrozenSet[str]
    ) -> int:
        """Get the transition window for a source pair.

        If interpolation_window_by_source_pair is configured, the window for
        the specific source pair is returned. If not configured, the default
        interpolation_window_in_minutes is used.

        Args:
            sources_before: Forecast sources active before the transition.
            sources_after: Forecast sources active after the transition.

        Returns:
            Window in seconds.
        """
        if self.interpolation_window_by_source_pair:
            source_pair = frozenset(sources_before | sources_after)
            if source_pair in self.interpolation_window_by_source_pair:
                return self.interpolation_window_by_source_pair[source_pair]

        return self.interpolation_window_in_minutes * 60

    def _parse_cluster_sources(self, cube: Cube) -> dict:
        """Parse the cluster_sources dictionary from cube attributes."""
        cluster_sources = cube.attributes.get(self.cluster_sources_attribute)
        if cluster_sources is None:
            return {}

        if isinstance(cluster_sources, str):
            try:
                cluster_sources = json.loads(cluster_sources)
            except json.JSONDecodeError as err:
                raise ValueError(f"Failed to parse cluster sources JSON: {err}")

        if not isinstance(cluster_sources, dict):
            raise ValueError(
                f"Cluster sources attribute must be a dictionary, got {type(cluster_sources)}"
            )

        return cluster_sources

    def _identify_source_transitions(
        self, cluster_sources: dict, realization_index: int
    ) -> list[tuple[int, FrozenSet[str], FrozenSet[str]]]:
        """Identify source transitions for a given realization.

        Returns tuples of (period_before, sources_before, sources_after), where
        period_before is the lead time immediately before the source-set change.
        """
        real_key = str(realization_index)
        if real_key not in cluster_sources:
            return []

        sources_dict = cluster_sources[real_key]
        period_to_sources = defaultdict(set)
        for source_name, periods in sources_dict.items():
            for period in periods:
                period_to_sources[int(period)].add(source_name)

        sorted_periods = sorted(period_to_sources)
        transitions = []
        for period_before, period_after in zip(sorted_periods[:-1], sorted_periods[1:]):
            sources_before = frozenset(period_to_sources[period_before])
            sources_after = frozenset(period_to_sources[period_after])
            if sources_before != sources_after:
                transitions.append((period_before, sources_before, sources_after))
        return transitions

    def _call_google_film_for_morphing(
        self,
        cube_a: Cube,
        cube_b: Cube,
        weight: float,
    ) -> Cube:
        """Use Google FILM to spatially morph between two source cubes.

        Args:
            cube_a: First source cube (weight=0).
            cube_b: Second source cube (weight=1).
            weight: Morphing weight (0.0 to 1.0).

        Returns:
            Morphed cube at the specified weight.

        Raises:
            ValueError: If weight is outside [0, 1] or if FILM config is missing.
        """
        if not (0.0 <= weight <= 1.0):
            raise ValueError(f"Weight must be in [0, 1], got {weight}")

        if self.model_path is None:
            raise ValueError("model_path must be provided to use Google FILM morphing")

        # Create interpolator
        interpolator = GoogleFilmInterpolation(
            model_path=self.model_path,
            scaling=self.scaling,
            clipping_bounds=self.clipping_bounds,
            clip_in_scaled_space=self.clip_in_scaled_space,
            clip_to_physical_bounds=self.clip_to_physical_bounds,
            max_batch=self.max_batch,
            parallel_backend=self.parallel_backend,
            n_workers=self.n_workers,
            model_loader=self.model_loader,
            interpolation_fractions=weight,
        )

        # Create template cube for interpolation result
        template = cube_a.copy()

        # Call FILM with weight as time_fraction
        result_cubes = interpolator.process(cube_a, cube_b, template)

        if len(result_cubes) == 0:
            raise RuntimeError("Google FILM interpolation returned no results")

        return result_cubes[0]

    def _select_single_source_cube(
        self,
        source_name: str,
        realization_index: int,
        forecast_cubes: CubeList,
    ) -> Optional[Cube]:
        """Select a single source cube for a given realization index.

        Args:
            source_name: Source model identifier from model_id_attr.
            realization_index: Realization index to extract.
            forecast_cubes: Input forecast cubes.

        Returns:
            Selected cube, or None if source/realization cannot be extracted.
        """
        try:
            selected = self._selection_helper.select_realizations_for_clusters(
                {self.cluster_number: (source_name, int(realization_index))},
                forecast_cubes,
            )
        except ValueError:
            return None

        if not selected:
            return None
        return selected[0]

    @staticmethod
    def _identify_source_transitions_with_bounds(
        cluster_sources: dict, realization_index: int
    ) -> list[tuple[int, int, FrozenSet[str], FrozenSet[str]]]:
        """Identify source transitions including both sides of the boundary.

        Returns tuples of
        (period_before, period_after, sources_before, sources_after).
        """
        real_key = str(realization_index)
        if real_key not in cluster_sources:
            return []

        sources_dict = cluster_sources[real_key]
        period_to_sources = defaultdict(set)
        for source_name, periods in sources_dict.items():
            for period in periods:
                period_to_sources[int(period)].add(source_name)

        sorted_periods = sorted(period_to_sources)
        transitions = []
        for period_before, period_after in zip(sorted_periods[:-1], sorted_periods[1:]):
            sources_before = frozenset(period_to_sources[period_before])
            sources_after = frozenset(period_to_sources[period_after])
            if sources_before != sources_after:
                transitions.append(
                    (period_before, period_after, sources_before, sources_after)
                )
        return transitions

    def _diagnose_realization_for_source(
        self,
        source_name: str,
        cluster_number: int,
        target_period: int,
        secondary_map: Optional[dict[str, dict[str, list[dict[str, list[int]]]]]],
        primary_map: dict[str, int],
        cluster_cube: Cube,
        full_cluster_to_selection: dict[int, tuple[str, int]],
    ) -> Optional[int]:
        """Diagnose the realization index for a source at/near a target period."""
        cluster_key = str(cluster_number)

        if secondary_map and source_name in secondary_map:
            cluster_entries = secondary_map[source_name].get(cluster_key, [])

            # Prefer exact target-period matches.
            for entry in cluster_entries:
                forecast_periods = [int(period) for period in entry["forecast_periods"]]
                if target_period in forecast_periods:
                    return int(entry["realization"])

            # Otherwise choose the closest mapped period for this source/cluster.
            nearest_realization = None
            nearest_distance = None
            for entry in cluster_entries:
                forecast_periods = [int(period) for period in entry["forecast_periods"]]
                if not forecast_periods:
                    continue
                distance = min(
                    abs(target_period - period) for period in forecast_periods
                )
                if nearest_distance is None or distance < nearest_distance:
                    nearest_distance = distance
                    nearest_realization = int(entry["realization"])
            if nearest_realization is not None:
                return nearest_realization

        # Fallback to the normal nearest-fp cluster selection if it uses this source.
        if cluster_number in full_cluster_to_selection:
            mapped_source, mapped_realization = full_cluster_to_selection[
                cluster_number
            ]
            if mapped_source == source_name:
                return int(mapped_realization)

        # Fallback to primary-map realization if this is the inferred primary model.
        primary_source = (
            self._selection_helper._extract_primary_model_from_cluster_sources(
                cluster_cube
            )
        )
        if source_name == primary_source and cluster_key in primary_map:
            return int(primary_map[cluster_key])

        return None

    def process(self, *cubes: Any) -> Cube:
        """Select realizations from forecast sources and apply spatial morphing.

        Uses RealizationSelection workflow to identify which realization from
        which forecast source corresponds to the requested cluster_number, then
        applies Google FILM spatial morphing if needed to create seamless blends.

        Args:
            *cubes: Input cubes (CubeList or multiple Cube objects) containing
                forecast cubes from different sources (all at same validity time)
                and a cluster cube with mapping attributes from
                RealizationClusterAndMatch.

        Returns:
            Single Cube containing the selected and (if applicable) spatially
            morphed realization, relabelled to cluster_number.

        Raises:
            ValueError: If cluster cube not found, forecast cubes missing, or
                if selected realization cannot be extracted.
            RuntimeError: If Google FILM morphing fails.
        """
        # Step 1: Flatten input then split forecast and cluster cubes.
        if len(cubes) == 1 and isinstance(cubes[0], CubeList):
            cubes = tuple(cubes[0])

        forecast_cubes, cluster_cube = (
            self._selection_helper.split_cubes_forecast_and_cluster(cubes)
        )
        # Hard code for testing.
        cluster_sources = json.loads(
            cluster_cube.attributes.get(self.cluster_sources_attribute)
        )
        cluster_sources["17"]["uk_ens"] = [
            43200,
            86400,
            129600,
            172800,
            216000,
            259200,
            302400,
            345600,
            388800,
            432000,
        ]
        cluster_cube.attributes["cluster_sources"] = json.dumps(cluster_sources)

        # Step 2: Validate all forecast cubes have same validity time
        self._selection_helper.validate_common_validity_time(forecast_cubes)

        # Step 3: Reset forecast reference time if cycletime provided
        if self.cycletime is not None:
            for cube in forecast_cubes:
                reset_forecast_reference_time_and_period(cube, self.cycletime)

        # Step 4: Parse cluster mapping attributes
        primary_map, secondary_map = self._selection_helper.parse_mapping_attributes(
            cluster_cube
        )

        # Step 5: Find nearest secondary mapping forecast period
        mapping_fps = set()
        if secondary_map:
            for cluster_dict in secondary_map.values():
                for cluster_list in cluster_dict.values():
                    for entry in cluster_list:
                        mapping_fps.update(entry["forecast_periods"])

        nearest_fp, use_secondary = (
            self._selection_helper.find_nearest_secondary_mapping_fp(
                mapping_fps, self.forecast_period
            )
        )

        # Step 6: Build cluster-to-selection mapping (only for requested cluster)
        full_cluster_to_selection = self._selection_helper.build_cluster_to_selection(
            nearest_fp, use_secondary, secondary_map, primary_map, cluster_cube
        )

        # Only process the requested cluster
        if self.cluster_number not in full_cluster_to_selection:
            raise ValueError(
                f"Cluster number {self.cluster_number} not found in cluster mapping."
            )

        cluster_to_selection = {
            self.cluster_number: full_cluster_to_selection[self.cluster_number]
        }

        # Step 7: Select the source/realization dictated by cluster mapping for this
        # forecast period. This is the baseline output when no transition morphing is
        # required.
        selected_cubes = self._selection_helper.select_realizations_for_clusters(
            cluster_to_selection, forecast_cubes
        )
        if len(selected_cubes) == 0:
            raise RuntimeError(
                f"No realization selected for cluster {self.cluster_number}"
            )
        result_cube = selected_cubes[0]

        # Step 8: Identify Source A -> Source B transitions for this cluster from
        # cluster_sources, using the same transition semantics as
        # ForecastTrajectoryGapFiller (_identify_source_transitions).
        cluster_sources = self._parse_cluster_sources(cluster_cube)

        transitions = self._identify_source_transitions_with_bounds(
            cluster_sources,
            self.cluster_number,
        )

        # Find the nearest transition window that contains this forecast_period.
        active_transition = None
        smallest_distance = None
        for trans_period, period_after, sources_before, sources_after in transitions:
            if len(sources_before) != 1 or len(sources_after) != 1:
                continue
            source_before = next(iter(sources_before))
            source_after = next(iter(sources_after))
            if source_before == source_after:
                continue

            window_in_seconds = self._get_transition_window_in_seconds(
                sources_before, sources_after
            )
            lower_bound = trans_period - window_in_seconds
            upper_bound = trans_period + window_in_seconds
            if lower_bound <= self.forecast_period <= upper_bound:
                distance = abs(self.forecast_period - trans_period)
                if smallest_distance is None or distance < smallest_distance:
                    smallest_distance = distance
                    active_transition = (
                        trans_period,
                        period_after,
                        source_before,
                        source_after,
                        window_in_seconds,
                    )

        if active_transition is not None:
            (
                trans_period,
                period_after,
                source_a,
                source_b,
                window_in_seconds,
            ) = active_transition

            source_a_realization = self._diagnose_realization_for_source(
                source_name=source_a,
                cluster_number=self.cluster_number,
                target_period=trans_period,
                secondary_map=secondary_map,
                primary_map=primary_map,
                cluster_cube=cluster_cube,
                full_cluster_to_selection=full_cluster_to_selection,
            )
            source_b_realization = self._diagnose_realization_for_source(
                source_name=source_b,
                cluster_number=self.cluster_number,
                target_period=period_after,
                secondary_map=secondary_map,
                primary_map=primary_map,
                cluster_cube=cluster_cube,
                full_cluster_to_selection=full_cluster_to_selection,
            )

            cube_a = self._select_single_source_cube(
                source_a,
                source_a_realization,
                forecast_cubes,
            )
            cube_b = self._select_single_source_cube(
                source_b,
                source_b_realization,
                forecast_cubes,
            )

            if cube_a is not None and cube_b is not None:
                # ForecastTrajectoryGapFiller-style window semantics: +/- window.
                lower_bound = trans_period - window_in_seconds
                upper_bound = trans_period + window_in_seconds
                if self.forecast_period <= lower_bound:
                    weight = 0.0
                elif self.forecast_period >= upper_bound:
                    weight = 1.0
                else:
                    weight = (self.forecast_period - lower_bound) / (
                        upper_bound - lower_bound
                    )
                    weight = float(np.clip(weight, 0.0, 1.0))

                # Apply smoothstep to weight for smoother transition
                if self.transition_weights_scheme == "smoothstep":
                    weight = weight * weight * (3.0 - 2.0 * weight)

                if weight <= 0.0:
                    result_cube = cube_a
                elif weight >= 1.0:
                    result_cube = cube_b
                else:
                    result_cube = self._call_google_film_for_morphing(
                        cube_a,
                        cube_b,
                        weight,
                    )

        # Ensure the output realization coordinate matches the requested cluster.
        if result_cube.coords("realization"):
            result_cube.coord("realization").points = [self.cluster_number]
            result_cube.coord("realization").units = "1"

        result_cube.attributes.pop(self.model_id_attr, None)

        # Step 9: Add selection attribute if requested
        if self.selection_attr is not None:
            result_cube.attributes[self.selection_attr] = self.selection_attr_value

        return result_cube
